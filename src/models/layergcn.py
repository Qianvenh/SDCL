import numpy as np
import scipy.sparse as sp
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, L2Loss

from utils.utils import FlowType, TrainState

class LayerGCN(GeneralRecommender):
    def __init__(self, config, dataset):
        super(LayerGCN, self).__init__(config, dataset)

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(
            form='coo').astype(np.float32)

        # load parameters info
        self.latent_dim = config['embedding_size']  # int type:the embedding size of lightGCN
        self.n_layers = config['n_layers']  # int type:the layer num of lightGCN
        self.reg_weight = config['reg_weight']  # float32 type: the weight decay for l2 normalizaton
        self.dropout = config['dropout']

        self.n_nodes = self.n_users + self.n_items

        # define layers and loss
        self.user_embeddings = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_users, self.latent_dim)))
        self.item_embeddings = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_items, self.latent_dim)))
        # svd augment
        self.user_augment = nn.Parameter(torch.zeros(self.n_users, self.latent_dim), requires_grad=False)
        self.item_augment = nn.Parameter(torch.zeros(self.n_items, self.latent_dim), requires_grad=False)
        # svd flaw
        self.user_flaw = nn.Parameter(torch.zeros(self.n_users, self.latent_dim), requires_grad=False)
        self.item_flaw = nn.Parameter(torch.zeros(self.n_items, self.latent_dim), requires_grad=False)

        # for svd
        self.svd_row = self.get_svd_row(self.latent_dim)
        self.all_pos_samples_4u = dataset.history_items_per_u
        self.config = dataset.config

        # normalized adj matrix
        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)
        self.masked_adj = None
        self.forward_adj = None
        self.pruning_random = False

        # edge prune
        self.edge_indices, self.edge_values = self.get_edge_info()

        self.mf_loss = BPRLoss()
        self.reg_loss = L2Loss()

    # def post_epoch_processing(self):
    #     with torch.no_grad():
    #         return '=== Layer weights: {}'.format(F.softmax(self.layer_weights.exp(), dim=0))

    def pre_epoch_processing(self):
        if self.dropout <= .0:
            self.masked_adj = self.norm_adj_matrix
            return
        keep_len = int(self.edge_values.size(0) * (1. - self.dropout))
        if self.pruning_random:
            # pruning randomly
            keep_idx = torch.tensor(random.sample(range(self.edge_values.size(0)), keep_len))
        else:
            # pruning edges by pro
            keep_idx = torch.multinomial(self.edge_values, keep_len)         # prune high-degree nodes
        self.pruning_random = True ^ self.pruning_random
        keep_indices = self.edge_indices[:, keep_idx]
        # norm values
        keep_values = self._normalize_adj_m(keep_indices, torch.Size((self.n_users, self.n_items)))
        all_values = torch.cat((keep_values, keep_values))
        # update keep_indices to users/items+self.n_users
        keep_indices[1] += self.n_users
        all_indices = torch.cat((keep_indices, torch.flip(keep_indices, [0])), 1)
        self.masked_adj = torch.sparse.FloatTensor(all_indices, all_values, self.norm_adj_matrix.shape).to(self.device)

    def _normalize_adj_m(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        col_sum = 1e-7 + torch.sparse.sum(adj.t(), -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        c_inv_sqrt = torch.pow(col_sum, -0.5)
        cols_inv_sqrt = c_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return values

    def get_edge_info(self):
        rows = torch.from_numpy(self.interaction_matrix.row)
        cols = torch.from_numpy(self.interaction_matrix.col)
        edges = torch.stack([rows, cols]).type(torch.LongTensor)
        # edge normalized values
        values = self._normalize_adj_m(edges, torch.Size((self.n_users, self.n_items)))
        return edges, values

    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_users + self.n_items,
                           self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users),
                             [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col),
                                  [1] * inter_M_t.nnz)))
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid Devide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor([row, col])
        data = torch.FloatTensor(L.data)

        return torch.sparse.FloatTensor(i, data, torch.Size((self.n_nodes, self.n_nodes)))

    def get_svd_row(self, x):
        # keep row_num larger than col_cum for matrix to be SVDed
        for i in range(int(np.sqrt(x)), x):
            if x % i == 0:
                return i
        return x

    def svd_decomposition(self, matrix_to_be_SVDed):
        r"""
            U has size: (26495, 8, 8)
            S has size: (26496, 8)
            V has size: (26496, 8, 8)
        """
        matrix_to_be_SVDed = matrix_to_be_SVDed.view(self.n_users + self.n_items, self.svd_row, -1)
        U, S, V = torch.linalg.svd(matrix_to_be_SVDed)
        S = torch.diag_embed(S) # size: (26495, 8, 8)
        zeros_to_fill = torch.zeros((S.shape[0], U.shape[2] - S.shape[1], S.shape[2])).to(self.device)
        S = torch.cat((S, zeros_to_fill), dim=1)
        return U, S, V

    def init_ego_augment_and_flaw(self):
    
        U, S, V = self.svd_decomposition(self.get_ego_embeddings().clone().detach())
        
        all_augment = torch.bmm(U, S).view(S.shape[0], -1)
        self.user_augment.data, self.item_augment.data = \
            torch.split(all_augment, [self.n_users, self.n_items])
        self.user_augment.requires_grad = self.item_augment.requires_grad = True

        all_flaw = torch.bmm(S, V).view(S.shape[0], -1)
        self.user_flaw.data, self.item_flaw.data = \
            torch.split(all_flaw, [self.n_users, self.n_items])
        self.user_flaw.requires_grad = self.item_flaw.requires_grad = True

    def get_ego_augment(self):
        return torch.cat([self.user_augment, self.item_augment], 0)
    
    def get_ego_flaw(self):
        return torch.cat([self.user_flaw, self.item_flaw], 0)

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.
        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        return torch.cat([self.user_embeddings, self.item_embeddings], 0) # size: (26495, 64)

    def forward(self, flow):
        # train flow
        ego_embeddings = self.get_ego_embeddings()
        if flow is FlowType.augment:
            ego_embeddings = (ego_embeddings + self.get_ego_augment()) / 2
        elif flow is FlowType.flaw:
            ego_embeddings = (ego_embeddings + self.get_ego_flaw()) / 2
        else:
            assert flow is FlowType.normal

        all_embeddings = ego_embeddings
        embeddings_layers = []

        for layer_idx in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.forward_adj, all_embeddings)
            _weights = F.cosine_similarity(all_embeddings, ego_embeddings, dim=-1)
            all_embeddings = torch.einsum('a,ab->ab', _weights, all_embeddings)
            embeddings_layers.append(all_embeddings)

        ui_all_embeddings = torch.sum(torch.stack(embeddings_layers, dim=0), dim=0)
        user_all_embeddings, item_all_embeddings = torch.split(ui_all_embeddings, [self.n_users, self.n_items])
        return user_all_embeddings, item_all_embeddings

    def bpr_loss(self, u_embeddings, i_embeddings, user, pos_item, neg_item):
        posi_embeddings = i_embeddings[pos_item]
        negi_embeddings = i_embeddings[neg_item]

        # calculate BPR Loss
        pos_scores = torch.mul(u_embeddings[user], posi_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings[user], negi_embeddings).sum(dim=1)
        m = torch.nn.LogSigmoid()
        bpr_loss = torch.sum(-m(pos_scores - neg_scores))
        #mf_loss = self.mf_loss(pos_scores, neg_scores)

        return bpr_loss

    def emb_loss(self, u_embeddings, i_embeddings, user, pos_item, neg_item):
        # calculate BPR Loss
        u_ego_embeddings = u_embeddings[user]
        posi_ego_embeddings = i_embeddings[pos_item]
        negi_ego_embeddings = i_embeddings[neg_item]

        reg_loss = self.reg_loss(u_ego_embeddings, posi_ego_embeddings, negi_ego_embeddings)
        return reg_loss

    def get_flaw_item(self, user_all_aug_copy, item_all_aug_copy, user_list):
        flaw_item = []
        scores = torch.matmul(user_all_aug_copy, item_all_aug_copy.transpose(0, 1))
        _, topk_index = torch.topk(scores, max(self.config['topk']), dim=-1)
        for u in user_list:
            cnt = 0
            while True:
                cnt += 1
                flaw_idx = random.sample(topk_index[u].tolist(), 1)[0]
                if flaw_idx not in self.all_pos_samples_4u[u] or cnt == max(self.config['topk']) * 2:
                    flaw_item.append(flaw_idx)
                    break
        return flaw_item

    def calculate_loss(self, interaction, state=TrainState.normal): # 很重要
        user = interaction[0]
        pos_item = interaction[1]
        neg_item = interaction[2]

        self.forward_adj = self.masked_adj

        if state is TrainState.normal:
            user_all_embeddings, item_all_embeddings = self.forward(FlowType.normal)
            mf_loss = self.bpr_loss(user_all_embeddings, item_all_embeddings, user, pos_item, neg_item)
            reg_loss = self.emb_loss(self.user_embeddings, self.item_embeddings, user, pos_item, neg_item)
            return mf_loss + self.reg_weight * reg_loss

        assert state is TrainState.dual_flow

        # loss for svd_augment
        user_all_aug_emb, item_all_aug_emb = self.forward(FlowType.augment)
        mf_loss = self.bpr_loss(user_all_aug_emb, item_all_aug_emb, user, pos_item, neg_item)
        reg_loss = self.emb_loss((self.user_embeddings + self.user_augment) / 2, (self.item_embeddings + self.item_augment) / 2, user, pos_item, neg_item)
        loss1 = mf_loss + self.reg_weight * reg_loss

        # # get flaw items
        flaw_item = self.get_flaw_item(
                      user_all_aug_copy=user_all_aug_emb.clone().detach(),
                      item_all_aug_copy=item_all_aug_emb.clone().detach(),
                      user_list=user.tolist()
                      )

        # # loss for svd_flaw
        user_all_flaw_emb, item_all_flaw_emb = self.forward(FlowType.flaw)
        mf_flaw_loss = self.bpr_loss(user_all_flaw_emb, item_all_flaw_emb, user, flaw_item, neg_item)
        reg_flaw_loss = self.emb_loss((self.user_embeddings + self.user_flaw) / 2, (self.item_embeddings + self.item_flaw) / 2, user, flaw_item, neg_item)
        loss2 = mf_flaw_loss + self.reg_weight * reg_flaw_loss

        
        return loss1 + loss2

    def full_sort_predict(self, interaction):
        user = interaction[0]

        self.forward_adj = self.norm_adj_matrix
        
        restore_user_e, restore_item_e = self.forward(FlowType.normal)
        u_embeddings = restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores


