import random
import numpy as np
import torch
import torch.nn as nn
from common.abstract_recommender import GeneralRecommender
import scipy.sparse as sp

from utils.utils import FlowType

class LightGCN_Encoder(GeneralRecommender):
    def __init__(self, config, dataset):
        super(LightGCN_Encoder, self).__init__(config, dataset)
        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(
            form='coo').astype(np.float32)

        self.user_count = self.n_users
        self.item_count = self.n_items
        self.latent_size = config['embedding_size']
        self.n_layers = 3 if config['n_layers'] is None else config['n_layers']
        self.layers = [self.latent_size] * self.n_layers

        self.drop_ratio = 1.0
        self.drop_flag = True

        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = self.get_norm_adj_mat().to(self.device)

        # svd augment
        self.user_augment = nn.Parameter(torch.zeros(self.user_count, self.latent_size), requires_grad=False)
        self.item_augment = nn.Parameter(torch.zeros(self.item_count, self.latent_size), requires_grad=False)
        # svd flaw
        self.user_flaw = nn.Parameter(torch.zeros(self.user_count, self.latent_size), requires_grad=False)
        self.item_flaw = nn.Parameter(torch.zeros(self.item_count, self.latent_size), requires_grad=False)

        # for svd
        self.svd_row = self.get_svd_row(self.latent_size)
        self.all_pos_samples_4u = dataset.history_items_per_u
        self.config = dataset.config
        self.flaw = []

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.user_count, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.item_count, self.latent_size)))
        })

        return embedding_dict

    def get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        # build adj matrix
        A = sp.dok_matrix((self.n_users + self.n_items,
                           self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col+self.n_users),
                             [1]*inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row+self.n_users, inter_M_t.col),
                                  [1]*inter_M_t.nnz)))
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
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def sparse_dropout(self, x, rate, noise_shape):
        random_tensor = 1 - rate
        random_tensor += torch.rand(noise_shape).to(self.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(self.device)
        return out * (1. / (1 - rate))
    
    def get_svd_row(self, x):
        # keep row_num larger than col_cum for matrix to be SVDed
        for i in range(int(np.sqrt(x)), x):
            if x % i == 0:
                return i
        return x

    def svd_decomposition(self, matrix_to_be_SVDed):
        matrix_to_be_SVDed = matrix_to_be_SVDed.view(self.n_users + self.n_items, self.svd_row, -1)
        U, S, V = torch.linalg.svd(matrix_to_be_SVDed)
        S = torch.diag_embed(S)
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
        return torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)

    def get_flaw_item(self, user_all_aug_copy, item_all_aug_copy, users_in_record):
        flaw_item = []
        scores = torch.matmul(user_all_aug_copy, item_all_aug_copy.transpose(0, 1))
        _, topk_index = torch.topk(scores, max(self.config['topk']), dim=-1)
        for u in users_in_record:
            cnt = -1
            while True:
                cnt += 1
                flaw_idx = random.sample(topk_index[u].tolist(), 1)[0]
                if flaw_idx not in self.all_pos_samples_4u[u] or cnt == max(self.config['topk']) * 2:
                    flaw_item.append(flaw_idx)
                    break
        return flaw_item

    def forward(self, inputs, flow):
        A_hat = self.sparse_dropout(self.sparse_norm_adj,
                                    np.random.random() * self.drop_ratio,
                                    self.sparse_norm_adj._nnz()) if self.drop_flag else self.sparse_norm_adj

        # train flow
        ego_embeddings = self.get_ego_embeddings()
        if flow is FlowType.augment:
            ego_embeddings = (ego_embeddings + self.get_ego_augment()) / 2
        elif flow is FlowType.flaw:
            ego_embeddings = (ego_embeddings + self.get_ego_flaw()) / 2
        else:
            assert flow is FlowType.normal

        all_embeddings = [ego_embeddings]

        for k in range(len(self.layers)):
            ego_embeddings = torch.sparse.mm(A_hat, ego_embeddings)
            all_embeddings += [ego_embeddings]

        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)

        user_all_embeddings = all_embeddings[:self.user_count, :]
        item_all_embeddings = all_embeddings[self.user_count:, :]

        users, items = inputs[0], inputs[1]

        if FlowType.augment:
            self.flaws = self.get_flaw_item(
                user_all_aug_copy=user_all_embeddings.clone().detach(),
                item_all_aug_copy=item_all_embeddings.clone().detach(),
                users_in_record=users.tolist()
            )
        elif FlowType.flaw:
            items = self.flaws

        user_embeddings = user_all_embeddings[users, :]
        item_embeddings = item_all_embeddings[items, :]

        return user_embeddings, item_embeddings

    @torch.no_grad()
    def get_embedding(self):
        A_hat = self.sparse_norm_adj

        # train flow
        ego_embeddings = self.get_ego_embeddings()
        all_embeddings = [ego_embeddings]

        for k in range(len(self.layers)):
            ego_embeddings = torch.sparse.mm(A_hat, ego_embeddings)
            all_embeddings += [ego_embeddings]

        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)

        user_all_embeddings = all_embeddings[:self.user_count, :]
        item_all_embeddings = all_embeddings[self.user_count:, :]

        return user_all_embeddings, item_all_embeddings
