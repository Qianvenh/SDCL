r"""
################################################
Self-supervised CF

Using the same implementation of LightGCN in BUIR
Adding regularization on embeddings


SELFCF_{ed}: embedding dropout
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# encoder_svd
from common.encoder_svd_Selfcfed import LightGCN_Encoder

from common.abstract_recommender import GeneralRecommender
from common.loss import L2Loss

from utils.utils import TrainState, FlowType

class SELFCFED_LGN(GeneralRecommender):
    def __init__(self, config, dataset):
        super(SELFCFED_LGN, self).__init__(config, dataset)
        self.user_count = self.n_users
        self.item_count = self.n_items
        self.latent_size = config['embedding_size']
        self.dropout = config['dropout']
        self.reg_weight = config['reg_weight']

        self.online_encoder = LightGCN_Encoder(config, dataset)
        self.predictor = nn.Linear(self.latent_size, self.latent_size)
        self.reg_loss = L2Loss()

    def init_ego_augment_and_flaw(self):
        self.online_encoder.init_ego_augment_and_flaw()

    def forward(self, inputs, flow):
        u_online, i_online = self.online_encoder(inputs, flow)
        with torch.no_grad():
            u_target, i_target = u_online.clone(), i_online.clone()
            u_target.detach()
            i_target.detach()
            u_target = F.dropout(u_target, self.dropout)
            i_target = F.dropout(i_target, self.dropout)

        return u_online, u_target, i_online, i_target

    @torch.no_grad()
    def get_embedding(self):
        u_online, i_online = self.online_encoder.get_embedding()
        return self.predictor(u_online), u_online, self.predictor(i_online), i_online

    def loss_fn(self, p, z):  # negative cosine similarity
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()

    def construct_loss(self, u_online, u_target, i_online, i_target):
        reg_loss = self.reg_loss(u_online, i_online)
        u_online, i_online = self.predictor(u_online), self.predictor(i_online)     
        loss_ui = self.loss_fn(u_online, i_target) / 2
        loss_iu = self.loss_fn(i_online, u_target) / 2
        return loss_ui + loss_iu + self.reg_weight * reg_loss

    def calculate_loss(self, interaction, state=TrainState.normal):
        if state is TrainState.normal:
            u_online, u_target, i_online, i_target = self.forward(interaction, flow=FlowType.normal)
            loss = self.construct_loss(u_online, u_target, i_online, i_target)
            return loss

        assert state is TrainState.dual_flow

        u_aug_online, u_aug_target, i_aug_online, i_aug_target = self.forward(interaction, flow=FlowType.augment)
        loss1 = self.construct_loss(u_aug_online, u_aug_target, i_aug_online, i_aug_target)

        u_flaw_online, u_flaw_target, i_flaw_online, i_flaw_target = self.forward(interaction, flow=FlowType.flaw)
        loss2 = self.construct_loss(u_flaw_online, u_flaw_target, i_flaw_online, i_flaw_target)

        return loss1 + loss2

    def full_sort_predict(self, interaction):
        user = interaction[0]
        u_online, u_target, i_online, i_target = self.get_embedding()
        score_mat_ui = torch.matmul(u_online[user], i_target.transpose(0, 1))
        score_mat_iu = torch.matmul(u_target[user], i_online.transpose(0, 1))
        scores = score_mat_ui + score_mat_iu

        return scores
