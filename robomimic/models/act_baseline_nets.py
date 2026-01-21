import math
from typing import Union, Optional, List
import copy
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn, Tensor
import numpy as np



def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")




def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )



def build_encoder(args):
    d_model = args.hidden_dim # 256
    dropout = args.dropout # 0.1
    nhead = args.nheads # 8
    dim_feedforward = args.dim_feedforward # 2048
    num_encoder_layers = args.enc_layers # 4 # TODO shared with VAE decoder
    normalize_before = args.pre_norm # False
    activation = "relu"

    encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                            dropout, activation, normalize_before)
    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
    encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    return encoder




################################################################################
####                              Transformer                               ####
################################################################################



class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


################################

class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):

        assert len(src.shape) == 3
        bs, c, _ = src.shape
        src = src.permute(2, 0, 1)  ##  (1, bs, h_dim)
        pos_embed = pos_embed.permute((2,0,1)).repeat(1, bs, 1)  ## (1, bs, h_dim)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)  ## (chunk, bs, h_dim)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        hs = hs.transpose(1, 2)
        return hs


################################################################################
####                              ACT main model                            ####
################################################################################



class ACTBaselineNet(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self,
                action_dim, 
                chunk_size=16,
                cond_dim=146, 
                ):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()

        ## set arguments
        args = SimpleNamespace()
        args.hidden_dim = 512
        args.dropout = 0.1
        args.dim_feedforward = 2048
        args.nheads = 8
        args.enc_layers = 4
        args.dec_layers = 7
        args.pre_norm = False
        args.activation = "relu"
        args.return_intermediate_dec = True

        self.num_queries = chunk_size
        self.transformer = build_transformer(args)
        self.encoder = build_encoder(args)
        hidden_dim = self.transformer.d_model

        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(chunk_size, hidden_dim)

        self.latent_dim = 32
        self.cls_embed = nn.Embedding(1, hidden_dim)
        self.encoder_action_proj = nn.Linear(action_dim, hidden_dim)
        self.encoder_obs_proj = nn.Linear(cond_dim, hidden_dim)
        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim * 2)
        self.register_buffer("input_pos_table", get_sinusoid_encoding_table(1 + 1 + chunk_size, hidden_dim))

        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim)

        self.dec_pos = nn.Parameter(torch.zeros(1, hidden_dim, 2))
        nn.init.normal_(self.dec_pos, std=0.02)

    

    def forward(self, obs, actions=None, is_pad=None):
        """
        obs: batch, obs_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        is_training = actions is not None
        bs, _ = obs.shape

        if is_training:
            if is_pad is None:
                is_pad = torch.zeros((bs, self.num_queries), device=obs.device, dtype=torch.bool)
            else:
                is_pad = is_pad.to(device=obs.device, dtype=torch.bool)

            cls_obs_is_pad = torch.full((bs, 2), False, device=obs.device, dtype=torch.bool)
            enc_pad = torch.cat([cls_obs_is_pad, is_pad], dim=1)

            input_pos_embed = self.input_pos_table.to(obs.device).permute(1, 0, 2)

            action_embed = self.encoder_action_proj(actions)
            obs_embed = self.encoder_obs_proj(obs).unsqueeze(1)
            cls_embed = self.cls_embed.weight.unsqueeze(0).repeat(bs, 1, 1)

            encoder_input = torch.cat([cls_embed, obs_embed, action_embed], dim=1).permute(1, 0, 2)

            encoder_output = self.encoder(encoder_input, pos=input_pos_embed, src_key_padding_mask=enc_pad)[0]
            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, : self.latent_dim]
            logvar = latent_info[:, self.latent_dim :]
            z = reparametrize(mu, logvar)
        else:
            mu = logvar = None
            z = torch.randn((bs, self.latent_dim), device=obs.device, dtype=obs.dtype)

        latent_token = self.latent_out_proj(z)
        obs_token = self.encoder_obs_proj(obs)

        src = torch.stack([latent_token, obs_token], dim=2)
        pos = self.dec_pos.to(obs.device)

        hs = self.transformer(src, None, self.query_embed.weight, pos)[0]
        a_hat = self.action_head(hs)
        return a_hat, mu, logvar


    @torch.no_grad()
    def sample_action(
        self,
        obs_cond,
        return_chunk=False,
    ):
        self.eval()

        bs, _ = obs_cond.shape

        z = torch.randn((bs, self.latent_dim), device=obs_cond.device, dtype=obs_cond.dtype)
        latent_token = self.latent_out_proj(z)
        obs_token = self.encoder_obs_proj(obs_cond)

        src = torch.stack([latent_token, obs_token], dim=2)
        pos = self.dec_pos.to(obs_cond.device)

        hs = self.transformer(src, None, self.query_embed.weight, pos)[0]
        a_hat = self.action_head(hs).clamp(-1, 1)

        if return_chunk:
            return a_hat
        else:
            return a_hat[:, 0, :]



# # class ACTBaselineNet(nn.Module):
# class ACTBaselineNet__old(nn.Module):
#     """
#     Baseline ACT:
#       cond = [z_img, z_q]
#       loss: BC + KL (contrastive 없음, sem/state split 없음)
#     """
#     def __init__(
#         self,
#         action_dim,
#         chunk_size=16,
#         cond_dim=146, 
#         # latent_dim=512,
#     ):
#         super().__init__()
#         # self.img_enc = ImageEncoderACTBaseline(z_img_dim=z_img_dim)
#         # self.q_enc = obsEncoder(q_dim=q_dim, z_q_dim=z_q_dim)
#         # cond_dim = z_img_dim + z_q_dim
#         self.act = ACTcVAE(cond_dim=cond_dim, action_dim=action_dim, chunk_size=chunk_size)

#     # def forward(self, img, obs, a_chunk, zq_dropout_p=0.0):
#     def forward(self, cond, a_chunk):
    
#         # z_img = self.img_enc(img)
#         # z_q = self.q_enc(obs)
#         # if zq_dropout_p > 0:
#         #     z_q = F.dropout(z_q, p=zq_dropout_p, training=self.training)

#         # cond = torch.cat([z_img, z_q], dim=1)
        
#         pred, mu, logvar = self.act(cond, a_chunk)
        
#         return pred, mu, logvar



#     @torch.no_grad()
#     def sample_action(
#         self,
#         # img,
#         # obs,
#         obs_cond,
#         n_samples=1,
#         temperature=1.0,
#         deterministic=False,
#         return_chunk=False,
#         # zq_dropout_p=0.0,
#         clamp=None,
#     ):
#         self.eval()

#         # if img.dim() == 3:
#         #     img = img.unsqueeze(0)
#         # if obs.dim() == 1:
#         #     obs = obs.unsqueeze(0)

#         # cond, _ = self.encode_obs(img, obs, zq_dropout_p=zq_dropout_p)
#         cond = obs_cond

#         B = cond.shape[0]
#         L = self.act.chunk_size
#         Da = self.act.action_dim
#         Dz = self.act.latent_dim

#         if deterministic:
#             z = torch.zeros((B, Dz), device=cond.device, dtype=cond.dtype)
#             chunk = self.act.decode(cond, z)
#             if clamp is not None:
#                 lo, hi = clamp
#                 chunk = chunk.clamp(lo, hi)
#             return chunk if return_chunk else chunk[:, 0, :]

#         if n_samples == 1:
#             z = temperature * torch.randn((B, Dz), device=cond.device, dtype=cond.dtype)
#             chunk = self.act.decode(cond, z)
#             if clamp is not None:
#                 lo, hi = clamp
#                 chunk = chunk.clamp(lo, hi)
#             return chunk if return_chunk else chunk[:, 0, :]

#         z = temperature * torch.randn((B, n_samples, Dz), device=cond.device, dtype=cond.dtype)
#         cond_rep = cond.unsqueeze(1).expand(B, n_samples, cond.shape[1]).reshape(B * n_samples, -1)
#         z_rep = z.reshape(B * n_samples, Dz)

#         chunk = self.act.decode(cond_rep, z_rep).view(B, n_samples, L, Da)
#         if clamp is not None:
#             lo, hi = clamp
#             chunk = chunk.clamp(lo, hi)
#         return chunk if return_chunk else chunk[:, :, 0, :]

