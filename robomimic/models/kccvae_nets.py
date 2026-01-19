""" This file contains nets used for KCCVAE Policy. """
import math
from typing import Union


import torch
import torch.nn as nn
import torch.nn.functional as F




class ObsEncoderSemState(nn.Module):
    def __init__(self, cond_dim=256, z_sem_dim=64, z_state_dim=256):
        super().__init__()

        self.head_sem = nn.Sequential(
            nn.Linear(cond_dim, cond_dim), 
            nn.ReLU(), 
            nn.Linear(cond_dim, z_sem_dim)
            )
        
        self.head_state = nn.Sequential(
            nn.Linear(cond_dim, cond_dim), 
            nn.ReLU(), 
            nn.Linear(cond_dim, z_state_dim)
            )

    def forward(self, obs_cond):

        z_sem = self.head_sem(obs_cond)
        z_state = self.head_state(obs_cond)
        return z_sem, z_state


class ACTcVAE(nn.Module):
    """
    간단한 conditional VAE:
      - encoder: (cond, action_chunk) -> (mu, logvar)
      - decoder(policy): (cond, latent) -> action_chunk
    """
    def __init__(self, cond_dim, action_dim, chunk_size=16, latent_dim=32, hidden=256):
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.latent_dim = latent_dim

        in_enc = cond_dim + action_dim * chunk_size
        self.enc = nn.Sequential(
            nn.Linear(in_enc, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, latent_dim)
        self.logvar = nn.Linear(hidden, latent_dim)

        in_dec = cond_dim + latent_dim
        self.dec = nn.Sequential(
            nn.Linear(in_dec, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim * chunk_size),
        )

    def encode(self, cond, a_chunk):
        x = torch.cat([cond, a_chunk.flatten(1)], dim=1)
        h = self.enc(x)
        return self.mu(h), self.logvar(h)

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, cond, z):
        x = torch.cat([cond, z], dim=1)
        out = self.dec(x)
        return out.view(-1, self.chunk_size, self.action_dim)

    def forward(self, cond, a_chunk):
        mu, logvar = self.encode(cond, a_chunk)
        z = self.reparam(mu, logvar)
        pred = self.decode(cond, z)
        return pred, mu, logvar




class KCCVAENet(nn.Module):
    def __init__(
        self,
        action_dim,
        cond_dim=146, 
        chunk_size=16,
        z_sem_dim=64,
        z_state_dim=256,
        latent_dim=32,
    ):
        super().__init__()
        self.obs_enc = ObsEncoderSemState(cond_dim=cond_dim, z_sem_dim=z_sem_dim, z_state_dim=z_state_dim)
        z_dim = z_sem_dim + z_state_dim
        self.act = ACTcVAE(cond_dim=z_dim, action_dim=action_dim, chunk_size=chunk_size, latent_dim=latent_dim)
    


    def forward(self, obs_cond, a_chunk):

        z_sem, z_state = self.obs_enc(obs_cond)
        cond = torch.cat([z_sem, z_state], dim=1)
        pred, mu, logvar = self.act(cond, a_chunk)
        return pred, mu, logvar, z_sem, z_state


    @torch.no_grad()
    def encode_obs(self, obs_cond):
        """
        img: (B,3,H,W) float [0,1]
        qpos: (B,q_dim)
        return: cond (B,cond_dim), (z_sem, z_state, z_q)
        """
        z_sem, z_state = self.obs_enc(obs_cond)
        cond = torch.cat([z_sem, z_state], dim=1)
        return cond, (z_sem, z_state)

    @torch.no_grad()
    def sample_action(
        self,
        obs_cond,
        n_samples=1,
        temperature=1.0,
        deterministic=False,
        return_chunk=False,
        clamp=None,  # e.g. (-1,1) or None
    ):
        """
        Inference용:
        - posterior 사용 X (a_chunk가 없으므로)
        - z ~ N(0, I) from prior
        - decode(cond, z) -> action_chunk
        - 첫 스텝 action 또는 전체 chunk 반환

        Args:
          img: (B,3,H,W) float [0,1]
          qpos: (B,q_dim)
          n_samples: 여러 z를 뽑고 첫 step만 모아 반환(현재는 그냥 다 반환 or 첫 샘플 사용)
          temperature: latent std 스케일 (z = temperature * eps)
          deterministic: True면 z=0 (mean action) 사용
          return_chunk: True면 (B,L,Da) 반환, 아니면 (B,Da) 첫 스텝만
          clamp: None 또는 (lo,hi)로 action을 clamp
        """
        self.eval()

        cond, _ = self.encode_obs(obs_cond)

        B = cond.shape[0]
        L = self.act.chunk_size
        Da = self.act.action_dim
        Dz = self.act.latent_dim

        if deterministic:
            z = torch.zeros((B, Dz), device=cond.device, dtype=cond.dtype)
            chunk = self.act.decode(cond, z)  # (B,L,Da)
            if clamp is not None:
                lo, hi = clamp
                chunk = chunk.clamp(lo, hi)
            return chunk if return_chunk else chunk[:, 0, :]

        # stochastic sampling
        if n_samples == 1:
            eps = torch.randn((B, Dz), device=cond.device, dtype=cond.dtype)
            z = temperature * eps
            chunk = self.act.decode(cond, z)
            if clamp is not None:
                lo, hi = clamp
                chunk = chunk.clamp(lo, hi)
            return chunk if return_chunk else chunk[:, 0, :]

        # n_samples > 1: 여러 샘플 chunk를 반환하거나, 첫 action만 (B,n,Da)
        eps = torch.randn((B, n_samples, Dz), device=cond.device, dtype=cond.dtype)
        z = temperature * eps
        cond_rep = cond.unsqueeze(1).expand(B, n_samples, cond.shape[1]).reshape(B * n_samples, -1)
        z_rep = z.reshape(B * n_samples, Dz)

        chunk = self.act.decode(cond_rep, z_rep).view(B, n_samples, L, Da)  # (B,n,L,Da)
        if clamp is not None:
            lo, hi = clamp
            chunk = chunk.clamp(lo, hi)

        if return_chunk:
            return chunk  # (B,n,L,Da)
        else:
            return chunk[:, :, 0, :]  # (B,n,Da)