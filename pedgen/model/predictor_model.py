"""Lightning wrapper for standalone Predictor training."""
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from torch.optim.lr_scheduler import MultiStepLR


class PredictorModel(LightningModule):
    """Lightning model for standalone global planning predictor."""

    def __init__(
            self,
            gpus: int,
            batch_size_per_device: int,
            latent_dim: int,
            optimizer_conf: Dict,
            lr_scheduler_conf: Dict,
            pred_horizon: int = 150,
    ) -> None:
        super().__init__()
        self.num_semantic_classes = 19
        self.pred_horizon = pred_horizon
        sem_dim = latent_dim // 2
        xyz_dim = latent_dim - sem_dim

        self.scene_xyz_embed = nn.Sequential(
            nn.Linear(3, xyz_dim),
            nn.ReLU(inplace=True),
            nn.Linear(xyz_dim, xyz_dim),
        )
        self.scene_semantic_embed = nn.Embedding(self.num_semantic_classes, sem_dim)
        self.scene_token_embed = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(inplace=True),
        )

        self.scene_memory_norm = nn.LayerNorm(latent_dim)
        self.scene_memory_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=latent_dim,
                nhead=4,
                dim_feedforward=latent_dim * 4,
                dropout=0.1,
                batch_first=True,
                activation="gelu",
            ),
            num_layers=2,
        )

        self.traj_pos_embed = nn.Parameter(torch.randn(1, self.pred_horizon, latent_dim) * 0.02)
        self.traj_input_embed = nn.Sequential(
            nn.LayerNorm(3),
            nn.Linear(3, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=latent_dim,
            nhead=4,
            dim_feedforward=latent_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.traj_decoder = nn.TransformerDecoder(decoder_layer, num_layers=3)
        self.decoder_out_norm = nn.LayerNorm(latent_dim)
        self.traj_out_head = nn.Linear(latent_dim, 3)

        self.gpus = gpus
        self.batch_size_per_device = batch_size_per_device
        self.optimizer_conf = optimizer_conf
        self.lr_scheduler_conf = lr_scheduler_conf

    def encode_scene(self, batch: Dict) -> torch.Tensor:
        scene_points = batch["scene_tokens"].to(self.traj_pos_embed.device)
        scene_xyz = scene_points[..., :3]
        scene_semantic_idx = scene_points[..., 3].long().clamp(min=0, max=self.num_semantic_classes - 1)
        scene_xyz_feat = self.scene_xyz_embed(scene_xyz)
        scene_sem_feat = self.scene_semantic_embed(scene_semantic_idx)
        scene_tokens = self.scene_token_embed(torch.cat([scene_xyz_feat, scene_sem_feat], dim=-1))
        return self.scene_memory_encoder(self.scene_memory_norm(scene_tokens))

    def predict_traj_150(self, batch: Dict) -> torch.Tensor:
        if "scene_tokens" not in batch:
            raise RuntimeError("scene_tokens is required for predictor")
        if "gt_init_pos" not in batch:
            raise RuntimeError("gt_init_pos is required for predictor")

        scene_memory = self.encode_scene(batch)
        gt_init_pos = batch["gt_init_pos"].to(scene_memory.device)
        batch_size = gt_init_pos.shape[0]

        decoder_query = gt_init_pos.unsqueeze(1).repeat(1, self.pred_horizon, 1)
        decoder_input = self.traj_input_embed(decoder_query)
        decoder_input = decoder_input + self.traj_pos_embed[:, :decoder_input.shape[1], :]
        decoded = self.traj_decoder(tgt=decoder_input, memory=scene_memory)
        pred_rel_traj_150 = self.traj_out_head(self.decoder_out_norm(decoded))
        pred_rel_traj_150 = torch.cat([torch.zeros_like(pred_rel_traj_150[:, :1, :]), pred_rel_traj_150[:, 1:, :]], dim=1)

        if pred_rel_traj_150.shape[1] != self.pred_horizon:
            raise RuntimeError(f"predictor output horizon mismatch: {pred_rel_traj_150.shape[1]} vs {self.pred_horizon}")
        if pred_rel_traj_150.shape[0] != batch_size:
            raise RuntimeError("predictor output batch size mismatch")
        return pred_rel_traj_150

    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        gt_init_pos = batch["gt_init_pos"].to(self.traj_pos_embed.device)
        pred_rel_traj_150 = self.predict_traj_150(batch)
        pred_traj_150 = gt_init_pos.unsqueeze(1) + pred_rel_traj_150
        output = {"pred_traj_150": pred_traj_150, "pred_rel_traj_150": pred_rel_traj_150}

        if "gt_traj_150" in batch:
            gt_traj_150 = batch["gt_traj_150"].to(pred_traj_150.device)
            gt_traj_150_mask = batch.get("gt_traj_150_mask", None)
            if gt_traj_150_mask is not None:
                traj_mask = gt_traj_150_mask.to(pred_traj_150.device).unsqueeze(-1)[:, 1:, :]
                traj_diff = torch.abs(pred_traj_150[:, 1:, :] - gt_traj_150[:, 1:, :])
                traj_denom = torch.clamp(traj_mask.sum() * traj_diff.shape[-1], min=1.0)
                loss_full_traj = (traj_diff * traj_mask).sum() / traj_denom
            else:
                loss_full_traj = F.l1_loss(pred_traj_150[:, 1:, :], gt_traj_150[:, 1:, :])
        else:
            loss_full_traj = torch.tensor(0.0, device=pred_traj_150.device)

        output["loss_full_traj"] = loss_full_traj
        output["gt_init_pos"] = gt_init_pos
        return output

    def training_step(self, batch: Dict) -> torch.Tensor:
        predictor_out = self(batch)
        loss_full_traj = predictor_out["loss_full_traj"]
        self.log("train/loss_full_traj", loss_full_traj, prog_bar=True, logger=True, on_step=True, on_epoch=False, batch_size=batch["batch_size"])
        return loss_full_traj

    def eval_step(self, batch: Dict) -> Dict[str, torch.Tensor]:
        gt_init_pos = batch["gt_init_pos"].to(self.traj_pos_embed.device)
        pred_rel_traj_150 = self.predict_traj_150(batch)
        out_dict = {"pred_traj_150": gt_init_pos.unsqueeze(1) + pred_rel_traj_150}
        if "meta" in batch:
            out_dict["meta"] = batch["meta"]
        return out_dict

    def validation_step(self, batch: Dict) -> Dict[str, torch.Tensor]:
        predictor_out = self(batch)
        loss_full_traj = predictor_out["loss_full_traj"]
        self.log("val/loss_full_traj", loss_full_traj, prog_bar=True, logger=True, on_step=False, on_epoch=True, batch_size=batch["batch_size"])
        out_dict = {"pred_traj_150": predictor_out["pred_traj_150"]}
        if "meta" in batch:
            out_dict["meta"] = batch["meta"]
        return out_dict

    def test_step(self, batch: Dict) -> Dict[str, torch.Tensor]:
        return self.eval_step(batch)

    def configure_optimizers(self):
        lr = self.optimizer_conf["basic_lr_per_img"] * self.batch_size_per_device * self.gpus
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-7)
        scheduler = MultiStepLR(
            optimizer,
            milestones=self.lr_scheduler_conf["milestones"],
            gamma=self.lr_scheduler_conf["gamma"],
        )
        return [[optimizer], [scheduler]]