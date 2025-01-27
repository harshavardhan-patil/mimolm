import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

from typing import Tuple
from torch import nn, Tensor
from einops import rearrange
from omegaconf import DictConfig

from src.external.hptr.src.models.modules.mlp import MLP
from src.external.hptr.src.models.modules.point_net import PointNet
from src.external.hptr.src.models.modules.transformer import TransformerBlock
from src.external.hptr.src.models.modules.multi_modal import MultiModalAnchors

from src.modeling.modules.lm_utils import create_vocabulary, tokenize_motion, get_attention_mask
from src.modeling.modules.transformer import TransformerDecoder

class MimoLM(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        agent_attr_dim: int,
        map_attr_dim: int,
        tl_attr_dim: int,
        n_pl_node: int,
        use_current_tl: bool,
        pl_aggr: bool,
        n_step_hist: int,
        n_decoders: int,
        decoder: DictConfig,
        tf_cfg: DictConfig,
        input_projections: DictConfig,
        early_fusion_encoder: DictConfig,
        **kwargs,
    ) -> None:
        super().__init__()
        self.n_pred = decoder.n_pred
        self.n_decoders = n_decoders
        self.pl_aggr = pl_aggr
        self.pred_subsampling_rate = kwargs.get("pred_subsampling_rate", 1)
        decoder["mlp_head"]["n_step_future"] = decoder["mlp_head"]["n_step_future"] // self.pred_subsampling_rate

        self.input_projections = InputProjections(
            hidden_dim=hidden_dim,
            agent_attr_dim=agent_attr_dim,
            map_attr_dim=map_attr_dim,
            tl_attr_dim=tl_attr_dim,
            pl_aggr=pl_aggr,
            use_current_tl=use_current_tl,
            n_step_hist=n_step_hist,
            n_pl_node=n_pl_node,
            **input_projections
        )

        self.encoder = EarlyFusionEncoder(
            hidden_dim=hidden_dim,
            tf_cfg=tf_cfg,
            **early_fusion_encoder
        )

        decoder["tf_cfg"] = tf_cfg
        decoder["hidden_dim"] = hidden_dim

        model_parameters = filter(lambda p: p.requires_grad, self.input_projections.parameters())
        total_params = sum([np.prod(p.size()) for p in model_parameters])
        print(f"Input projections parameters: {total_params/1000000:.2f}M")
        model_parameters = filter(lambda p: p.requires_grad, self.encoder.parameters())
        total_params = sum([np.prod(p.size()) for p in model_parameters])
        print(f"Encoder parameters: {total_params/1000000:.2f}M")
        model_parameters = filter(lambda p: p.requires_grad, self.decoder.parameters())
        total_params = sum([np.prod(p.size()) for p in model_parameters])
        print(f"Decoder parameters: {total_params/1000000:.2f}M")

    def forward(
        self,
        target_valid: Tensor,
        target_type: Tensor,
        target_attr: Tensor,
        other_valid: Tensor,
        other_attr: Tensor,
        tl_valid: Tensor,
        tl_attr: Tensor,
        map_valid: Tensor,
        map_attr: Tensor,
        inference_repeat_n: int = 1,
        inference_cache_map: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            target_type: [n_scene, n_target, 3]
            # target history, other history, map
            target_valid: [n_scene, n_target, n_step_hist], bool
            target_attr: [n_scene, n_target, n_step_hist, agent_attr_dim]
            other_valid: [n_scene, n_target, n_other, n_step_hist], bool
            other_attr: [n_scene, n_target, n_other, n_step_hist, agent_attr_dim]
            map_valid: [n_scene, n_target, n_map, n_pl_node], bool
            map_attr: [n_scene, n_target, n_map, n_pl_node, map_attr_dim]

        Returns: will be compared to "output/gt_pos": [n_scene, n_agent, n_step_future, 2]
            valid: [n_scene, n_target]
            conf: [n_decoder, n_scene, n_target, n_pred], not normalized!
            pred: [n_decoder, n_scene, n_target, n_pred, n_step_future, pred_dim]
        """
        for _ in range(inference_repeat_n):
            valid = target_valid if self.pl_aggr else target_valid.any(-1)  # [n_scene, n_target]
            target_emb, target_valid, other_emb, other_valid, tl_emb, tl_valid, map_emb, map_valid = self.input_projections(
                target_valid=target_valid,
                target_attr=target_attr,
                other_valid=other_valid,
                other_attr=other_attr,
                map_valid=map_valid,
                map_attr=map_attr,
                tl_valid=tl_valid,
                tl_attr=tl_attr,
            )

            fused_emb, fused_emb_invalid = self.encoder(
                target_emb, target_valid, other_emb, other_valid, tl_emb, tl_valid, map_emb, map_valid, target_type, valid
            )

            conf, pred = self.decoder(valid=valid, target_type=target_type, emb=fused_emb, emb_invalid=fused_emb_invalid)

            if self.pred_subsampling_rate != 1:
                n_decoder, n_scene, n_target, n_pred, n_step_future, pred_dim = pred.shape
                pred = rearrange(
                    pred,
                    "n_decoder n_scene n_target n_pred n_step_future pred_dim -> (n_decoder n_scene n_target n_pred) pred_dim n_step_future",
                )
                pred = F.interpolate(pred, mode="linear", scale_factor=self.pred_subsampling_rate)
                pred = rearrange(
                    pred,
                    "(n_decoder n_scene n_target n_pred) pred_dim n_step_future -> n_decoder n_scene n_target n_pred n_step_future pred_dim",
                    n_decoder=n_decoder, n_scene=n_scene, n_target=n_target, n_pred=n_pred, pred_dim=pred_dim,
                )

        assert torch.isfinite(conf).all()
        assert torch.isfinite(pred).all()
        return valid, conf, pred


class InputProjections(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        agent_attr_dim: int,
        map_attr_dim: int,
        n_step_hist: int,
        n_pl_node: int,
        add_learned_pe: bool,
        n_layer_mlp: int,
        mlp_cfg: DictConfig,
    ) -> None:
        super().__init__()
        self.add_learned_pe = add_learned_pe

        self.fc_target = MLP([agent_attr_dim] + [hidden_dim] * n_layer_mlp, **mlp_cfg)
        self.fc_other = MLP([agent_attr_dim] + [hidden_dim] * n_layer_mlp, **mlp_cfg)
        self.fc_map = MLP([map_attr_dim] + [hidden_dim] * n_layer_mlp, **mlp_cfg)

        if self.add_learned_pe:
            self.pe_target = nn.Parameter(torch.zeros([1, n_step_hist, hidden_dim]), requires_grad=True)
            self.pe_other = nn.Parameter(torch.zeros([1, 1, n_step_hist, hidden_dim]), requires_grad=True)
            self.pe_map = nn.Parameter(torch.zeros([1, 1, n_pl_node, hidden_dim]), requires_grad=True)

    def forward(
        self,
        target_valid: Tensor,
        target_attr: Tensor,
        other_valid: Tensor,
        other_attr: Tensor,
        map_valid: Tensor,
        map_attr: Tensor,
    ) -> Tuple[
        Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor
    ]:
        """
        Args:
            # target history, other history, map
            target_valid: [n_scene, n_target, n_step_hist], bool
            target_attr: [n_scene, n_target, n_step_hist, agent_attr_dim]
            other_valid: [n_scene, n_target, n_other, n_step_hist], bool
            other_attr: [n_scene, n_target, n_other, n_step_hist, agent_attr_dim]
            map_valid: [n_scene, n_target, n_map, n_pl_node], bool
            map_attr: [n_scene, n_target, n_map, n_pl_node, map_attr_dim]

        Returns:
            target_emb: [n_batch, 1 or n_step_hist, hidden_dim], n_batch = n_scene * n_target (agent-centric)
            target_valid: [n_batch, 1 or n_step_hist]
            other_emb: [n_batch, n_other or n_other * n_step_hist, hidden_dim]
            other_valid: [n_batch, n_other or n_other * n_step_hist]
            map_emb: [n_batch, n_map or n_map * n_pl_node, hidden_dim]
            map_valid: [n_batch, n_map or n_map * n_pl_node]
        """
        # [n_batch, n_map, (n_pl_node), map_attr_dim]
        map_valid = map_valid.flatten(0, 1)
        map_emb = self.fc_map(map_attr.flatten(0, 1), map_valid)
        # [n_batch, n_other, (n_step_hist), agent_attr_dim]
        other_valid = other_valid.flatten(0, 1)
        other_emb = self.fc_other(other_attr.flatten(0, 1), other_valid)
        # [n_batch, (n_step_hist), agent_attr_dim]
        target_valid = target_valid.flatten(0, 1)
        target_emb = self.fc_target(target_attr.flatten(0, 1), target_valid)

        if self.add_learned_pe:
            map_emb = map_emb + self.pe_map
            other_emb = other_emb + self.pe_other
            target_emb = target_emb + self.pe_target

        # target_emb: [n_batch, n_step_hist/1, :], target_valid: [n_batch, n_step_hist/1]
        map_emb = map_emb.flatten(1, 2)  # [n_batch, n_map * n_pl_node, :]
        map_valid = map_valid.flatten(1, 2)  # [n_batch, n_map * n_pl_node]
        other_emb = other_emb.flatten(1, 2)  # [n_batch, n_other * n_step_hist, :]
        other_valid = other_valid.flatten(1, 2)  # [n_batch, n_other * n_step_hist]

        return (
            target_emb, target_valid,
            other_emb, other_valid,
            map_emb, map_valid,
        )


class EarlyFusionEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        tf_cfg: DictConfig,
        latent_query: DictConfig,
        n_latent_query: int,
        n_encoder_layers: int,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_encoder_layers = n_encoder_layers
        self.n_latent_query = n_latent_query

        self.latent_query = MultiModalAnchors(
            hidden_dim=hidden_dim, emb_dim=hidden_dim, n_pred=self.n_latent_query, **latent_query
        )
        self.tf_latent_cross = TransformerBlock(
            d_model=hidden_dim, d_feedforward=hidden_dim * 4, n_layer=1, **tf_cfg
        )
        self.tf_latent_self = TransformerBlock(
            d_model=hidden_dim, d_feedforward=hidden_dim * 4, n_layer=n_encoder_layers, **tf_cfg
        )

    def forward(
        self,
        target_emb, target_valid,
        other_emb, other_valid,
        map_emb, map_valid,
        target_type, valid,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            target_emb: [n_batch, 1 or n_step_hist, hidden_dim], n_batch = n_scene * n_target (agent-centric)
            target_valid: [n_batch, 1 or n_step_hist]
            other_emb: [n_batch, n_other or n_other * n_step_hist, hidden_dim]
            other_valid: [n_batch, n_other or n_other * n_step_hist]
            map_emb: [n_batch, n_map or n_map * n_pl_node, hidden_dim]
            map_valid: [n_batch, n_map or n_map * n_pl_node] 
            target_type: [n_scene, n_target, 3], bool one_hot [Vehicle=0, Pedestrian=1, Cyclist=2]
            valid: [n_scene, n_target]

        Returns:
            emb: [n_scene * n_target, :, hidden_dim]
            emb_invalid: [n_scene * n_target, :]
        """
        emb = torch.cat([target_emb, other_emb, map_emb], dim=1)
        emb_invalid = ~torch.cat([target_valid, other_valid, map_valid], dim=1)
        # [n_scene * n_agent, n_latent_query, out_dim]
        lq_emb = self.latent_query(valid.flatten(0, 1), None, target_type.flatten(0, 1))
        emb, _ = self.tf_latent_cross(src=lq_emb, tgt=emb, tgt_padding_mask=emb_invalid)
        emb, _ = self.tf_latent_self(src=emb, tgt=emb)
        emb_invalid = (~valid).flatten(0, 1).unsqueeze(-1).expand(-1, lq_emb.shape[1])

        return emb, emb_invalid

#to-do: reducing the sampling rate of the predictions
class MotionDecoder(nn.Module):
    def __init__(
            self,
            max_delta: float = 4.0, #meters
            n_quantization_bins: int = 128,
            n_verlet_steps: int = 13,
            emb_dim: int = 256,
            n_time_steps: int = 110,
            n_target: int = 6, #should be same as AgentCentricProcessing
            time_step_end: int = 49,
            dropout_rate: int = 0.0,
            n_rollouts: int = 64,
            n_heads: int = 2,
            n_layers: int = 1,
            ) -> None:
        super().__init__()

        self.max_delta = max_delta
        self.n_quantization_bins = n_quantization_bins
        self.n_verlet_steps = n_verlet_steps
        self.emb_dim = emb_dim
        self.n_time_steps = n_time_steps
        self.n_target = n_target
        self.dropout_rate = dropout_rate
        self.n_rollouts = n_rollouts
        self.n_agents = 0
        self.n_batch = 0
        self.n_heads = n_heads

        self.vocabulary, self.pos_bins, self.verlet_wrapper = create_vocabulary(self.max_delta, 
                                                                                self.n_quantization_bins, 
                                                                                self.n_verlet_steps)
        self.vocab_size = len(self.vocabulary)
        self.time_indices = torch.arange(0, n_time_steps)

        self.val_emb_layer = nn.Embedding(self.vocab_size, self.emb_dim)
        self.step_emb_layer = nn.Embedding(self.n_time_steps, self.emb_dim)
        self.type_emb_layer = nn.Embedding(3, self.emb_dim)

        self.decoder_layers = nn.ModuleList(
            [TransformerDecoder(num_heads=self.n_heads,
                                emb_dim=self.emb_dim,
                                dropout_rate=self.dropout_rate,
                                n_time_steps=self.n_time_steps,
                                n_rollouts=self.n_rollouts) for _ in range(n_layers)]
        )

        self.fully_connected_layers = nn.Sequential(
            nn.Linear(self.emb_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, self.vocab_size)
        )
    
    def forward(
            self,
            motion_tokens: Tensor, # [n_batch, n_agents, n_time_steps, 2]
            target_types: Tensor,
            fused_emb: Tensor,
            fused_emb_invalid: Tensor,
            ) -> Tuple[Tensor, Tensor]:

        self.n_agents = motion_tokens.shape[1]
        self.n_batch = motion_tokens.shape[0]
        motion_tokens = tokenize_motion(motion_tokens, # [n_batch, n_agents, n_time_steps]
                                        self.pos_bins, 
                                        self.verlet_wrapper, 
                                        self.n_verlet_steps, 
                                        self.n_time_steps)
        print(motion_tokens.shape)
        # we compute a learned value embedding and two learned positional embeddings (representing the timestep and agent identity) for each discrete motion token, which are combined via an element-wise sum prior to being input to the transformer decoder.
        val_embeddings = self.val_emb_layer(motion_tokens)
        step_embeddings = self.step_emb_layer(self.time_indices).unsqueeze(0).repeat(motion_tokens.shape[1], 1, 1).unsqueeze(0).repeat(motion_tokens.shape[0], 1, 1, 1)
        type_embeddings = self.type_emb_layer(target_types
                                              .int().argmax(dim=-1)).unsqueeze(2).repeat(1, 1, self.n_time_steps, 1)
        motion_embeddings = (val_embeddings + step_embeddings + type_embeddings).flatten(1, 2) # [n_batch, n_agents * n_time_steps, emb_dim]
        attn_mask = get_attention_mask(self.n_time_steps, motion_embeddings.shape[1])
        query = motion_embeddings
        for decoder_block in self.decoder_layers:
            query = decoder_block(query = query, 
                                  key = fused_emb, 
                                  attn_mask = attn_mask,
                                  n_agents = motion_tokens.shape[1])

        out = self.fully_connected_layers(query)
        return out
