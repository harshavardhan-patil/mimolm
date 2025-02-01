from collections import OrderedDict
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import lightning as pl

from typing import Tuple
from torch import nn, Tensor
from einops import rearrange
from omegaconf import DictConfig, OmegaConf

from src.external.hptr.src.models.modules.mlp import MLP
from src.external.hptr.src.models.modules.transformer import TransformerBlock
from src.external.hptr.src.models.modules.multi_modal import MultiModalAnchors
from src.external.hptr.src.data_modules.agent_centric import AgentCentricPreProcessing
from src.external.hptr.src.data_modules.ac_global import AgentCentricGlobal

from src.modeling.modules.lm_utils import create_vocabulary, tokenize_motion, get_attention_mask
from src.modeling.modules.transformer import TransformerDecoder
from tensordict import TensorDict

class MimoLM(pl.LightningModule):
    def __init__(
        self,
        # hidden_dim: int,
        # agent_attr_dim: int,
        # map_attr_dim: int,
        # tl_attr_dim: int,
        # n_pl_node: int,
        # use_current_tl: bool,
        # pl_aggr: bool,
        # n_step_hist: int,
        # n_decoders: int,
        # decoder: DictConfig,
        # tf_cfg: DictConfig,
        # input_projections: DictConfig,
        # early_fusion_encoder: DictConfig,
        data_size,
        n_rollouts = 1,
        sampling_rate = 5,
        **kwargs,
    ) -> None:
        super().__init__()
        self.n_rollouts = n_rollouts
        self.samping_rate = sampling_rate
        self.sampling_step = 10 // sampling_rate
        self.preprocessor = nn.Sequential(OrderedDict([
            ('pre_1', AgentCentricPreProcessing(sampling_rate = 5,
                                        data_size=data_size, 
                                        time_step_current=49, 
                                        n_target=8,
                                        n_other=48,
                                        n_map=512,
                                        mask_invalid=False)),
            ('pre_2', AgentCentricGlobal(sampling_rate = 5,
                                data_size=data_size,
                               time_step_current=49,
                                dropout_p_history=0.15, 
                                add_ohe=True,
                                pl_aggr=False,
                                pose_pe= {"agent": "xy_dir",
                                        "map": "mpa_pl"}))
                                        ]))

        self.input_projections = InputProjections(agent_attr_dim=self.preprocessor.pre_2.model_kwargs["agent_attr_dim"],
                                    map_attr_dim=self.preprocessor.pre_2.model_kwargs["map_attr_dim"],
                                    n_step_hist=self.preprocessor.pre_2.model_kwargs["n_step_hist"],
                                    n_pl_node=self.preprocessor.pre_2.model_kwargs["n_pl_node"],
                                    hidden_dim=256,
                                    add_learned_pe=True,
                                    n_layer_mlp=3,
                                    mlp_cfg={"end_layer_activation" : True,
                                                "use_layernorm" :  False,
                                                "use_batchnorm" : False,
                                                "dropout_p" : None,})

        self.encoder = EarlyFusionEncoder(hidden_dim=256,
                                    tf_cfg={"n_head": 2,
                                            "dropout_p": 0.1,
                                            "norm_first": True,
                                            "bias": True},
                                    latent_query={"use_agent_type": False,
                                                "mode_emb": "none", # linear, mlp, add, none
                                                "mode_init": "xavier", # uniform, xavier
                                                "scale": 5.0},
                                    n_latent_query=192,
                                    n_encoder_layers=2)  

        self.decoder = MotionDecoder(max_delta = 8.0, #meters
                                n_quantization_bins = 128,
                                n_verlet_steps = 13,
                                emb_dim = 256,
                                sampling_rate = 5,
                                n_time_steps = 110,
                                n_target = 8, #should be same as AgentCentricProcessing
                                time_step_end = 49,
                                dropout_rate = 0.0,
                                n_rollouts = self.n_rollouts,
                                n_heads = 2,
                                n_layers = 2,)
        
        self.logsoftmax = torch.nn.LogSoftmax(dim=-1)
        self.criterion = torch.nn.NLLLoss()


    def on_before_batch_transfer(self, batch, idx):
        batch_tensor = TensorDict({
                    "agent/valid": batch["agent/valid"],
                    "agent/pos": batch["agent/pos"],
                    "agent/vel": batch["agent/vel"],
                    "agent/spd": batch["agent/spd"],
                    "agent/acc": batch["agent/acc"],
                    "agent/yaw_bbox": batch["agent/yaw_bbox"],
                    "agent/yaw_rate": batch["agent/yaw_rate"],
                # agent attributes
                    "agent/type": batch["agent/type"],
                    "agent/role": batch["agent/role"],
                    "agent/size": batch["agent/size"],
                # map polylines
                    "map/valid": batch["map/valid"],
                    "map/type": batch["map/type"],
                    "map/pos": batch["map/pos"],
                    "map/dir": batch["map/dir"]
        })
        return batch_tensor

    def training_step(self, batch, **kwargs):
        with torch.no_grad():
            batch = self.preprocessor(batch)
            actuals = tokenize_motion(batch["gt/pos"],
                                self.decoder.pos_bins, 
                                self.decoder.verlet_wrapper, 
                                self.decoder.n_verlet_steps, 
                                self.decoder.n_time_steps)
            
        motion_tokens = torch.cat((batch["ac/target_pos"], batch["gt/pos"][:, :, ::self.sampling_step,]), dim = -2)
        target_types = batch["ac/target_type"]
        input_dict = {
        k.split("input/")[-1]: v for k, v in batch.items() if "input/" in k
        }
        valid = input_dict["target_valid"].any(-1)
        target_emb, target_valid, other_emb, other_valid, map_emb, map_valid = self.input_projections(target_valid = input_dict["target_valid"], 
                target_attr = input_dict["target_attr"],
                other_valid = input_dict["other_valid"],
                other_attr = input_dict["other_attr"],
                map_valid = input_dict["map_valid"],
                map_attr = input_dict["map_attr"],)
        
        fused_emb, fused_emb_invalid = self.encoder(
                    target_emb, target_valid, other_emb, other_valid, map_emb, map_valid, input_dict["target_type"], valid
                )
        pred = self.decoder(motion_tokens, target_types, fused_emb, fused_emb_invalid)
        pred = F.interpolate(pred.permute(0, 2, 1), size=60, mode="linear", align_corners=True).permute(0, 2, 1)
        loss = self.criterion(
            self.logsoftmax(pred.flatten(0, 1)), 
            actuals.flatten(0, 1).flatten(0, 1).repeat(self.n_rollouts))

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {'params': self.input_projections.parameters(),
            'params': self.encoder.parameters(),
            'params': self.decoder.parameters()}], lr=1e-3)
        return optimizer

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
            sampling_rate: int = 5,
            n_time_steps: int = 110,
            n_target: int = 8, #should be same as AgentCentricProcessing
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
        self.sampling_rate = sampling_rate
        self.sampling_step = 10 // sampling_rate #Argov2 is sampled at 10 Hz native
        self.n_time_steps = n_time_steps // self.sampling_step
        self.step_current = (time_step_end + 1) // self.sampling_step - 1
        self.n_target = n_target
        self.dropout_rate = dropout_rate
        self.n_rollouts = n_rollouts
        self.n_heads = n_heads

        vocabulary, pos_bins, verlet_wrapper = create_vocabulary(self.max_delta, 
                                                                self.n_quantization_bins, 
                                                                self.n_verlet_steps)
        self.register_buffer("vocabulary", vocabulary)
        self.register_buffer("pos_bins", pos_bins)
        self.register_buffer("verlet_wrapper", verlet_wrapper)

        self.vocab_size = len(self.vocabulary)
        time_indices = torch.arange(0, self.n_time_steps)
        self.register_buffer("time_indices", time_indices)

        self.val_emb_layer = nn.Embedding(self.vocab_size, self.emb_dim)
        self.step_emb_layer = nn.Embedding(self.n_time_steps, self.emb_dim)
        self.type_emb_layer = nn.Embedding(3, self.emb_dim)

        self.register_buffer("attn_type", torch.tensor([True]))
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

        n_agents = motion_tokens.shape[1]
        n_batch = motion_tokens.shape[0]

        # quantized, discretized, verlet-wrapped motion tokens
        motion_tokens = tokenize_motion(motion_tokens, # [n_batch, n_agents, n_time_steps]
                                        self.pos_bins, 
                                        self.verlet_wrapper, 
                                        self.n_verlet_steps, 
                                        self.n_time_steps)
        
        # we compute a learned value embedding and two learned positional embeddings (representing the timestep and agent identity) for each discrete motion token, which are combined via an element-wise sum prior to being input to the transformer decoder.
        val_embeddings = self.val_emb_layer(motion_tokens)
        step_embeddings = self.step_emb_layer(self.time_indices
                                              ).unsqueeze(0
                                              ).repeat(motion_tokens.shape[1], 1, 1
                                              ).unsqueeze(0
                                              ).repeat(motion_tokens.shape[0], 1, 1, 1)
        type_embeddings = self.type_emb_layer(target_types.int().argmax(dim=-1)
                                              ).unsqueeze(2
                                              ).repeat(1, 1, self.n_time_steps, 1)
        # [n_batch, n_agents * n_time_steps, emb_dim]
        motion_embeddings = (val_embeddings + 
                             step_embeddings + 
                             type_embeddings
                             ).flatten(1, 2) 
        
        #self attending motion tokens + cross attending to scene emebeddings
        query = motion_embeddings
         # type_as casting simply to move to right device with Lightning
        for decoder_block in self.decoder_layers:
            if query.shape[0] != n_batch:
                query = query.unflatten(dim=0, sizes=(n_batch, n_agents)).flatten(1, 2)
            attn_mask = get_attention_mask(self.n_time_steps, query.shape[1]).type_as(self.attn_type)
            query = decoder_block(query = query, 
                                  key = fused_emb, 
                                  attn_mask = attn_mask,
                                  n_agents = n_agents,
                                  step_current = self.step_current)
            
        out = self.fully_connected_layers(query)
        return out
