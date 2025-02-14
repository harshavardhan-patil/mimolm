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

from src.modeling.modules.lm_utils import create_vocabulary, tokenize_motion, get_attention_mask, nucleus_sampling, interpolate_trajectory, cluster_rollouts, non_maximum_suppression
from src.external.hptr.src.utils.transform_utils import torch_pos2global
from src.modeling.modules.transformer import TransformerDecoder
from src.modeling.modules.av2_metrics import (
    compute_world_ade,
    compute_world_fde,
    compute_world_brier_fde,
    compute_world_misses,
    compute_world_collisions
)

class MimoLM(pl.LightningModule):
    def __init__(
        self,
        data_size,
        n_rollouts,
        learning_rate,
        sampling_rate = 5,
        n_targets = 8,
        enc_dim = 128,
        dec_dim = 256,
        n_heads = 2,
        n_layers = 2,
        **kwargs,
    ) -> None:
        super().__init__()
        self.n_rollouts = n_rollouts
        self.n_targets = n_targets
        self.sampling_rate = sampling_rate
        self.sampling_step = 10 // sampling_rate
        self.inference_steps = 60 // self.sampling_step # AV2 has 60 future timesteps
        self.inference_start = 50 // self.sampling_step
        self.learning_rate = learning_rate

        self.preprocessor = nn.Sequential(OrderedDict([
            ('pre_1', AgentCentricPreProcessing(sampling_rate = self.sampling_rate,
                                        time_step_current=49, 
                                        n_target=self.n_targets,
                                        n_other=48,
                                        n_map=512,
                                        mask_invalid=False)),
            ('pre_2', AgentCentricGlobal(sampling_rate = self.sampling_rate,
                                data_size=data_size,
                               time_step_current=49,
                                dropout_p_history=0.1, 
                                add_ohe=True,
                                pl_aggr=False,
                                pose_pe= {"agent": "xy_dir",
                                        "map": "mpa_pl"}))
                                        ]))

        self.input_projections = InputProjections(agent_attr_dim=self.preprocessor.pre_2.model_kwargs["agent_attr_dim"],
                                    map_attr_dim=self.preprocessor.pre_2.model_kwargs["map_attr_dim"],
                                    n_step_hist=self.preprocessor.pre_2.model_kwargs["n_step_hist"],
                                    n_pl_node=self.preprocessor.pre_2.model_kwargs["n_pl_node"],
                                    hidden_dim=enc_dim,
                                    add_learned_pe=True,
                                    n_layer_mlp=3,
                                    mlp_cfg={"end_layer_activation" : True,
                                                "use_layernorm" :  False,
                                                "use_batchnorm" : False,
                                                "dropout_p" : None,})

        self.encoder = EarlyFusionEncoder(hidden_dim=enc_dim,
                                    tf_cfg={"n_head": n_heads,
                                            "dropout_p": 0.1,
                                            "norm_first": True,
                                            "bias": True},
                                    latent_query={"use_agent_type": False,
                                                "mode_emb": "none", # linear, mlp, add, none
                                                "mode_init": "xavier", # uniform, xavier
                                                "scale": 5.0},
                                    n_latent_query=enc_dim - 32,
                                    n_encoder_layers=n_layers)  

        self.decoder = MotionDecoder(max_delta = 8.0, #meters
                                n_quantization_bins = 128,
                                n_verlet_steps = 13,
                                emb_dim = dec_dim,
                                enc_dim = enc_dim,
                                sampling_rate = self.sampling_rate,
                                n_time_steps = 110,
                                n_target = self.n_targets, #should be same as AgentCentricProcessing
                                time_step_end = 49,
                                dropout_rate = 0.2,
                                n_rollouts = self.n_rollouts,
                                n_heads = n_heads,
                                n_layers = n_layers,)
        
        self.criterion = torch.nn.CrossEntropyLoss()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW([
            {'params': self.input_projections.parameters(),
            'params': self.encoder.parameters(),
            'params': self.decoder.parameters()}]
            , lr=self.learning_rate
            , weight_decay=0.6)
        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.StepLR(optimizer
                                                         , step_size=5
                                                         , gamma=0.9),
            "interval": "epoch",
            # How many epochs/steps should pass between calls to `scheduler.step()`. 1 corresponds to updating the learning rate after every epoch/step.
            "frequency": 1,
        }
        return [optimizer], [lr_scheduler]
    
    def training_step(self, batch, **kwargs):
        with torch.no_grad():
            batch = self.preprocessor(batch)
            actuals, _ = tokenize_motion(batch["gt/pos"],
                                self.decoder.pos_bins, 
                                self.decoder.verlet_wrapper, 
                                self.decoder.n_verlet_steps)
            n_batch = batch["ac/target_pos"].shape[0]
            
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
        pred, _ = self.decoder(motion_tokens, target_types, fused_emb, fused_emb_invalid)
        pred = pred[:, self.inference_start - 1: -1, :]
        loss = self.criterion(pred.flatten(0, 1), actuals[:, :, ::self.sampling_step].flatten(0, 2))
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=n_batch)
        return loss
    
    def validation_step(self, batch, **kwargs):
        batch = self.preprocessor(batch)
        actuals, _ = tokenize_motion(batch["gt/pos"],
            self.decoder.pos_bins, 
            self.decoder.verlet_wrapper, 
            self.decoder.n_verlet_steps)
        n_batch, n_agents = batch["ac/target_pos"].shape[0], batch["ac/target_pos"].shape[1]
        
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
        preds = []
        for _ in range(self.inference_steps):
            last_pos = batch["ac/target_pos"][:, :, -1]
            motion_tokens = batch["ac/target_pos"]
            target_types = batch["ac/target_type"]
            pred, last_token = self.decoder(motion_tokens, target_types, fused_emb, fused_emb_invalid)
            preds.append(pred[:, -1].unsqueeze(1))
            pred = F.softmax(pred[:, -1], dim=-1).argmax(dim=1)
            pred = self.decoder.vocabulary[pred][:, 1:].unflatten(dim=0, sizes=(n_batch, n_agents))
            pred = self.decoder.verlet_wrapper[pred]
            pred = torch.clamp(last_token + pred, min=0, max=127)
            pred = self.decoder.pos_bins[pred.long()]
            pred = last_pos + pred
            batch["ac/target_pos"] = torch.cat((batch["ac/target_pos"], pred.unsqueeze(2)), dim = -2)

        preds = torch.cat(preds, dim=1)
        loss = self.criterion(preds.flatten(0, 1), actuals[:, :, ::self.sampling_step].flatten(0, 2))
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=n_batch) 
        return loss
    
    def test_step(self, batch, **kwargs):
        batch = self.preprocessor(batch)
        actuals, _ = tokenize_motion(batch["gt/pos"],
            self.decoder.pos_bins, 
            self.decoder.verlet_wrapper, 
            self.decoder.n_verlet_steps)
        n_batch, n_agents = batch["ac/target_pos"].shape[0], batch["ac/target_pos"].shape[1]
        
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
        
        batch["ac/target_pos"] = batch["ac/target_pos"].repeat_interleave(repeats=self.n_rollouts, dim=0)
        batch["ac/target_type"] = batch["ac/target_type"].repeat_interleave(repeats=self.n_rollouts, dim=0)
        fused_emb = fused_emb.repeat_interleave(repeats=self.n_rollouts, dim=0)
        for _ in range(self.inference_steps):
            last_pos = batch["ac/target_pos"][:, :, -1]
            motion_tokens = batch["ac/target_pos"]
            target_types = batch["ac/target_type"]
            pred, last_token = self.decoder(motion_tokens, target_types, fused_emb, fused_emb_invalid)
            pred = nucleus_sampling(pred[:, -1])
            pred = self.decoder.vocabulary[pred][:, 1:].unflatten(dim=0, sizes=(n_batch * self.n_rollouts, n_agents))
            pred = self.decoder.verlet_wrapper[pred]
            pred = torch.clamp(last_token + pred, min=0, max=127)
            pred = self.decoder.pos_bins[pred.long()]
            pred = last_pos + pred
            batch["ac/target_pos"] = torch.cat((batch["ac/target_pos"], pred.unsqueeze(2)), dim = -2)

        preds = batch["ac/target_pos"][:, :, 25:, ]
        # upsample from 5 Hz to 10 Hz
        preds = interpolate_trajectory(preds, self.sampling_step, self.device)
        # NMS to remove redundant rollouts
        # filtered_rollouts = non_maximum_suppression(preds, threshold=3.0)
        brierminfde = [0] * n_batch
        minade = [0] * n_batch
        minfde = [0] * n_batch
        # KMeans to find cluster centres aka output worlds
        for n in range(n_batch):
            mode_trajectories, mode_probs = cluster_rollouts(preds[n * self.n_rollouts:(n + 1) * self.n_rollouts,], n_clusters=6)
            # transform to global coordinate
            trajs = torch_pos2global(mode_trajectories, batch['ref/pos'][n:n+1].repeat(6, 1, 1, 1), batch["ref/rot"][n:n+1].repeat(6, 1, 1, 1))
            gt_pos = torch_pos2global(batch["gt/pos"][n:n+1], batch['ref/pos'][n:n+1], batch["ref/rot"][n:n+1])
            
            forecasted_trajs = trajs.permute(1, 0, 2, 3).cpu()
            gt_trajs = gt_pos.squeeze(0).cpu()
            mode_probs = mode_probs.cpu()
            brierminfde[n] = min(compute_world_brier_fde(forecasted_trajs, gt_trajs, mode_probs)[:6])
            minade[n] = min(compute_world_ade(forecasted_trajs, gt_trajs))
            minfde[n] = min(compute_world_fde(forecasted_trajs, gt_trajs))
        
        self.log("BrierMinFDE", np.mean(brierminfde), on_step=True, on_epoch=True, prog_bar=True, logger=True) 
        self.log("MinADE", np.mean(minade), on_step=True, on_epoch=True, prog_bar=True, logger=True) 
        self.log("MinFDE", np.mean(minfde), on_step=True, on_epoch=True, prog_bar=True, logger=True) 
        return -1
     

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


class MotionDecoder(nn.Module):
    def __init__(
            self,
            max_delta: float, #meters
            sampling_rate: int,
            n_target: int,
            emb_dim: int,
            enc_dim: int,
            dropout_rate: int,
            n_rollouts: int,
            n_quantization_bins: int = 128,
            n_verlet_steps: int = 13,
            n_time_steps: int = 110,
            time_step_end: int = 49,
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
        self.register_buffer("time_indices_type", time_indices)

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

        self.scene_emb_layers = nn.Sequential(
            nn.Linear(enc_dim, enc_dim + 64),
            nn.GELU(),
            nn.Linear(enc_dim + 64, self.emb_dim)
        )

        self.fully_connected_layers = nn.Sequential(
            nn.Linear(self.emb_dim, 512),
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

        n_batch = motion_tokens.shape[0]
        n_agents = motion_tokens.shape[1]
        n_steps = motion_tokens.shape[2]
        time_indices = torch.arange(0, n_steps).type_as(self.time_indices_type)

        # quantized, discretized, verlet-wrapped motion tokens
        motion_tokens, last_token = tokenize_motion(motion_tokens, # [n_batch, n_agents, n_time_steps, 1]
                                        self.pos_bins, 
                                        self.verlet_wrapper, 
                                        self.n_verlet_steps)
        
        # we compute a learned value embedding and two learned positional embeddings (representing the timestep and agent identity) for each discrete motion token, which are combined via an element-wise sum prior to being input to the transformer decoder.
        val_embeddings = self.val_emb_layer(motion_tokens)
        # todo: is repeating embeddings a good idea?!
        step_embeddings = self.step_emb_layer(time_indices
                                              ).unsqueeze(0
                                              ).repeat(n_agents, 1, 1
                                              ).unsqueeze(0
                                              ).repeat(n_batch, 1, 1, 1)
        type_embeddings = self.type_emb_layer(target_types.int().argmax(dim=-1)
                                              ).unsqueeze(2
                                              ).repeat(1, 1, n_steps, 1)
        
        # [n_batch, n_agents * n_time_steps, emb_dim]
        motion_embeddings = (val_embeddings + 
                             step_embeddings + 
                             type_embeddings
                             ).flatten(1, 2) 
        
        #self attending motion tokens + cross attending to scene emebeddings
        query = motion_embeddings
        # ensure scene embedding matches decoder dimension
        fused_emb = self.scene_emb_layers(fused_emb)
        key = fused_emb
        # MotionLM repeats embeddings across rollouts before cross-attention only, here we repeat for both self and cross attention
        for decoder_block in self.decoder_layers:
            if query.shape[0] != n_batch :
                query = query.unflatten(dim=0, sizes=(n_batch, n_agents)).flatten(1, 2)
            attn_mask = get_attention_mask(n_steps, query.shape[1]).type_as(self.attn_type) # type_as casting simply to move to right device with Lightning
            query = decoder_block(query = query, 
                                  key = key, 
                                  attn_mask = attn_mask,
                                  n_agents = n_agents,
                                  n_steps = n_steps)
            
        out = self.fully_connected_layers(query)
        return out, last_token