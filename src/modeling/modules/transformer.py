import torch
from torch import nn
import torch.nn.functional as F


class TransformerDecoder(nn.Module):
    def __init__(
            self,
            num_heads: int,
            emb_dim: int,
            dropout_rate: float,
            n_time_steps: int,
            n_rollouts: int
            ):
        super().__init__()
        self.num_heads = num_heads
        self.emb_dim = emb_dim
        self.dropout_rate = dropout_rate
        self.n_time_steps = n_time_steps
        self.n_rollouts = n_rollouts
        
        self.self_attn = nn.MultiheadAttention(embed_dim = self.emb_dim, 
                                        num_heads = self.num_heads,
                                        dropout = self.dropout_rate,
                                        batch_first=True)

        self.cross_attn = nn.MultiheadAttention(embed_dim = self.emb_dim, 
                                                num_heads = self.num_heads,
                                                dropout = self.dropout_rate,
                                                batch_first=True)
        
        self.layer_norm_1 = nn.LayerNorm(normalized_shape = self.emb_dim)
        self.layer_norm_2 = nn.LayerNorm(normalized_shape = self.emb_dim)
        self.layer_norm_3 = nn.LayerNorm(normalized_shape = self.emb_dim)

        self.ffn_1 = nn.Linear(in_features=self.emb_dim, out_features=emb_dim)
        
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self,
                query,
                key,
                attn_mask,
                key_padding_mask,
                cross_attn_mask,
                invalid_agents,
                n_agents,
                n_steps,
                ):
        attn_out_1, _ = self.self_attn(query = query, 
                                        key = query, 
                                        value = query, 
                                        attn_mask = attn_mask,
                                        key_padding_mask = key_padding_mask)
        #[n_rollouts * n_batch * n_agents, n_time_steps, emb_dim]
        attn_out_2 = self.layer_norm_1(query + attn_out_1
                                         ).unflatten(dim=1, sizes=(n_agents, n_steps)
                                         ).flatten(0, 1) 
        #[n_rollouts * n_batch * n_agents, : , emb_dim]
        value = key
        # cross attend each motion token to corresponding scene embeddings,
        attn_out_3, _ = self.cross_attn(query = attn_out_2, 
                                        key = key, 
                                        value = value, 
                                        key_padding_mask = cross_attn_mask)

        attn_out_3[invalid_agents,] = 0.0
        attn_out_4 = self.layer_norm_2(attn_out_2 + attn_out_3)
        #attn_out_4 = self.layer_norm_2(attn_out_2 + torch.nan_to_num(attn_out_3, nan=0.0)) # to handle invalid agents
        #feed-forward
        ffn_out_1 = F.gelu(self.ffn_1(attn_out_4))
        out = self.layer_norm_3(self.dropout(ffn_out_1 + attn_out_4))

        return out

       