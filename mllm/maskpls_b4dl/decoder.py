# Adapted from PRBonn/MaskPLS @ 84881ba8; see LICENSE-MaskPLS.txt and NOTICE.md.
# Modified by Rodrigo Marcuzzi from https://github.com/facebookresearch/Mask2Former
from . import blocks
from collections import OrderedDict
import hashlib
import torch
from .positional_encoder import PositionalEncoder
from torch import nn
from torch.utils.checkpoint import checkpoint
from .query_init import farthest_point_indices


class MaskedTransformerDecoder(nn.Module):
    def __init__(self, cfg, bb_cfg, data_cfg):
        super().__init__()
        hidden_dim = cfg.HIDDEN_DIM

        cfg.POS_ENC.FEAT_SIZE = cfg.HIDDEN_DIM

        self.pe_layer = PositionalEncoder(cfg.POS_ENC)

        self.num_layers = cfg.FEATURE_LEVELS * cfg.DEC_BLOCKS
        self.nheads = cfg.NHEADS
        self.checkpoint_decoder = cfg.CHECKPOINT

        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                blocks.SelfAttentionLayer(
                    d_model=hidden_dim, nhead=self.nheads, dropout=0.0
                )
            )
            self.transformer_cross_attention_layers.append(
                blocks.CrossAttentionLayer(
                    d_model=hidden_dim, nhead=self.nheads, dropout=0.0
                )
            )
            self.transformer_ffn_layers.append(
                blocks.FFNLayer(
                    d_model=hidden_dim, dim_feedforward=cfg.DIM_FFN, dropout=0.0
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = cfg.NUM_QUERIES
        self.num_feature_levels = cfg.FEATURE_LEVELS

        self.query_feat = nn.Embedding(cfg.NUM_QUERIES, hidden_dim)
        self.query_embed = nn.Embedding(cfg.NUM_QUERIES, hidden_dim)
        self.query_initialization = getattr(cfg, "QUERY_INITIALIZATION", "learned")
        self.query_feat.requires_grad_(self.query_initialization == "learned")
        self.query_embed.requires_grad_(self.query_initialization != "zero_fps")
        self._fps_cache = OrderedDict()  # only input hashes and row ids, no labels/features
        self.last_query_point_indices = None
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)

        self.mask_feat_proj = nn.Sequential()
        in_channels = bb_cfg.CHANNELS
        if in_channels[-1] != hidden_dim:
            self.mask_feat_proj = nn.Linear(in_channels[-1], hidden_dim)

        in_channels = in_channels[:-1][-self.num_feature_levels :]
        self.input_proj = nn.ModuleList()
        for ch in in_channels:
            if ch != hidden_dim:  # linear projection to hidden_dim
                self.input_proj.append(nn.Linear(ch, hidden_dim))
            else:
                self.input_proj.append(nn.Sequential())

        # output FFNs
        self.class_embed = nn.Linear(hidden_dim, data_cfg.NUM_CLASSES + 1)
        self.mask_embed = blocks.MLP(hidden_dim, hidden_dim, cfg.HIDDEN_DIM, 3)
        self.mask_residual_mode = getattr(cfg, "MASK_RESIDUAL_MODE", "none")
        if self.mask_residual_mode != "none":
            # Both treatment and capacity control have exactly the same head.
            # A zero final layer preserves every parent logit at initialization;
            # its gradients open the branch without a frozen zero scalar gate.
            self.mask_residual_embed = blocks.MLP(hidden_dim, hidden_dim, hidden_dim, 3)
            nn.init.zeros_(self.mask_residual_embed.layers[-1].weight)
            nn.init.zeros_(self.mask_residual_embed.layers[-1].bias)

    def forward(self, feats, coors, pad_masks):
        # Do not mutate the backbone feature containers (used by the auxiliary head).
        feats, coors, pad_masks = list(feats), list(coors), list(pad_masks)
        last_coors = coors.pop()
        feature_basis = self.mask_feat_proj(feats.pop())
        position_basis = self.pe_layer(last_coors)
        mask_features = feature_basis + position_basis
        residual_basis = (feature_basis if self.mask_residual_mode == "feature" else
                          position_basis if self.mask_residual_mode == "position" else None)
        last_pad = pad_masks.pop()
        src = []
        pos = []
        size_list = []

        for i in range(self.num_feature_levels):
            size_list.append(feats[i].shape[1])
            pos.append(self.pe_layer(coors[i]))
            feat = self.input_proj[i](feats[i])
            src.append(feat)

        bs = src[0].shape[0]
        query_embed, output = self.initialize_queries(last_coors, last_pad)

        predictions_class = []
        predictions_mask = []

        # predictions on learnable query features, first attn_mask
        outputs_class, outputs_mask, attn_mask = self.pred_heads(
            output,
            mask_features,
            pad_mask=last_pad,
            residual_basis=residual_basis,
        )
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            if attn_mask is not None:
                attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            args = (i, output, src[level_index], attn_mask,
                    pad_masks[level_index], pos[level_index], query_embed)
            if self.training and self.checkpoint_decoder and torch.is_grad_enabled():
                output = checkpoint(self.decode_layer, *args, use_reentrant=False)
            else:
                output = self.decode_layer(*args)

            # get predictions and attn mask for next feature level
            outputs_class, outputs_mask, attn_mask = self.pred_heads(
                output,
                mask_features,
                pad_mask=last_pad,
                residual_basis=residual_basis,
            )
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        assert len(predictions_class) == self.num_layers + 1

        out = {"pred_logits": predictions_class[-1], "pred_masks": predictions_mask[-1],
               "queries": self.decoder_norm(output)}

        out["aux_outputs"] = self.set_aux(predictions_class, predictions_mask)

        return out, last_pad

    def initialize_queries(self, coordinates, padding):
        bs = coordinates.shape[0]
        self.last_query_point_indices = None
        if self.query_initialization == "zero_fps":
            positions, indices = [], []
            for xyz, pad in zip(coordinates, padding):
                valid_ids = (~pad).nonzero(as_tuple=True)[0]
                raw = xyz[valid_ids].detach().cpu().numpy()
                key = hashlib.sha256(raw.tobytes()).hexdigest()
                if key not in self._fps_cache:
                    self._fps_cache[key] = farthest_point_indices(raw, self.num_queries)
                    if len(self._fps_cache) > 2048:
                        self._fps_cache.popitem(last=False)
                local = torch.as_tensor(self._fps_cache[key], device=xyz.device)
                ids = valid_ids[local]
                positions.append(xyz[ids])
                indices.append(ids.detach().cpu())
            query_embed = self.pe_layer(torch.stack(positions))
            self.last_query_point_indices = torch.stack(indices)
        else:
            query_embed = self.query_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
        output = (self.query_feat.weight.unsqueeze(0).repeat(bs, 1, 1)
                  if self.query_initialization == "learned" else torch.zeros_like(query_embed))
        return query_embed, output

    def decode_layer(self, index, output, memory, attn_mask, padding, position, query_embed):
        output = self.transformer_cross_attention_layers[index](
            output, memory, attn_mask=attn_mask, padding_mask=padding,
            pos=position, query_pos=query_embed)
        output = self.transformer_self_attention_layers[index](
            output, attn_mask=None, padding_mask=None, query_pos=query_embed)
        return self.transformer_ffn_layers[index](output)

    def pred_heads(
        self,
        output,
        mask_features,
        pad_mask=None,
        residual_basis=None,
    ):
        decoder_output = self.decoder_norm(output)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bpc->bpq", mask_embed, mask_features)
        if self.mask_residual_mode != "none":
            if residual_basis is None:
                raise ValueError("active residual mask head requires its input basis")
            residual_embed = self.mask_residual_embed(decoder_output)
            outputs_mask = outputs_mask + torch.einsum("bqc,bpc->bpq", residual_embed, residual_basis)

        attn_mask = (outputs_mask.sigmoid() < 0.5).detach().bool()
        attn_mask[pad_mask] = True
        attn_mask = (
            attn_mask.unsqueeze(1)
            .repeat(1, self.nheads, 1, 1)
            .flatten(0, 1)
            .permute(0, 2, 1)
        )

        return outputs_class, outputs_mask, attn_mask

    @torch.jit.unused
    def set_aux(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_masks": b}
            for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
        ]
