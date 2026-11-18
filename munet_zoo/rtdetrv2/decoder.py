# Adapted for MuNet from lyuwenyu/RT-DETR (rtdetrv2_pytorch), commit e97db5a685a874c01c7977cb3a42ab0684ea134b.
# Original work Copyright (c) 2023 lyuwenyu and RT-DETR contributors, Apache-2.0.
# MuNet changes: native graph operations, NumPy initialization, prepared denoising inputs, constant anchors.
# See LICENSE-RT-DETR and NOTICE in this directory.
"""RT-DETRv2 decoder: query selection and the iterative multi-scale deformable decoder."""
import copy
from collections import OrderedDict

import numpy as np
from munet import nn
from munet.core import cat, stack
from munet.nn import functional as F

from . import initialization as init
from .utils import bias_init_with_prob, deformable_attention_core_func_v2, get_activation, inverse_sigmoid

__all__ = ['RTDETRTransformerv2']


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act='relu'):
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.act = get_activation(act)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MSDeformableAttention(nn.Module):
    """Multi-scale deformable attention. num_points is one count for every level or a list with one count per level."""
    def __init__(self, embed_dim=256, num_heads=8, num_levels=4, num_points=4, method='default', offset_scale=0.5):
        if embed_dim % num_heads: raise ValueError("embed_dim must be divisible by num_heads")
        if method not in ('default', 'discrete'): raise ValueError("cross attention method must be default or discrete")
        points = list(num_points) if isinstance(num_points, (list, tuple)) else [num_points] * num_levels
        if len(points) != num_levels: raise ValueError("num_points needs one entry per level")
        self.embed_dim, self.num_heads, self.num_levels = embed_dim, num_heads, num_levels
        self.head_dim = embed_dim // num_heads
        self.method, self.offset_scale = method, offset_scale
        self.num_points_list = points
        self.num_points_scale = np.array([1 / n for n in points for _ in range(n)], np.float32).reshape(-1, 1)
        self.total_points = num_heads * sum(points)

        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2)
        self.attention_weights = nn.Linear(embed_dim, self.total_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self._reset_parameters()
        if method == 'discrete': self.sampling_offsets.requires_grad_(False)

    def _reset_parameters(self):
        init.constant_(self.sampling_offsets.weight, 0)
        thetas = np.arange(self.num_heads, dtype=np.float32) * (2 * np.pi / self.num_heads)
        grid = np.stack([np.cos(thetas), np.sin(thetas)], axis=-1)
        grid = grid / np.abs(grid).max(axis=-1, keepdims=True)
        grid = np.tile(grid[:, None, :], (1, sum(self.num_points_list), 1))
        grid *= np.concatenate([np.arange(1, n + 1, dtype=np.float32) for n in self.num_points_list])[None, :, None]
        self.sampling_offsets.bias.assign(grid.reshape(-1))

        init.constant_(self.attention_weights.weight, 0)
        init.constant_(self.attention_weights.bias, 0)
        init.xavier_uniform_(self.value_proj.weight)
        init.constant_(self.value_proj.bias, 0)
        init.xavier_uniform_(self.output_proj.weight)
        init.constant_(self.output_proj.bias, 0)

    def forward(self, query, reference_points, value, value_spatial_shapes, value_mask=None):
        """query [bs, q, C]; reference_points [bs, q, 1, 4] as cx, cy, w, h in [0, 1]; value [bs, sum(h*w), C];
        value_mask [bs, sum(h*w)], 1 for real positions. Returns [bs, q, C]."""
        if reference_points.shape[-1] != 4: raise ValueError("reference points must be cx, cy, w, h")
        bs, len_q = query.shape[:2]
        total = sum(self.num_points_list)
        value = self.value_proj(value)
        if value_mask is not None: value = value * value_mask.unsqueeze(-1)
        value = value.reshape(bs, value.shape[1], self.num_heads, self.head_dim)

        offsets = self.sampling_offsets(query).reshape(bs, len_q, self.num_heads, total, 2)
        weights = self.attention_weights(query).reshape(bs, len_q, self.num_heads, total).softmax(-1)
        offsets = offsets * self.num_points_scale * reference_points[:, :, None, :, 2:] * self.offset_scale
        locations = reference_points[:, :, None, :, :2] + offsets
        output = deformable_attention_core_func_v2(value, value_spatial_shapes, locations, weights, self.num_points_list, self.method)
        return self.output_proj(output)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, n_head=8, dim_feedforward=1024, dropout=0., activation='relu', n_levels=4, n_points=4,
                 cross_attn_method='default'):
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.cross_attn = MSDeformableAttention(d_model, n_head, n_levels, n_points, method=cross_attn_method)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = get_activation(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        init.xavier_uniform_(self.linear1.weight)
        init.xavier_uniform_(self.linear2.weight)

    def forward(self, target, reference_points, memory, memory_spatial_shapes, attn_mask=None, memory_mask=None, query_pos_embed=None):
        q = k = target if query_pos_embed is None else target + query_pos_embed
        target2, _ = self.self_attn(q, k, value=target, attn_mask=attn_mask, need_weights=False)
        target = self.norm1(target + self.dropout1(target2))

        target2 = self.cross_attn(target if query_pos_embed is None else target + query_pos_embed,
                                  reference_points, memory, memory_spatial_shapes, memory_mask)
        target = self.norm2(target + self.dropout2(target2))

        target2 = self.linear2(self.dropout3(self.activation(self.linear1(target))))
        return self.norm3(target + self.dropout4(target2))


class TransformerDecoder(nn.Module):
    def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1):
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx

    def step(self, i, output, ref_points, memory, memory_spatial_shapes, bbox_head, score_head, query_pos_head,
             attn_mask=None, memory_mask=None):
        """Layer i alone. ref_points: the boxes the layer before it refined (for layer 0, the sigmoid of the initial
        ones). Returns the layer's output, the boxes it refined, and its logits and boxes for the loss (None in eval
        mode for every layer but eval_idx)."""
        ref_points_detach = ref_points if i == 0 else ref_points.detach()
        query_pos_embed = query_pos_head(ref_points_detach)
        output = self.layers[i](output, ref_points_detach.unsqueeze(2), memory, memory_spatial_shapes, attn_mask, memory_mask, query_pos_embed)
        inter_ref_bbox = F.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points_detach))
        if self.training:
            logits = score_head[i](output)
            return output, inter_ref_bbox, logits, inter_ref_bbox if i == 0 else F.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points))
        if i == self.eval_idx: return output, inter_ref_bbox, score_head[i](output), inter_ref_bbox
        return output, inter_ref_bbox, None, None

    def forward(self, target, ref_points_unact, memory, memory_spatial_shapes, bbox_head, score_head, query_pos_head,
                attn_mask=None, memory_mask=None):
        dec_out_bboxes = []
        dec_out_logits = []
        output, ref_points = target, F.sigmoid(ref_points_unact)
        for i in range(len(self.layers)):
            output, ref_points, logits, boxes = self.step(i, output, ref_points, memory, memory_spatial_shapes, bbox_head, score_head,
                                                          query_pos_head, attn_mask, memory_mask)
            if logits is not None:
                dec_out_logits.append(logits)
                dec_out_bboxes.append(boxes)
            if not self.training and i == self.eval_idx: break
        return stack(dec_out_bboxes), stack(dec_out_logits)


class RTDETRTransformerv2(nn.Module):
    """query_select_method: 'default' ranks positions by their best class score, 'one2many' ranks position/class pairs,
    'agnostic' ranks by a single objectness score. cross_attn_method: 'default' (bilinear) or 'discrete' (nearest cell)."""
    def __init__(self, num_classes=80, hidden_dim=256, num_queries=300, feat_channels=(512, 1024, 2048), feat_strides=(8, 16, 32),
                 num_levels=3, num_points=4, nhead=8, num_layers=6, dim_feedforward=1024, dropout=0., activation='relu',
                 num_denoising=100, label_noise_ratio=0.5, box_noise_scale=1.0, learn_query_content=False, eval_spatial_size=None,
                 eval_idx=-1, eps=1e-2, aux_loss=True, cross_attn_method='default', query_select_method='default'):
        if len(feat_channels) > num_levels or len(feat_strides) != len(feat_channels): raise ValueError("feature channels, strides and levels disagree")
        if query_select_method not in ('default', 'one2many', 'agnostic'): raise ValueError("query selection must be default, one2many or agnostic")
        feat_strides = list(feat_strides)
        for _ in range(num_levels - len(feat_strides)): feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim, self.nhead = hidden_dim, nhead
        self.feat_strides, self.num_levels = feat_strides, num_levels
        self.num_classes, self.num_queries = num_classes, num_queries
        self.eps, self.num_layers = eps, num_layers
        self.eval_spatial_size, self.aux_loss = eval_spatial_size, aux_loss
        self.cross_attn_method, self.query_select_method = cross_attn_method, query_select_method

        self._build_input_proj_layer(feat_channels)

        decoder_layer = TransformerDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout, activation, num_levels, num_points,
                                                cross_attn_method=cross_attn_method)
        self.decoder = TransformerDecoder(hidden_dim, decoder_layer, num_layers, eval_idx)

        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        if num_denoising > 0:
            self.denoising_class_embed = nn.Embedding(num_classes + 1, hidden_dim, padding_idx=num_classes)

        self.learn_query_content = learn_query_content
        if learn_query_content: self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, 2)

        self.enc_output = nn.Sequential(OrderedDict([('proj', nn.Linear(hidden_dim, hidden_dim)), ('norm', nn.LayerNorm(hidden_dim))]))
        self.enc_score_head = nn.Linear(hidden_dim, 1 if query_select_method == 'agnostic' else num_classes)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3)

        self.dec_score_head = nn.ModuleList([nn.Linear(hidden_dim, num_classes) for _ in range(num_layers)])
        self.dec_bbox_head = nn.ModuleList([MLP(hidden_dim, hidden_dim, 4, 3) for _ in range(num_layers)])

        if self.eval_spatial_size: self.anchors, self.valid_mask = self._generate_anchors()
        self._reset_parameters()

    def _reset_parameters(self):
        bias = bias_init_with_prob(0.01)
        init.constant_(self.enc_score_head.bias, bias)
        init.constant_(self.enc_bbox_head.layers[-1].weight, 0)
        init.constant_(self.enc_bbox_head.layers[-1].bias, 0)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            init.constant_(cls_.bias, bias)
            init.constant_(reg_.layers[-1].weight, 0)
            init.constant_(reg_.layers[-1].bias, 0)
        init.xavier_uniform_(self.enc_output.proj.weight)
        if self.learn_query_content: init.xavier_uniform_(self.tgt_embed.weight)
        init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)
        for proj in self.input_proj: init.xavier_uniform_(proj.conv.weight)

    def _build_input_proj_layer(self, feat_channels):
        self.input_proj = nn.ModuleList()
        for in_channels in feat_channels:
            self.input_proj.append(nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(in_channels, self.hidden_dim, 1, bias=False)),
                ('norm', nn.BatchNorm2d(self.hidden_dim))])))
        in_channels = feat_channels[-1]
        for _ in range(self.num_levels - len(feat_channels)):
            self.input_proj.append(nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(in_channels, self.hidden_dim, 3, 2, padding=1, bias=False)),
                ('norm', nn.BatchNorm2d(self.hidden_dim))])))
            in_channels = self.hidden_dim

    def _get_encoder_input(self, feats):
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        for i in range(len(proj_feats), self.num_levels):
            proj_feats.append(self.input_proj[i](feats[-1] if i == len(feats) else proj_feats[-1]))
        # [b, c, h, w] -> [b, h*w, c], levels concatenated
        return cat([feat.flatten(2).permute(0, 2, 1) for feat in proj_feats], 1), [list(feat.shape[2:]) for feat in proj_feats]

    def _generate_anchors(self, spatial_shapes=None, grid_size=0.05):
        if spatial_shapes is None:
            spatial_shapes = [(self.eval_spatial_size[0] // s, self.eval_spatial_size[1] // s) for s in self.feat_strides]
        anchors = []
        for level, (h, w) in enumerate(spatial_shapes):
            yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
            xy = (np.stack([xx, yy], axis=-1) + 0.5) / np.array([w, h], np.float32)
            wh = np.full_like(xy, grid_size * (2 ** level))
            anchors.append(np.concatenate([xy, wh], axis=-1).reshape(1, h * w, 4))
        anchors = np.concatenate(anchors, axis=1)
        valid = np.all((anchors > self.eps) & (anchors < 1 - self.eps), axis=-1, keepdims=True)
        return np.where(valid, np.log(anchors / (1 - anchors)), np.inf).astype(np.float32), valid.astype(np.float32)

    def _select_topk(self, memory, logits, coords_unact):
        if self.query_select_method == 'default':
            _, index = logits.amax(-1).topk(self.num_queries, dim=1)
        elif self.query_select_method == 'one2many':
            _, index = logits.flatten(1).topk(self.num_queries, dim=1)
            index = (index / self.num_classes).floor()
        else:
            _, index = logits.squeeze(-1).topk(self.num_queries, dim=1)
        bs = memory.shape[0]
        pick = lambda x: x.gather(dim=1, index=index.unsqueeze(-1).expand(bs, self.num_queries, x.shape[-1]))
        return pick(memory), pick(logits), pick(coords_unact)

    def _get_decoder_input(self, memory, spatial_shapes, denoising_class=None, denoising_bbox_unact=None):
        if self.training or self.eval_spatial_size is None: anchors, valid_mask = self._generate_anchors(spatial_shapes)
        else: anchors, valid_mask = self.anchors, self.valid_mask

        output_memory = self.enc_output(memory * valid_mask)
        enc_logits = self.enc_score_head(output_memory)
        enc_coords_unact = self.enc_bbox_head(output_memory) + anchors
        topk_memory, topk_logits, topk_coords_unact = self._select_topk(output_memory, enc_logits, enc_coords_unact)
        enc_aux = [{'pred_logits': topk_logits, 'pred_boxes': F.sigmoid(topk_coords_unact)}] if self.training else []

        if self.learn_query_content: content = self.tgt_embed.weight.unsqueeze(0).expand(memory.shape[0], -1, -1)
        else: content = topk_memory.detach()
        ref_points_unact = topk_coords_unact.detach()
        if denoising_bbox_unact is not None:
            ref_points_unact = cat([denoising_bbox_unact, ref_points_unact], 1)
            content = cat([denoising_class, content], 1)
        return content, ref_points_unact, enc_aux

    def _denoising_meta(self, denoising):
        return {"dn_num_group": denoising["num_group"], "max_gt": denoising["max_gt"],
                "dn_num_split": [denoising["labels"].shape[1], self.num_queries]}

    def queries(self, feats, denoising=None):
        """Everything before the decoder layers. Returns the flattened features, their spatial shapes, the query content,
        the initial reference boxes (unactivated), the attention mask, the encoder's auxiliary outputs and the denoising
        split (None without denoising)."""
        memory, spatial_shapes = self._get_encoder_input(feats)

        if self.training and denoising is not None:
            denoising_class = self.denoising_class_embed(denoising["labels"])
            denoising_bbox_unact = denoising["boxes_unact"]
            attn_mask = denoising["attn_mask"]
            dn_meta = self._denoising_meta(denoising)
        else:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

        content, ref_points_unact, enc_aux = self._get_decoder_input(memory, spatial_shapes, denoising_class, denoising_bbox_unact)
        return memory, spatial_shapes, content, ref_points_unact, attn_mask, enc_aux, dn_meta

    def outputs(self, out_bboxes, out_logits, enc_aux, dn_meta):
        """The model's output from the boxes and logits of the decoder layers, stacked on dim 0."""
        if self.training and dn_meta is not None:
            dn_out_bboxes, out_bboxes = out_bboxes.split(dn_meta['dn_num_split'], dim=2)
            dn_out_logits, out_logits = out_logits.split(dn_meta['dn_num_split'], dim=2)

        out = {'pred_logits': out_logits[-1], 'pred_boxes': out_bboxes[-1]}
        if self.training and self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(out_logits[:-1], out_bboxes[:-1]) if out_logits.shape[0] > 1 else []
            out['enc_aux_outputs'] = enc_aux
            out['enc_meta'] = {'class_agnostic': self.query_select_method == 'agnostic'}
            if dn_meta is not None:
                out['dn_aux_outputs'] = self._set_aux_loss(dn_out_logits, dn_out_bboxes)
                out['dn_meta'] = dn_meta
        return out

    def forward(self, feats, denoising=None):
        """denoising: the dict from prepare_denoising (training only), or None."""
        memory, spatial_shapes, content, ref_points_unact, attn_mask, enc_aux, dn_meta = self.queries(feats, denoising)
        out_bboxes, out_logits = self.decoder(content, ref_points_unact, memory, spatial_shapes, self.dec_bbox_head,
                                              self.dec_score_head, self.query_pos_head, attn_mask=attn_mask)
        return self.outputs(out_bboxes, out_logits, enc_aux, dn_meta)

    @staticmethod
    def _set_aux_loss(outputs_class, outputs_coord):
        return [{'pred_logits': a, 'pred_boxes': b} for a, b in zip(outputs_class, outputs_coord)]
