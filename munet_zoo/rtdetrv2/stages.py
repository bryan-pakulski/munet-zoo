"""RT-DETRv2 cut into stage functions for a training pool (``munet.pool.Pool``), so that no node has to hold all of it."""
from munet.core import stack
from munet.nn import functional as F


def pool_stages(model, criterion, denoising=None, *, layers_per_stage=1):
    """Stage functions for ``munet.pool.Pool(stages, ...)``: the backbone, the encoder, query selection, then the decoder
    layers ``layers_per_stage`` at a time. The last stage also computes the loss.

    A shard is ``[images, labels, boxes, valid]`` (``pad_targets``), followed by ``labels``, ``boxes_unact`` and
    ``attn_mask`` of ``prepare_denoising`` when ``denoising`` is given. denoising: one ``prepare_denoising`` result; every
    shard must have its ``num_group`` and ``max_gt``. The model must be in training mode.
    """
    decoder, levels = model.decoder, len(model.backbone.out_strides)
    count = decoder.num_layers
    if not model.training: raise ValueError("pool stages train the model: call model.train() first")
    if not 1 <= layers_per_stage <= count: raise ValueError("layers_per_stage must be between 1 and the number of decoder layers")

    def payload(noise):
        if denoising is None: return None
        return {'labels': noise[0], 'boxes_unact': noise[1], 'attn_mask': noise[2],
                'num_group': denoising['num_group'], 'max_gt': denoising['max_gt']}

    def shapes(images):
        return [[images.shape[2] // s, images.shape[3] // s] for s in decoder.feat_strides]

    def backbone(images, *rest):
        return tuple(model.backbone(images if model.mean is None else (images - model.mean) / model.std))

    def encoder(*args):
        return tuple(model.encoder(list(args[:levels])))

    def queries(*args):
        images, noise = args[levels], args[levels + 4:]
        memory, spatial, content, ref_points_unact, _, enc_aux, _ = decoder.queries(list(args[:levels]), payload(noise))
        if [list(s) for s in spatial] != shapes(images): raise ValueError("feature maps are not the image size over each stride")
        return memory, content, F.sigmoid(ref_points_unact), enc_aux[0]['pred_logits'], enc_aux[0]['pred_boxes']

    def layers(first, last):
        # Receives the flattened features, the queries, their reference boxes, the encoder's auxiliary logits and boxes,
        # then the logits and boxes of every layer before `first`.
        def stage(*args):
            memory, output, ref_points, enc_logits, enc_boxes = args[:5]
            done = list(args[5:5 + 2 * first])
            images, labels, boxes, valid, *noise = args[5 + 2 * first:]
            for i in range(first, last):
                output, ref_points, logits, refined = decoder.decoder.step(
                    i, output, ref_points, memory, shapes(images), decoder.dec_bbox_head, decoder.dec_score_head,
                    decoder.query_pos_head, attn_mask=None if denoising is None else noise[2])
                done += [logits, refined]
            if last < count: return (memory, output, ref_points, enc_logits, enc_boxes, *done)
            out = decoder.outputs(stack(done[1::2]), stack(done[0::2]), [{'pred_logits': enc_logits, 'pred_boxes': enc_boxes}],
                                  None if denoising is None else decoder._denoising_meta(payload(noise)))
            return criterion.total(criterion(out, {'labels': labels, 'boxes': boxes, 'valid': valid}))
        return stage

    return [backbone, encoder, queries, *(layers(first, min(first + layers_per_stage, count)) for first in range(0, count, layers_per_stage))]
