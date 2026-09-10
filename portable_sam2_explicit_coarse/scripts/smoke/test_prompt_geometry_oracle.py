"""CPU tests of actual geometry hooks and production support rasterizer."""
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("RSPROMPTER_LIGHT_IMPORT", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from inference.probes.prompt_geometry_oracle_probe import GeometryHooks, expanded_boxes
from rsprompter.sam2_mask_head_helpers import _MaskHeadPromptHelpers


class PE(torch.nn.Module):
    def forward(self, *, points, boxes, masks):
        return points, boxes, masks


class Head(torch.nn.Module):
    _dense_embedding_box_support = _MaskHeadPromptHelpers._dense_embedding_box_support
    prompt_encoder_image_size = 64

    def __init__(self):
        super().__init__()
        self.prompt_encoder = PE()

    def _forward_dense_embeddings(self, image, dense_pe, shape_dense_valid, boxes):
        return dense_pe * self._dense_embedding_box_support(boxes, (16, 16), dense_pe.dtype)

    def forward(self, *, boxes, roi_img_ids):
        coords = boxes[:, None, :2].expand(-1, 4, -1)
        labels = torch.tensor([[1, 1, 0, 0]]).expand(len(boxes), -1)
        masks = torch.ones(len(boxes), 1, 16, 16)
        pe = self.prompt_encoder(points=(coords, labels), boxes=boxes, masks=masks)
        dense = self._forward_dense_embeddings(None, pe[2], None, boxes)
        return pe, dense


def main():
    boxes = torch.tensor([[10., 10., 30., 30.], [10., 10., 30., 30.], [40., 40., 50., 50.]])
    original = boxes.clone()
    out = expanded_boxes(boxes, torch.tensor([True, False, False]))
    assert torch.equal(out[0], torch.tensor([9., 9., 31., 31.]))
    assert torch.equal(out[1:], boxes[1:]) and torch.equal(boxes, original)
    head = Head()
    gt_boxes = torch.tensor([[11., 11., 31., 31.]])
    gt_labels = torch.tensor([0])
    gts = [(gt_boxes, gt_labels, torch.ones(1, 64, 64))]
    # Same geometry but wrong class must remain unmatched, as must distant ROI.
    results = [SimpleNamespace(labels=torch.tensor([0, 1, 0]))]
    roi = SimpleNamespace(mask_head=head)
    roi.predict_mask = lambda x, metas, results_list: head(boxes=boxes, roi_img_ids=torch.zeros(3, dtype=torch.long))
    model = SimpleNamespace(roi_head=roi)
    before = roi.predict_mask
    standard = before(None, None, results)
    summaries = {}
    for mode in ("raw", "box_gt", "support_expand"):
        hooks = GeometryHooks(model, mode)
        hooks.state.update(gts=gts, image_ids=[123], row_counts=[0])
        try:
            hooks.install()
            actual = roi.predict_mask(None, None, results)
        finally:
            hooks.close()
        assert roi.predict_mask is before
        assert "_dense_embedding_box_support" not in head.__dict__
        assert not head._forward_pre_hooks and not head.prompt_encoder._forward_pre_hooks
        assert hooks.counts["matched"] == 1 and hooks.counts["proposals"] == 3
        assert [r["target_index"] for r in hooks.rows] == [0, None, None]
        assert [r["row"] for r in hooks.rows] == [0, 1, 2]
        assert all(r["image_id"] == 123 for r in hooks.rows)
        assert torch.equal(boxes, original)
        assert torch.equal(actual[0][0][0], standard[0][0][0])
        assert torch.equal(actual[0][0][1], standard[0][0][1])
        assert torch.equal(actual[0][2], standard[0][2])
        if mode == "box_gt":
            assert torch.equal(actual[0][1][0], gt_boxes[0])
            assert torch.equal(actual[0][1][1:], boxes[1:])
            assert torch.equal(actual[1], standard[1])
        else:
            assert torch.equal(actual[0][1], boxes)
        if mode == "raw":
            assert torch.equal(actual[1], standard[1])
        summaries[mode] = {k: v.hexdigest() for k, v in hooks.hashes.items()}
    for key in ("points_coords", "points_labels", "canvas", "pre_support_dense", "roi_boxes"):
        assert len({value[key] for value in summaries.values()}) == 1
    assert summaries["raw"]["support_grid"] == summaries["box_gt"]["support_grid"]
    assert summaries["raw"]["pe_boxes_after"] == summaries["support_expand"]["pe_boxes_after"]
    print("PASS: expansion ratio, immutable inputs, class-aware/unmatched rules, actual PE-only/support-only hooks, row identity, tensor hashes, restoration")


if __name__ == "__main__":
    main()
