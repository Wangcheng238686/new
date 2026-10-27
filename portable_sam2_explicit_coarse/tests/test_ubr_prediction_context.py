"""CPU contracts for a prediction-only helper, not detector integration tests."""

import importlib.util
from pathlib import Path
import sys
import unittest

import torch


_PATH = Path(__file__).resolve().parents[1] / 'rsprompter/uav/ubr_prediction_context.py'
_SPEC = importlib.util.spec_from_file_location('ubr_prediction_context', _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
build_prediction_context = _MODULE.build_prediction_context
select_prediction_candidates = _MODULE.select_prediction_candidates


class PredictionContextTests(unittest.TestCase):
    def inputs(self):
        return dict(
            boxes=torch.tensor([[0., 0., 4., 4.], [0., 0., 4., 4.],
                                [6., 0., 8., 2.], [0., 0., 4., 4.]]),
            scores=torch.tensor([.8, .8, .6, .9]),
            mask_probabilities=torch.tensor([[[1., 0.], [0., 0.]],
                                             [[0., 1.], [0., 0.]],
                                             [[0., 1.], [1., 0.]],
                                             [[1., 1.], [1., 1.]]]),
            image_ids=torch.tensor([0, 0, 0, 1]),
            candidate_ids=torch.tensor([20, 10, 30, 40]),
            source_ids=('rpn', 'uav', 'extra_source', 'rpn'),
            nms_iou_threshold=.5,
        )

    def test_self_and_suppressed_group_exclusion(self):
        context = build_prediction_context(**self.inputs())
        self.assertEqual(context.retained_ids.tolist(), [10, 30, 40])
        self.assertEqual(context.lineage(), {20: 10, 10: 10, 30: 30, 40: 40})
        expected = torch.tensor([[0., 1.], [1., 0.]])
        torch.testing.assert_close(context.neighbors(10), expected)
        torch.testing.assert_close(context.neighbors(20), expected)
        torch.testing.assert_close(context.neighbors(40), torch.zeros(2, 2))
        self.assertEqual(context.retained_source_ids, ('uav', 'extra_source', 'rpn'))

    def test_permutation_and_tie_determinism(self):
        data = self.inputs()
        expected = build_prediction_context(**data)
        for permutation in ([3, 2, 1, 0], [1, 3, 0, 2], [2, 0, 3, 1]):
            changed = dict(data)
            for key in ('boxes', 'scores', 'mask_probabilities', 'image_ids', 'candidate_ids'):
                changed[key] = data[key][permutation]
            changed['source_ids'] = tuple(data['source_ids'][i] for i in permutation)
            result = build_prediction_context(**changed)
            self.assertEqual(result.retained_ids.tolist(), expected.retained_ids.tolist())
            self.assertEqual(result.lineage(), expected.lineage())
            torch.testing.assert_close(result.neighbors(20), expected.neighbors(20))

    def test_two_stage_selection_then_retained_masks_only(self):
        data = self.inputs()
        expected = build_prediction_context(**data)
        probabilities = data.pop('mask_probabilities')
        selection = select_prediction_candidates(**data)
        self.assertEqual(selection.retained_indices.tolist(), [1, 2, 3])
        result = selection.with_masks(probabilities[selection.retained_indices],
                                      selection.retained_ids)
        self.assertEqual(result.lineage(), expected.lineage())
        torch.testing.assert_close(result.retained_masks, expected.retained_masks)
        for candidate_id in data['candidate_ids'].tolist():
            torch.testing.assert_close(result.neighbors(candidate_id), expected.neighbors(candidate_id))
        with self.assertRaises(ValueError):
            selection.with_masks(probabilities, data['candidate_ids'])
        with self.assertRaises(ValueError):
            selection.with_masks(probabilities[selection.retained_indices],
                                 selection.retained_ids.flip(0))

    def test_cap_and_missing_identity(self):
        context = build_prediction_context(**self.inputs(), max_per_image=1)
        self.assertEqual(context.retained_ids.tolist(), [10, 40])
        # Capped target 30 is known but has no contribution in the retained context.
        torch.testing.assert_close(context.neighbors(30), torch.tensor([[0., 1.], [0., 0.]]))
        with self.assertRaises(KeyError):
            context.neighbors(999)

    def test_no_grad_snapshot_and_probability_semantics(self):
        data = self.inputs()
        for key in ('boxes', 'scores', 'mask_probabilities'):
            data[key] = data[key].half().requires_grad_()
        data['mask_probabilities'].data[1, 0, 0] = .5
        result = build_prediction_context(**data, probability_threshold=.5)
        self.assertEqual(result.retained_masks.dtype, torch.bool)
        self.assertFalse(result.neighbors(30).requires_grad)
        self.assertEqual(result.neighbors(30).dtype, torch.float32)
        self.assertEqual(result.neighbors(30).device, data['boxes'].device)
        self.assertEqual(result.neighbors(30)[0, 0].item(), 1.)
        data['mask_probabilities'].data.zero_()
        self.assertEqual(result.neighbors(30)[0, 0].item(), 1.)

    def test_empty(self):
        result = build_prediction_context(
            torch.empty(0, 4), torch.empty(0), torch.empty(0, 2, 3),
            torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64), (),
            nms_iou_threshold=.5)
        self.assertEqual(result.retained_masks.shape, (0, 2, 3))
        with self.assertRaises(KeyError):
            result.neighbors(0)

    def test_sum_not_union_and_threshold_boundaries(self):
        context = build_prediction_context(
            boxes=torch.tensor([[0., 0., 1., 1.], [2., 0., 3., 1.], [4., 0., 5., 1.]]),
            scores=torch.ones(3), mask_probabilities=torch.full((3, 1, 1), .4),
            image_ids=torch.zeros(3, dtype=torch.int64), candidate_ids=torch.tensor([1, 2, 3]),
            source_ids=('rpn', 'uav', 'rpn'), nms_iou_threshold=0.)
        self.assertEqual(context.retained_ids.tolist(), [1, 2, 3])
        self.assertEqual(context.neighbors(1).item(), 2.)
        # Identical boxes at IoU exactly 1 remain distinct when threshold is 1.
        data = self.inputs()
        data['nms_iou_threshold'] = 1.
        self.assertEqual(build_prediction_context(**data).retained_ids.tolist(), [10, 20, 30, 40])

    def test_reject_invalid_inputs(self):
        for key, replacement in (
            ('candidate_ids', torch.tensor([10, 10, 30, 40])),
            ('image_ids', torch.tensor([0., 0., 0., 1.])),
            ('source_ids', ('rpn', '', 'uav', 'rpn')),
            ('mask_probabilities', torch.full((4, 2, 2), -1.)),
            ('scores', torch.tensor([.8, float('nan'), .6, .9])),
        ):
            data = self.inputs()
            data[key] = replacement
            with self.subTest(key=key), self.assertRaises(ValueError):
                build_prediction_context(**data)


if __name__ == '__main__':
    unittest.main()
