"""CPU safety and attribution contracts; never prepares real worktrees or CUDA."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

WORK = Path(__file__).resolve().parents[1]
PATH = WORK / 'scripts/experiments/ue_staged_uav_control.py'
SPEC = importlib.util.spec_from_file_location('staged_control_under_test', PATH)
control = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(control)


class ControlContracts(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        api, self.matrix, self.protocol = control.load_setup(control.REPO)
        self.api = api
        self.manifest = dict(root=str(self.root), runtime_source='/pinned/repo', git_commit='abc',
            recipes=control.checked_recipes(api, self.matrix), stages=self.matrix['stages'],
            base_lock=None, cache_receipts={}, source_sha256={'file.py': 'source'},
            protocol_sha256=control.canonical_sha(self.protocol), matrix_sha256='matrix')

    def tearDown(self):
        self.directory.cleanup()

    def marker_value(self, action, recipe='C'):
        row = dict(world_size=4, recipe=recipe,
            recipe_sha256=self.manifest['recipes'][recipe]['recipe_sha256'],
            source_sha256=self.manifest['source_sha256'],
            protocol_sha256=self.manifest['protocol_sha256'])
        if action == 'gate':
            row.update(passed=True, max_reserved_mib=6000, max_resident_mib=6500,
                       initialization=dict(satellite_sha256='same-satellite', head_sha256='same-head',
                                           rgb_sha256=None if recipe == 'S' else 'same-rgb',
                                           full_model_sha256='full-' + recipe, audit={'recipe': recipe}))
        else:
            row.update(complete=True, epoch=100, updates=26700, best={'epoch': 73, 'score': .8})
            folder = self.root / recipe
            folder.mkdir(exist_ok=True)
            (folder / 'last.pth').write_bytes(b'last-' + recipe.encode())
            (folder / 'best_model.pth').write_bytes(b'best-' + recipe.encode())
            row.update(last_checkpoint_sha256=control.sha(folder / 'last.pth'),
                       best_checkpoint_sha256=control.sha(folder / 'best_model.pth'))
        control.write(control.marker(self.manifest, action, recipe), row)
        return row

    def test_pure_plan_matrix_no_gpu_and_exact_canonical_digest(self):
        self.assertTrue(self.api.audit_matrix(self.matrix)['passed'])
        for recipe in self.manifest['recipes'].values():
            self.assertEqual(control.recipe_digest(recipe), recipe['recipe_sha256'])
        self.assertEqual(control.canonical_sha({'说明': '值'}), self.api.canonical_sha({'说明': '值'}))
        self.assertEqual(control.stage_recipes(self.manifest, 'structure'), ['S', 'U', 'T', 'C'])

    def test_deferred_stages_cannot_infer_base(self):
        for stage in ('modality', 'views'):
            with self.assertRaisesRegex(RuntimeError, 'Explicit validation-based lock'):
                control.stage_recipes(self.manifest, stage)

    def test_structure_training_queue_never_exposes_test_before_lock(self):
        jobs = control.job_plan(self.manifest, 'structure')
        self.assertEqual(len(jobs), 8)
        self.assertEqual({row['action'] for row in jobs.values()}, {'gate', 'train'})
        for name in 'SUTC':
            self.assertEqual(jobs['train_' + name]['dependencies'], ['gate_' + n for n in 'SUTC'])
        with self.assertRaisesRegex(RuntimeError, 'before exposing structure test'):
            control.job_plan(self.manifest, 'structure', evaluation_only=True)
        self.manifest['base_lock'] = {'base_arm': 'U'}
        evaluations = control.job_plan(self.manifest, 'structure', evaluation_only=True)
        self.assertEqual(set(evaluations), {'eval_' + x for x in 'SUTC'})

    def test_followup_queue_tests_only_after_whole_stage_training(self):
        self.manifest['base_lock'] = {'base_arm': 'T'}
        self.manifest['recipes'] = control.checked_recipes(self.api, self.matrix, 'T')
        jobs = control.job_plan(self.manifest, 'modality')
        for name in ('mod_constant', 'mod_no_geometry'):
            self.assertEqual(jobs['eval_' + name]['dependencies'],
                             ['train_mod_constant', 'train_mod_no_geometry'])

    def test_structure_initialization_family_compares_only_shared_states(self):
        for name in 'SUTC':
            self.marker_value('gate', name)
        # Full model hashes/audit differ by design; S has no RGB state at all.
        self.assertTrue(control.verify_initialization_family(self.manifest, 'U')['passed'])
        self.assertTrue(control.verify_initialization_family(self.manifest, 'S')['passed'])
        path = control.marker(self.manifest, 'gate', 'T')
        for key in ('satellite_sha256', 'head_sha256', 'rgb_sha256'):
            row = self.marker_value('gate', 'T')
            row['initialization'][key] = 'wrong'
            control.write(path, row)
            with self.assertRaisesRegex(RuntimeError, 'Initialization family differs'):
                control.verify_initialization_family(self.manifest, 'C')

    def test_own_gate_is_not_enough_to_start_stage_training(self):
        self.marker_value('gate', 'C')
        with self.assertRaisesRegex(RuntimeError, 'Every gate'):
            control.verify_initialization_family(self.manifest, 'C')

    def test_followup_initialization_requires_full_state_equality_to_locked_base(self):
        self.manifest['base_lock'] = {'base_arm': 'U'}
        self.manifest['recipes'] = control.checked_recipes(self.api, self.matrix, 'U')
        base = self.marker_value('gate', 'U')['initialization']
        for name in ('mod_constant', 'mod_no_geometry'):
            row = self.marker_value('gate', name)
            row['initialization'] = dict(base, audit={'different-description': name})
            control.write(control.marker(self.manifest, 'gate', name), row)
        self.assertTrue(control.verify_initialization_family(self.manifest, 'mod_constant')['passed'])
        path = control.marker(self.manifest, 'gate', 'mod_no_geometry')
        row = control.read(path)
        row['initialization']['full_model_sha256'] = 'different-model'
        control.write(path, row)
        with self.assertRaisesRegex(RuntimeError, 'full_model_sha256'):
            control.verify_initialization_family(self.manifest, 'mod_constant')

    def test_satellite_cache_receipt_never_touches_uav_npz(self):
        data = self.root / 'satellite-data'
        for relative in ['dataset.json', 'SHA256SUMS.txt'] + [
                family + '/' + split + '.json'
                for family in ('splits', 'annotations') for split in ('train', 'val', 'test')]:
            path = data / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{}')
        self.manifest['stages'] = {'structure': ['S']}
        self.manifest['legacy_cache_manifest'] = str(self.root / 'DO_NOT_READ_MISSING_UAV_MANIFEST')
        with patch.object(control, 'DATA_ROOT', data):
            control.cache_stage(self.manifest, 'structure')
        receipt = control._verify_cache_receipt(self.manifest, 'S', deep=True)
        self.assertEqual(receipt['kind'], 'pure_satellite')
        self.assertFalse(receipt['sample_uav_cache_dependency'])
        self.assertEqual(len(receipt['files']), 8)
        self.assertFalse(any(path.endswith('.npz') for path in receipt['files']))

    def test_start_prepares_only_requested_stage_then_detaches_queue(self):
        self.manifest['base_lock'] = {'base_arm': 'C'}
        self.manifest['recipes'] = control.checked_recipes(self.api, self.matrix, 'C')
        events = []
        def prepare(root):
            events.append(('prepare', root))
            return self.manifest
        def cache(manifest, stage):
            events.append(('cache', stage))
        def queue(manifest, stage):
            events.append(('queue', stage))
            return {'queued': True, 'stage': stage}
        with patch.object(control, 'prepare', side_effect=prepare), \
             patch.object(control, 'cache_stage', side_effect=cache), \
             patch.object(control, 'queue_stage', side_effect=queue), \
             patch('builtins.print'):
            result = control.start_stage(self.root, 'views')
        self.assertEqual(events, [('prepare', self.root), ('cache', 'views'), ('queue', 'views')])
        self.assertEqual(result, {'queued': True, 'stage': 'views'})

    def test_start_live_owner_never_rewrites_caches_or_spawns_another_owner(self):
        control.write(self.root / 'queue/structure/state.json', {'owner': {'pid': 12, 'start_ticks': '99'}})
        with patch.object(control, 'prepare', return_value=self.manifest), \
             patch.object(control, 'live', return_value=True), \
             patch.object(control, 'cache_stage') as cache, \
             patch.object(control, 'queue_stage') as queue:
            result = control.start_stage(self.root, 'structure')
        self.assertTrue(result['already_queued'])
        cache.assert_not_called()
        queue.assert_not_called()

    def test_start_dead_or_completed_state_requires_explicit_recovery(self):
        control.write(self.root / 'queue/structure/state.json', {'status': 'complete', 'owner': {'pid': 12, 'start_ticks': '99'}})
        with patch.object(control, 'prepare', return_value=self.manifest), \
             patch.object(control, 'live', return_value=False), \
             patch.object(control, 'cache_stage') as cache, \
             patch.object(control, 'queue_stage') as queue:
            with self.assertRaisesRegex(RuntimeError, 'explicit recover'):
                control.start_stage(self.root, 'structure')
        cache.assert_not_called()
        queue.assert_not_called()

    def test_start_later_stage_still_requires_explicit_base_lock(self):
        with patch.object(control, 'prepare', return_value=self.manifest), \
             patch.object(control, 'cache_stage') as cache, \
             patch.object(control, 'queue_stage') as queue:
            with self.assertRaisesRegex(RuntimeError, 'Explicit validation-based lock'):
                control.start_stage(self.root, 'modality')
        cache.assert_not_called()
        queue.assert_not_called()

    def test_gate_rejects_wrong_recipe_source_or_protocol(self):
        self.marker_value('gate')
        self.assertTrue(control.accepted(self.manifest, 'gate', 'C'))
        path = control.marker(self.manifest, 'gate', 'C')
        for key, bad in [('recipe', 'U'), ('recipe_sha256', 'wrong'), ('source_sha256', {}),
                         ('protocol_sha256', 'wrong'), ('world_size', 1), ('max_reserved_mib', 0)]:
            row = self.marker_value('gate')
            row[key] = bad
            control.write(path, row)
            self.assertFalse(control.accepted(self.manifest, 'gate', 'C'), key)

    def test_ten_epoch_completion_cannot_satisfy_full_budget(self):
        row = self.marker_value('train')
        self.assertTrue(control.accepted(self.manifest, 'train', 'C'))
        row.update(epoch=10, updates=2670)
        control.write(control.marker(self.manifest, 'train', 'C'), row)
        self.assertFalse(control.accepted(self.manifest, 'train', 'C'))

    def test_checkpoint_corruption_rejects_training_marker(self):
        self.marker_value('train')
        (self.root / 'C/best_model.pth').write_bytes(b'other-task-checkpoint')
        self.assertFalse(control.accepted(self.manifest, 'train', 'C'))

    def test_explicit_lock_uses_validation_receipts_never_test(self):
        for name in ('S', 'U', 'T', 'C'):
            self.marker_value('train', name)
        reads = []
        original = control.read
        def observed(path):
            reads.append(str(path))
            return original(path)
        with patch.object(control, 'load_setup', return_value=(self.api, self.matrix, self.protocol)):
            with patch.object(control, 'read', side_effect=observed):
                control.lock_base(self.manifest, 'U', 'validation and measured memory')
        self.assertFalse(any('/test/' in value for value in reads))
        self.assertEqual(self.manifest['base_lock']['base_arm'], 'U')
        self.assertEqual(control.stage_recipes(self.manifest, 'views'), ['views_1', 'views_3'])
        self.assertEqual(self.manifest['recipes']['mod_constant']['arm'], 'U')
        with self.assertRaisesRegex(RuntimeError, 'already locked'):
            control.lock_base(self.manifest, 'C', 'changed mind')

    def test_lock_rejects_incomplete_structure(self):
        self.marker_value('train', 'C')
        with self.assertRaisesRegex(RuntimeError, 'All structure trainings'):
            control.lock_base(self.manifest, 'C', 'validation')

    def test_all_commands_are_four_gpu_full_runner_and_recipe_isolated(self):
        argv = control.command(self.manifest, 'train', 'C', resume=True)
        self.assertIn('--nproc_per_node=4', argv)
        self.assertNotIn('--epochs', argv)
        self.assertEqual(argv[-2:], ['--resume', str(self.root / 'C/last.pth')])
        self.assertIn('/pinned/repo/portable_sam2_explicit_coarse/scripts/probes/ue_staged_uav_run_20261028.py', argv)
        self.assertEqual(control.environment()['CUDA_VISIBLE_DEVICES'], '0,1,2,3')

    def test_analysis_is_explicit_cpu_command_in_pinned_worktree(self):
        argv = control.analysis_command(self.manifest, 'structure')
        self.assertEqual(argv[2], '/pinned/repo/portable_sam2_explicit_coarse/scripts/analysis/ue_staged_uav_analysis_20261028.py')
        self.assertEqual(argv[-4:], ['--root', str(self.root), '--stage', 'structure'])
        self.assertNotIn('torch.distributed.run', argv)
        self.assertNotIn('--nproc_per_node=4', argv)
        self.assertFalse(any(row['action'] == 'analyze' for row in control.job_plan(self.manifest, 'structure').values()))

    def test_analysis_requires_stage_and_locked_base_completed_evaluation(self):
        self.manifest['base_lock'] = {'base_arm': 'U'}
        self.manifest['recipes'] = control.checked_recipes(self.api, self.matrix, 'U')
        names = ['mod_constant', 'mod_no_geometry', 'U']
        for name in names:
            training = self.marker_value('train', name)
            evaluation = dict(world_size=4, recipe=name, complete=True,
                recipe_sha256=self.manifest['recipes'][name]['recipe_sha256'],
                source_sha256=self.manifest['source_sha256'], protocol_sha256=self.manifest['protocol_sha256'],
                split='test', images=459, precision='FP32', checkpoint_sha256=training['best_checkpoint_sha256'])
            control.write(control.marker(self.manifest, 'eval', name), evaluation)
        self.assertEqual(control.require_analysis_inputs(self.manifest, 'modality'), names)
        control.marker(self.manifest, 'eval', 'U').unlink()
        with self.assertRaisesRegex(RuntimeError, 'required for analysis: U'):
            control.require_analysis_inputs(self.manifest, 'modality')

    def test_all_card_startup_reservation_before_rank_receipt(self):
        running = dict(recipe='C', rank_launch_record=str(self.root / 'not_yet_written.json'),
                       launcher={'start_ticks': '20'})
        replies = ['\n'.join('%d, GPU-%d, 24000' % (g, g) for g in range(4)), '']
        with patch.object(control.subprocess, 'check_output', side_effect=replies):
            cards = control.resources(self.manifest, [running])
        self.assertEqual(set(cards), {0, 1, 2, 3})
        for card in cards.values():
            self.assertEqual(card['admissible_free_mib'], 24000 - 1024 - 15872)

    def test_cache_receipt_detects_same_size_silent_content_change(self):
        data = self.root / 'cache.npz'
        data.write_bytes(b'first')
        receipt = dict(recipe_sha256=self.manifest['recipes']['C']['recipe_sha256'],
            protocol_sha256=self.manifest['protocol_sha256'], source_sha256=self.manifest['source_sha256'],
            files={str(data): dict(sha256=control.sha(data), stat=control._stamp(data))})
        path = self.root / 'receipt.json'
        control.write(path, receipt)
        self.manifest['cache_receipts']['C'] = dict(path=str(path), sha256=control.sha(path))
        control._verify_cache_receipt(self.manifest, 'C', deep=True)
        stat = data.stat()
        data.write_bytes(b'other')
        os.utime(data, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        with self.assertRaisesRegex(RuntimeError, 'Cache bytes'):
            control._verify_cache_receipt(self.manifest, 'C', deep=True)


if __name__ == '__main__':
    unittest.main()
