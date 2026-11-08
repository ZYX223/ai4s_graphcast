import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'harness/short8'))
sys.path.insert(0, str(ROOT / 'harness/recovery'))
import prediction
import reproduce as r
import summarize as audit
from resume_metric_prefix import CONTEXT_KEYS, stage_prefix


class ResultTests(unittest.TestCase):
    def test_actual_historical_metric_regression(self):
        expected = r.read(ROOT / 'fixtures/historical-results.json')
        full = audit.full38(r.read(ROOT / 'fixtures/accepted-full38-metrics.json'))
        self.assertAlmostEqual(full['mfu_pct'], expected['mfu_pct'], places=12)
        self.assertAlmostEqual(full['h100_ratio_pct'], expected['mfu_ratio_pct'], places=10)
        short = prediction.evaluate(r.read(ROOT / 'fixtures/accepted-short8-metrics.json'))
        self.assertTrue(short['curve_gate_passed'])
        self.assertAlmostEqual(short['conditional_mfu_pct'], expected['short_prediction_mfu'], places=10)
        self.assertFalse(short['formal_mfu_eligible'])

    def test_partial_and_nonfinite_full_run_rejected(self):
        rows = r.read(ROOT / 'fixtures/accepted-full38-metrics.json')
        for bad in (rows[2:], rows + [rows[-1]], [{**rows[0], 'loss': float('nan')}] + rows[1:]):
            with self.assertRaises(ValueError):
                audit.full38(bad)

    def test_unstable_r1_has_no_conditional_prediction(self):
        rows = r.read(ROOT / 'fixtures/accepted-short8-metrics.json')
        rows[1]['end_to_end_step_seconds'] *= 2
        result = prediction.evaluate(rows)
        self.assertFalse(result['curve_gate_passed'])
        self.assertIsNone(result['conditional_mfu_pct'])

    def test_frozen_full_and_recovery_command_construction(self):
        for mode, entry in [('full38', 'run_full38.py'), ('recovery', 'run_recovery.py')]:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                cfg = r.read(ROOT / 'example-settings.json')
                job = r.prepare(mode, cfg, Path(tmp) / 'job')
                c, template = r.read(job / 'config.json'), r.read(job / 'candidate-template.json')
                text = (job / entry).read_text()
                # Execute the real argument/environment construction, without its
                # preceding I/O or subsequent subprocess: no substitute builder.
                block = text[text.index('oldout = '):text.index('if a.prepare_only:')]
                ns = dict(c=c, template=template, out=job / 'candidate',
                          source=Path(c['arms']['candidate']['source']), model=Path(c['model']),
                          arm=c['arms']['candidate'], before={}, Path=Path)
                exec(compile(block, entry + ':command-only', 'exec'), ns)
                cmd, plan = ns['cmd'], ns['plan']
                self.assertEqual(cmd[cmd.index('--stop-after-global-step') + 1], '38')
                self.assertEqual(cmd[cmd.index('--warmup-updates') + 1], '6')
                self.assertNotIn('--profile-hotspots', cmd)
                if mode == 'recovery':
                    self.assertEqual(plan['requested_updates'], 2)
                    self.assertEqual(plan['actual_update_steps'], [37, 38])
                    self.assertNotIn('--init-checkpoint', cmd)
                    self.assertIn('--resume-training-checkpoint', cmd)
                    self.assertFalse(plan['performance_eligible'])
                else:
                    self.assertEqual(plan['timed_steps'], list(range(1, 39)))

    def test_prefix_requires_real_contiguous_history_and_exact_context(self):
        context = {k: 'same' for k in CONTEXT_KEYS}
        context.update(resume_contract={'schedule_digest': 'sched'},
                       start_optimizer_step=2, optimizer={'lr': .1, 'betas': [.9, .95]})
        before = {**context, 'start_optimizer_step': 0}
        cp = dict(checkpoint='/fixture-only/cp2', checkpoint_sha256='a' * 64,
                  recorded_sha256='a' * 64, hash_matches=True, all_model_finite=True,
                  optimizer_all_finite=True,
                  cursor=dict(global_optimizer_step=2, epoch=0, next_batch_index=2, epoch_plan_digest='plan'),
                  schedule_state=dict(next_update_index=2, digest='sched', last_applied_learning_rate=.1))
        rows = [dict(optimizer_step=i, epoch=0, epoch_batch_index=i-1,
                     epoch_plan_digest='plan', learning_rate=.1) for i in (1, 2)]
        for failure in (None, 'missing', 'duplicate', 'context', 'hash'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                parent, target = Path(tmp) / 'parent', Path(tmp) / 'new'
                (parent / 'metrics').mkdir(parents=True)
                selected = rows[:-1] if failure == 'missing' else rows + [rows[-1]] if failure == 'duplicate' else rows
                metrics = parent / 'metrics/segment_00000001.jsonl'
                metrics.write_text(''.join(json.dumps(x) + '\n' for x in selected))
                r.dump(metrics.with_suffix('.start.json'), before)
                expected = {**context, 'config_sha256': 'wrong'} if failure == 'context' else context
                audit_cp = {**cp, 'hash_matches': False} if failure == 'hash' else cp
                original = metrics.read_bytes()
                if failure:
                    with self.assertRaises(ValueError):
                        stage_prefix(parent, target, audit_cp, expected)
                    self.assertFalse(target.exists())
                else:
                    receipt = stage_prefix(parent, target, audit_cp, expected)
                    self.assertEqual(receipt['training_updates_replayed'], 0)
                    with self.assertRaises(FileExistsError):
                        stage_prefix(parent, target, audit_cp, expected)
                self.assertEqual(metrics.read_bytes(), original)

    def test_validation_requires_all_windows_and_unchanged_model(self):
        from test_journal import fixture, invoke
        ns = fixture()
        result = invoke(ns, ns['namespace']['evaluate_validation'])
        state = dict(model_unchanged=True, gradient_tensors=0)
        audit.validation(result, state, list(range(1, 112)))
        for kind in ('missing', 'mean', 'duplicate', 'nonfinite', 'updates'):
            bad = copy.deepcopy(result)
            if kind == 'missing':
                bad['steps'].pop()
            elif kind == 'mean':
                bad['mean_loss'] += 1
            elif kind == 'duplicate':
                bad['steps'][1]['origins_by_rank'][0]['local_window_indices'] = [1]
            elif kind == 'nonfinite':
                bad['steps'][0]['loss'] = float('nan')
            else:
                bad['optimizer_updates'] = 1
            with self.assertRaises((ValueError, AssertionError)):
                audit.validation(bad, state, list(range(1, 112)))
        with self.assertRaises(ValueError):
            audit.validation(result, {**state, 'model_unchanged': False}, list(range(1, 112)))


if __name__ == '__main__':
    unittest.main()
