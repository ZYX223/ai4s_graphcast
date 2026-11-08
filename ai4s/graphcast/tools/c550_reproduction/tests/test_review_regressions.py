import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import reproduce as r
import summarize as a


class ReviewRegressions(unittest.TestCase):
    def test_stop_cleans_child_that_created_another_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / 'child'
            script = "import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],start_new_session=True);open(sys.argv[1],'w').write(str(p.pid));time.sleep(60)"
            parent = subprocess.Popen([sys.executable, '-c', script, str(marker)], start_new_session=True)
            child = None
            try:
                deadline = time.monotonic() + 5
                while not marker.exists() and time.monotonic() < deadline:
                    time.sleep(.02)
                child = int(marker.read_text())
                r.stop_group(parent)
                state = subprocess.run(['ps', '-o', 'stat=', '-p', str(child)], capture_output=True, text=True).stdout.strip()
                self.assertTrue(not state or state.startswith('Z'), 'escaped owned child still running: ' + state)
            finally:
                if parent.poll() is None:
                    parent.kill()
                parent.wait()
                if child is not None:
                    try:
                        os.kill(child, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    def test_recovery_self_comparison_cannot_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = r.read(ROOT / 'example-settings.json')
            job = r.prepare('recovery', settings, Path(tmp) / 'job')
            out = job / 'candidate'
            (out / 'run').mkdir(parents=True)
            cfg = r.read(job / 'config.json')
            metrics = r.read(ROOT / 'fixtures/accepted-full38-metrics.json')
            for row in metrics:
                row['grad_norm_before_clip'] = 1.
            r.dump(out / 'run/summary.json', dict(metrics=metrics, checkpoint_sha256='a' * 64))
            r.dump(out / 'launch.json', dict(requested_updates=2, actual_update_steps=[37, 38]))
            (out / 'training.log').write_text('official_graphcast_train update=37/38\nofficial_graphcast_train update=38/38\n')
            r.dump(out / 'resume-start-guard.json', dict(actual={'x':'x'}, expected={'x':'x'}, scheduled_lr=3e-7))
            rng = cfg['resume_configuration']['expected_resume_rng_sha']
            r.dump(out / 'resume-rng-guard.json', dict(exact=True, actual=rng, expected=rng))
            r.dump(out / 'resume-prefix-hook-receipt.json', dict(actual_context_checked=True, training_updates_replayed=0, resume_step=36))
            comparison = dict(continuous_sha256='a' * 64, resumed_sha256='a' * 64,
                              hashes_match_recorded=True, cuda_initialized=False)
            comparison.update({k: True for k in ['cursor_equal','schedule_equal','rng_equal','resume_contract_equal','initial_lineage_equal']})
            comparison.update({k:dict(all_finite=True, relative_l2=0.) for k in ['model','optimizer']})
            with self.assertRaises(ValueError):
                a.recovery(job, out, out / 'run', comparison)


if __name__ == '__main__':
    unittest.main()
