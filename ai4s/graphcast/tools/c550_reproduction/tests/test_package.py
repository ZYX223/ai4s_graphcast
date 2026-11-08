import importlib.util
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import reproduce as r


class PackagingTests(unittest.TestCase):
    def test_render_keys_and_values(self):
        self.assertEqual(r.render({'${ROOT}/a': ['${ROOT}/b']}, {'ROOT': '/tmp/x'}),
                         {'/tmp/x/a': ['/tmp/x/b']})
        with self.assertRaises(ValueError):
            r.render('${MISSING}', {})

    def test_accepted_model_identity(self):
        files = json.loads((ROOT / 'locks/model-files.json').read_text())
        self.assertEqual(len(files), 144)
        for path, digest in files.items():
            self.assertEqual(r.sha(ROOT.parents[1] / path), digest, path)

    def test_all_modes_prepare_without_gpu_or_external_assets(self):
        for mode in ('full38', 'short8', 'recovery', 'evaluation'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                cfg = json.loads((ROOT / 'example-settings.json').read_text())
                cfg['model_root'] = str(ROOT.parents[1])
                job = Path(tmp) / mode
                r.prepare(mode, cfg, job)
                self.assertTrue((job / 'package-manifest.json').is_file())
                r.verify_job(job)
                with self.assertRaises(FileExistsError):
                    r.prepare(mode, cfg, job)
                (job / 'config.json').write_text('{}')
                with self.assertRaises(ValueError):
                    r.verify_job(job)
        self.assertNotIn('torch', sys.modules)

    def test_fresh_gate_binds_job_device_and_zero_processes(self):
        gate = dict(allowed=True, platform='c550', owner_user_id='test-only',
                    healthy=True, process_count=0, utilization_percent=0,
                    memory_used_mib=0, epoch=100, physical_gpu=7,
                    pci_bus='0000:da:00.0', device_uuid='test-only',
                    container_mapping_verified=True, environment_verified=True,
                    raw_evidence_sha256='a' * 64, job_digest='b' * 64)
        device = dict(physical_gpu=7, pci_bus='0000:da:00.0')
        r.validate_gate(gate, device, 'b' * 64, now=101)
        for changes in ({'process_count': 1}, {'epoch': -100},
                        {'pci_bus': '0000:01:00.0'}, {'healthy': False},
                        {'job_digest': 'c' * 64}, {'utilization_percent': 2}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                r.validate_gate({**gate, **changes}, device, 'b' * 64, now=101)

    def test_recovery_relocation_rejected_before_gpu(self):
        cfg = json.loads((ROOT / 'example-settings.json').read_text())
        with self.assertRaisesRegex(ValueError, 'path contract differs'):
            r.validate_resume_paths({'required_resume_paths': {'config': '/old/config'}}, cfg)

    def test_bootstrap_reaches_launcher_and_worker_without_site(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            site = base / 'installed'
            site.mkdir()
            (site / 'sentinel.py').write_text('value=123\n')
            for mode in ('full38', 'short8', 'recovery'):
                cfg = r.read(ROOT / 'example-settings.json')
                cfg.update(python=sys.executable, model_root=str(ROOT.parents[1]),
                           flaggems_root=str(ROOT.parents[1]), dependency_paths=[],
                           python_site_paths=[str(site)])
                job = r.prepare(mode, cfg, base / mode)
                plan = dict(command=[sys.executable, '-m', 'torch.distributed.run',
                                     '--standalone', '--nproc-per-node', '1'],
                            output_dir=str(job / 'candidate'))
                prefix, worker = r.training_command(plan, job)
                self.assertEqual(prefix[1:3], ['-S', str(job / 'bootstrap.py')])
                self.assertEqual(worker[:3], ['--no-python', sys.executable, '-S'])
                # A standalone CPU script verifies actual -S bootstrap resolution.
                script = job / 'probe.py'
                script.write_text('import sys,sentinel; assert sys.flags.no_site; assert sentinel.value==123\n')
                result = subprocess.run(prefix[:3] + [str(script)], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                (site / 'module_probe.py').write_text(script.read_text())
                result = subprocess.run(prefix[:3] + ['--module', 'module_probe'], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                source = (job / r.ENTRIES[mode]).read_text()
                self.assertIn("'--foreground'", source)

    def test_published_harness_provenance_matches_files(self):
        for path, record in r.read(ROOT / 'provenance.json')['files'].items():
            self.assertEqual(r.sha(ROOT / path), record['packaged_sha256'], path)
            if record['change'] == 'byte-identical':
                self.assertEqual(record['original_sha256'], record['packaged_sha256'], path)

    def test_training_data_hash_receipt_reuse_and_drift_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = {'train_data_root': str(root / 'data')}
            data = root / 'data/official_era5/era5_2017-01-01_07_graphcast_operational_pretrain.nc'
            original_stat = Path.stat
            stamp = [123]
            def stat(path, *args, **kwargs):
                return SimpleNamespace(st_size=5896612497, st_mtime_ns=stamp[0]) if path == data else original_stat(path, *args, **kwargs)
            with mock.patch.object(Path, 'stat', stat), mock.patch.object(r, 'sha', return_value='f277accdd54a8d652b2357035fde107fc80f7cdcfceb71d0ba155ed92b3c02a3') as hashing:
                r.verify_training_data(root, settings)
                r.verify_training_data(root, settings)
                self.assertEqual(hashing.call_count, 1)
                stamp[0] += 1
                with self.assertRaisesRegex(ValueError, 'changed after verification'):
                    r.verify_training_data(root, settings)


if __name__ == '__main__':
    unittest.main()
