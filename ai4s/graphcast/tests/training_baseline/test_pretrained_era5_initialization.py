from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/training'))
sys.path.insert(0, str(ROOT / 'src'))
from run_official_epoch_training import parse_args


def command():
    return ['--config','config.yaml','--stats-dir','stats','--corpus-manifest','corpus.json',
            '--dataset-kind','era5','--training-stage','era5_pretrain',
            '--tensor-core','enabled','--peak-tflops-per-device','0',
            '--peak-tflops-source','unknown','--init-mode','random',
            '--run-dir','output','--save-checkpoint','output/final.pt',
            '--schedule-recipe','official_2_to_12','--rollout-steps','12',
            '--rollout-checkpoint-segments','12','--activation-checkpoint',
            '--data-parallel-replicas','1','--mp-devices-per-replica','4',
            '--warmup-updates','6','--training-updates','10','--updates','22','--formal']


def pretrained():
    c=command();c[c.index('--init-mode')+1]='checkpoint'
    return c+['--init-checkpoint','official_model.pt','--allow-pretrained-era5-init']


class PretrainedInitializationTests(unittest.TestCase):
    def test_existing_random_contract(self):
        self.assertEqual(parse_args(command()).init_mode, 'random')

    def test_checkpoint_requires_explicit_opt_in(self):
        with self.assertRaisesRegex(ValueError,'fixed seed'):
            parse_args(pretrained()[:-1])

    def test_explicit_pretrained_keeps_formal_schedule_and_topology(self):
        for dp in (1,2):
            for pp in (2,4):
                c=pretrained()+['--data-parallel-replicas',str(dp),
                                '--mp-devices-per-replica',str(pp),
                                '--batch-size',str(dp),'--micro-batch-size',str(dp)]
                a=parse_args(c)
                self.assertTrue(a.formal)
                self.assertEqual((a.warmup_updates,a.training_updates,a.finetune_updates),(6,10,22))
                self.assertEqual((a.data_parallel_replicas,a.mp_devices_per_replica),(dp,pp))

    def test_pretrained_flag_cannot_change_exact_resume(self):
        with self.assertRaisesRegex(ValueError,'new ERA5'):
            parse_args(pretrained()+['--resume-training-checkpoint','resume.pt'])

    def test_full_model_and_batch_checks_remain_active(self):
        for extra in (['--lat-count','10'],['--mesh-size','3'],
                      ['--rollout-steps','6'],['--batch-size','3','--micro-batch-size','2']):
            with self.subTest(extra=extra),self.assertRaises(ValueError):
                parse_args(pretrained()+extra)

    def test_random_cannot_claim_pretrained_initialization(self):
        with self.assertRaisesRegex(ValueError,'new ERA5'):
            parse_args(command()+['--allow-pretrained-era5-init'])

if __name__=='__main__':
    unittest.main()
