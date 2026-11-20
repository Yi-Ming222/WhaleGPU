import csv
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run import aggregate, write_csv


class RecordValidationTests(unittest.TestCase):
    def fixture(self, directory, samples):
        write_csv(directory / 'plan.csv', [dict(id='a',n=4,rounds=2)])
        write_csv(directory / 'resources.csv', [dict(id='a',passes=1)])
        write_csv(directory / 'samples.csv', samples)

    def test_success_preserves_slow_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            self.fixture(path,[dict(id='a',round=0,kernel_ms=1,e2e_ms=2,correct='PASS'),
                               dict(id='a',round=1,kernel_ms=9,e2e_ms=10,correct='PASS')])
            result=aggregate(path)[0]
            self.assertEqual(result['kernel_mean'],5)
            self.assertEqual(result['kernel_maximum'],9)

    def test_reject_failed_missing_duplicate_or_unknown(self):
        valid=dict(id='a',round=0,kernel_ms=1,e2e_ms=2,correct='PASS')
        invalid_sets=[[valid], [valid,valid], [valid,dict(valid,round=1,correct='FAIL')],
                      [valid,dict(valid,round=1,id='unknown')]]
        for samples in invalid_sets:
            with self.subTest(samples=samples), tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp); self.fixture(path,samples)
                with self.assertRaises(ValueError): aggregate(path)


if __name__=='__main__':
    unittest.main()
