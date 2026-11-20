import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from experiment import flops,tflops,aggregate,write_csv,NAIVE,TILED,BOUNDARIES,validation_plan
from stats import summarize


class FrameworkTests(unittest.TestCase):
    def test_rectangular_flops_and_units(self):
        self.assertEqual(flops(3,5,7),210)
        self.assertEqual(flops(0,5,7),0)
        self.assertAlmostEqual(tflops(2_000_000_000,2),1)
        with self.assertRaises(ValueError):tflops(10,0)

    def test_inherited_statistics(self):
        s=summarize([1,2,3,4,5])
        self.assertEqual(s['median'],3)
        self.assertAlmostEqual(s['stddev'],2.5**.5)
        self.assertAlmostEqual(s['p95'],4.8)

    def test_validation_coverage(self):
        self.assertEqual(len(validation_plan(NAIVE+TILED)),1200)
        self.assertTrue(any(m!=n and n!=k and m!=k for m,n,k in BOUNDARIES))
        self.assertIn((7,9,0),BOUNDARIES)

    def test_reject_missing_or_failed_data(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)
            write_csv(p/'plan.csv',[dict(id='a',rounds=2,m=3,n=5,k=7)])
            write_csv(p/'correctness.csv',[dict(id='a',correct='PASS')])
            write_csv(p/'resources.csv',[dict(id='a',threads=256)])
            write_csv(p/'samples.csv',[dict(id='a',round=0,kernel_ms=1,e2e_ms=2,pack_ms=0,correct='PASS')])
            with self.assertRaises(ValueError):aggregate(p)
            write_csv(p/'samples.csv',[dict(id='a',round=0,correct='FAIL')])
            with self.assertRaises(ValueError):aggregate(p)


if __name__=='__main__':unittest.main()
