import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from stats import summarize, percentile, effective_gbps


class StatisticsTests(unittest.TestCase):
    def test_known_distribution(self):
        s = summarize([1., 2., 3., 4., 5.])
        self.assertEqual(s['mean'], 3.)
        self.assertEqual(s['median'], 3.)
        self.assertAlmostEqual(s['stddev'], 2.5 ** .5)
        self.assertAlmostEqual(s['p95'], 4.8)

    def test_tail_is_not_discarded(self):
        s = summarize([1.] * 19 + [101.])
        self.assertEqual(s['mean'], 6.)
        self.assertAlmostEqual(s['p95'], 6.)
        self.assertEqual(s['maximum'], 101.)

    def test_small_and_invalid(self):
        self.assertEqual(summarize([2.])['stddev'], 0)
        self.assertEqual(percentile([10, 20], .5), 15)
        for values in [[], [float('nan')], [-1], [float('inf')]]:
            with self.assertRaises(ValueError):
                summarize(values)

    def test_bandwidth_units(self):
        self.assertAlmostEqual(effective_gbps(249999, 1), 1.)


if __name__ == '__main__':
    unittest.main()
