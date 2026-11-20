"""Run one level at a time. The reference uses ordinary CPU operations."""
import sys
import unittest

import puzzles as p


class Map(unittest.TestCase):
    def test_map(self):
        self.assertEqual(p.map_double([-3, 0, 4]), [-6, 0, 8])
        self.assertEqual(p.map_double([]), [])


class Zip(unittest.TestCase):
    def test_zip(self):
        self.assertEqual(p.zip_add([1, 5], [9, -2]), [10, 3])
        self.assertEqual(p.zip_add([], []), [])
        with self.assertRaises(ValueError):
            p.zip_add([1], [])


class Stride(unittest.TestCase):
    def test_stride(self):
        self.assertEqual(p.grid_stride_indices(13, 2, 3),
                         [[0, 6, 12], [1, 7], [2, 8], [3, 9], [4, 10], [5, 11]])
        self.assertEqual(p.grid_stride_indices(1, 2, 2), [[0], [], [], []])
        self.assertEqual(p.grid_stride_indices(0, 1, 2), [[], []])


class Shared(unittest.TestCase):
    def test_shared(self):
        self.assertEqual(p.shared_neighbor_sums([1, 2, 3, 4, 5, 6], 4),
                         [3, 5, 7, 4, 11, 6])
        self.assertEqual(p.shared_neighbor_sums([], 4), [])


class Reduction(unittest.TestCase):
    def test_reduction(self):
        blocks, threads = 4, 64
        for n in [0, 3, 255, 256, 257, 1024, 1031]:
            with self.subTest(n=n):
                xs = list(range(1, n + 1))
                # Each group of 256 input elements assigns 64 to each block.
                expected = [
                    sum(sum(xs[base + b * threads:base + (b + 1) * threads])
                        for base in range(0, n, blocks * threads))
                    for b in range(blocks)
                ]
                actual = p.block_sums(xs, blocks, threads)
                self.assertEqual(actual, expected)
                self.assertEqual(sum(actual), sum(xs))

    def test_one_lane(self):
        self.assertEqual(p.block_sums([1, 2, 3], 2, 1), [4, 2])


if __name__ == "__main__":
    unittest.main()
