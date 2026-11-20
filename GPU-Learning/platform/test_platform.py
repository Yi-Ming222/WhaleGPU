import sys
import time
import unittest
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from lessons import reduction_steps
from runner import execute
from app import Workbench

class PlatformTests(unittest.TestCase):
    def test_reduction(self):
        self.assertEqual(reduction_steps([1,2,3,4,5]), [[1,2,3,4,5],[3,7,5],[10,5],[15]])
    def test_whitelist(self):
        for action,experiment,count in [('bad','vector-add',1),('run','../bad',1),('benchmark','vector-add',21)]:
            with self.assertRaises(ValueError):execute(action,experiment,count)
    def test_pages(self):
        app=Workbench()
        try:
            for page in ['Learning Center','Kernel Laboratory','GPU Visualization','Benchmark','Performance Analysis']:
                app.show(page); app.update()
        finally:app.destroy()

if __name__=='__main__': unittest.main()
