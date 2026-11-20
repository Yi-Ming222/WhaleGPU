import unittest
from reduction_model import trace,two_stage,OPS
from app import Workbench
class ReductionTests(unittest.TestCase):
    def test_rounds_and_reference(self):
        for op,reference in [('SUM',sum),('MAX',max),('MIN',min)]:
            for n in [1,3,8,17,32,33,64,95,256]:
                data=[(i*7)%23-12 for i in range(n)]
                for mode in ['shared','block']+(['shuffle'] if n<=32 else []):
                    states=trace(data,mode,op)
                    self.assertEqual(states[-1].after[0],reference(data),(op,n,mode))
                    for s in states:
                        for i,j in s.edges:
                            expected=reference([s.before[i],s.before[j]])
                            self.assertEqual(s.after[i],expected)
                        for i in set(range(len(s.after)))-set(dict(s.edges)):
                            self.assertEqual(s.after[i],s.before[i])
    def test_examples(self):
        s=trace(list(range(1,9)))
        self.assertEqual([x.offset for x in s if x.edges],[4,2,1])
        self.assertEqual([x.after[:x.offset] for x in s if x.edges],[(6,8,10,12),(16,20),(36,)])
        s=trace(list(range(1,33)),'shuffle')
        for step in s[1:]:
            # Independent subset reference: contributions at this stage have spacing offset.
            for lane in range(32):self.assertEqual(step.after[lane],sum(range(lane+1,33,step.offset)))
        self.assertEqual(s[-1].after[0],528)
    def test_two_stage(self):
        for op,ref in [('SUM',sum),('MAX',max),('MIN',min)]:
            for size in [8,33,256]:
                data=list(range(-103,410));s=two_stage(data,op,size)
                self.assertEqual(s[-1].after[0],ref(data))
                self.assertTrue(any('Kernel 1 完成' in x.title for x in s))
    def test_controls(self):
        app=Workbench();app.show('GPU Visualization');v=app.reduction_view
        try:
            v.move(1);self.assertEqual(v.index,1);v.move(-1);self.assertEqual(v.index,0)
            v.play();timer=v.timer;v.play();self.assertEqual(v.timer,timer)
            v.pause();self.assertIsNone(v.timer);v.move(3);v.reset();self.assertEqual(v.index,0)
            v.play();app.show('Kernel Laboratory');self.assertIsNone(v.timer)
        finally:app.destroy()
if __name__=='__main__':unittest.main()
