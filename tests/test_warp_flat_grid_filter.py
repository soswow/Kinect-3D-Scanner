"""Source and rational containment checks; actual CUDA parity is still required."""
import ast
from fractions import Fraction
import math
from pathlib import Path
import random
import unittest

from scripts.research import flat_grid_filter_bound as bound

ROOT=Path(__file__).resolve().parents[1]
SHADER=ROOT/"scripts/research/research_warp_flat_grid_filtered_nn.cu"
ORIGINAL=ROOT/"scripts/research/microbatch_warp_flat_grid_nn.cu"
DRIVER=ROOT/"scripts/research/benchmark_warp_flat_grid_filtered_nn.py"


def rn32(value,ftz):
    value=Fraction(value)
    if not value:return 0.
    sign=-1 if value<0 else 1;value=abs(value)
    exponent=value.numerator.bit_length()-value.denominator.bit_length()
    if value<Fraction(2)**exponent:exponent-=1
    step=max(exponent-23,-149);scaled=value/(Fraction(2)**step)
    whole,remainder=divmod(scaled.numerator,scaled.denominator)
    if 2*remainder>scaled.denominator or 2*remainder==scaled.denominator and whole%2:whole+=1
    answer=sign*float(whole*(Fraction(2)**step))
    return math.copysign(0.,answer) if ftz and 0<abs(answer)<bound.MU32 else answer


def approximate(point,query,ftz):
    squares=[]
    for p,q in zip(point,query):
        delta=rn32(Fraction(rn32(q,ftz))-Fraction(rn32(p,ftz)),ftz)
        squares.append(rn32(Fraction(delta)**2,ftz))
    return rn32(Fraction(rn32(Fraction(squares[0])+Fraction(squares[1]),ftz))+Fraction(squares[2]),ftz)


def two(values):
    # Original IDs are distinct; coincident XYZ still retain separate IDs.
    return sorted(set(values),key=lambda item:(item[0],item[1]))[:2]


def warp_two(values):
    lanes=[two(values[lane::32]) for lane in range(32)]
    for stride in (16,8,4,2,1):
        for lane in range(stride):lanes[lane]=two(lanes[lane]+lanes[lane+stride])
    return lanes[0]


class WarpFilterTests(unittest.TestCase):
    def test_distinct_two_reduction_with_empty_lanes_duplicates_and_ties(self):
        rng=random.Random(721)
        for count in (0,1,2,31,32,33,127,128,1025):
            values=[(float(rng.randrange(0,9)),i) for i in range(count)]
            rng.shuffle(values)
            self.assertEqual(warp_two(values),two(values))
        self.assertEqual(warp_two([(0.,17),(0.,2),(0.,33),(0.,5)]),[(0.,2),(0.,5)])

    def test_host_cap_bound_encloses_all_actual_lower_query_caps_and_top_two(self):
        rng=random.Random(442)
        for shift in (0,3,5,10,20):
            width=math.ldexp(1.,-shift)
            for query in ([0.,-0.,0.],[4.,-4.,4.],[rng.uniform(-3.9,3.9) for _ in range(3)]):
                cells=[math.floor(x/width) for x in query]
                points=[[(cell+rng.randrange(-1,2)+rng.random())*width for cell in cells] for _ in range(41)]
                maximum=max(abs(x) for point in points for x in point)
                error=bound.error_bound(maximum,4.,shift)
                exact=[bound.original_metric(point,query) for point in points]
                for ftz in (False,True):
                    scores=[approximate(point,query,ftz) for point in points]
                    self.assertTrue(all(abs(Fraction(score)-Fraction(metric))<=Fraction(error) for score,metric in zip(scores,exact)))
                    approx_two=warp_two([(score,i) for i,score in enumerate(scores)])
                    threshold=bound.threshold(approx_two[1][0],error)
                    survivors=[(metric,i) for i,metric in enumerate(exact) if scores[i]<=threshold]
                    self.assertEqual(warp_two(survivors),two([(metric,i) for i,metric in enumerate(exact)]))

    def test_cap_domain_and_declined_screen_do_not_omit_exact_candidates(self):
        for query in ([4.11,0.,0.],[-4.11,0.,0.],[1048575.5,0.,0.]):
            screen=max(map(abs,query))<=4.
            self.assertFalse(screen)
            points=[[query[0]-.01,0.,0.],[query[0]+.01,0.,0.]]
            exact=[(bound.original_metric(point,query),i) for i,point in enumerate(points)]
            self.assertEqual(warp_two(exact),two(exact))
        with self.assertRaises(ValueError):bound.threshold(math.inf,1.)
        with self.assertRaises(ValueError):bound.threshold(1.,math.inf)

    def test_original_double_metric_and_distinct_id_functions_unchanged(self):
        source=SHADER.read_text(encoding="utf-8");original=ORIGINAL.read_text(encoding="utf-8")
        start="__device__ inline bool before(double";end='extern "C" __global__'
        self.assertEqual(source[source.index(start):source.index("__device__ inline bool before32")],original[original.index(start):original.index(end)])
        metric="double dx=__dsub_rn(qx,points[3*index]),dy=__dsub_rn(qy,points[3*index+1]),dz=__dsub_rn(qz,points[3*index+2]);\n        double d=__dadd_rn(__dadd_rn(__dmul_rn(dx,dx),__dmul_rn(dy,dy)),__dmul_rn(dz,dz));"
        self.assertIn(metric,source);self.assertIn(metric,original)
        self.assertIn("if(screen&&(double)approximate(fq_x,fq_y,fq_z,shadow,index)>limit)continue;",source)
        self.assertIn("__dadd_ru((double)second,__dmul_ru(2.,certified_error))",source)
        self.assertIn("q<=query_cap",source);self.assertIn("screen=!bad&&second_id>=0&&isfinite(second)",source)

    def test_warp_phase_boundaries_and_original_output_width(self):
        source=SHADER.read_text(encoding="utf-8")
        self.assertNotIn("__syncthreads",source)
        self.assertIn("if(row>=count) return;",source)
        self.assertIn("__any_sync(mask,invalid!=0)",source)
        self.assertIn("output[5*row+4]=visits",source)
        self.assertIn("diagnostic[3*row]=rechecks",source)
        self.assertIn("((attempted?1:0)+(screen?1:0))*visits",source)

    def test_driver_lazy_import_options_source_pins_and_failure_guard(self):
        source=DRIVER.read_text(encoding="utf-8");tree=ast.parse(source)
        imported=[]
        for node in tree.body:
            if isinstance(node,ast.Import):imported.extend(x.name for x in node.names)
            elif isinstance(node,ast.ImportFrom):imported.append(node.module)
        self.assertFalse(any(name.startswith(("numpy","cupy","open3d")) for name in imported))
        options=next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="OPTIONS" for t in n.targets))
        self.assertEqual(ast.literal_eval(options),("--std=c++11","--fmad=false"))
        for path in ("tests/test_warp_flat_grid_filter.py","scripts/research/microbatch_warp_flat_grid_nn.cu","scripts/research/flat_grid_filter_bound.py"):
            self.assertIn(path,source)
        self.assertIn("value.copy().view(np.uint64),reference.copy().view(np.uint64)",source)
        self.assertIn('label=="outside-host-cap"',source)
        self.assertIn("args.output.exists()",source)
        self.assertIn("if primary is not None:raise primary from error",source)


if __name__=="__main__":unittest.main()
