"""Meaningful stdlib bound and top-two containment tests, no CUDA imports."""
from fractions import Fraction
import math
import random
import struct
import unittest
from scripts.research import flat_grid_filter_bound as b


def exact_rn32(value,ftz=False):
    """Exact binary-rational nearest-even oracle, including halfway cases."""
    value=Fraction(value)
    if value==0:return 0.
    sign=-1 if value<0 else 1;value=abs(value)
    exponent=value.numerator.bit_length()-value.denominator.bit_length()
    power=Fraction(2)**exponent
    if value<power:exponent-=1
    step=max(exponent-23,-149)
    scaled=value/(Fraction(2)**step)
    whole,remainder=divmod(scaled.numerator,scaled.denominator)
    if 2*remainder>scaled.denominator or 2*remainder==scaled.denominator and whole%2:whole+=1
    answer=sign*float(whole*(Fraction(2)**step))
    return math.copysign(0.,answer) if ftz and 0<abs(answer)<b.MU32 else answer


def exact_approximate(point,query,ftz=False):
    values=[]
    for p,q in zip(point,query):
        a,c=exact_rn32(Fraction(p),ftz),exact_rn32(Fraction(q),ftz)
        delta=exact_rn32(Fraction(c)-Fraction(a),ftz)
        values.append(exact_rn32(Fraction(delta)**2,ftz))
    first=exact_rn32(Fraction(values[0])+Fraction(values[1]),ftz)
    return exact_rn32(Fraction(first)+Fraction(values[2]),ftz)


class BoundTests(unittest.TestCase):
    def test_two_error_terms_are_necessary_independent_of_float_sample(self):
        exact=[1.,1.01,1.02,1.03];approx=[2.,2.01,.02,.03];error=1.
        second=sorted(approx)[1]
        wrong=[i for i,a in enumerate(approx) if a<=second+error]
        right=[i for i,a in enumerate(approx) if a<=b.threshold(second,error)]
        self.assertNotEqual(sorted(wrong,key=lambda i:(exact[i],i))[:2],[0,1])
        self.assertEqual(sorted(right,key=lambda i:(exact[i],i))[:2],[0,1])

    def test_exact_threshold_tied_second_is_not_strictly_pruned(self):
        exact=[0.,1.,1.];approx=[0.,2.,0.];limit=2.
        strict=[i for i,a in enumerate(approx) if a<limit]
        inclusive=[i for i,a in enumerate(approx) if a<=limit]
        self.assertEqual(sorted(inclusive,key=lambda i:(exact[i],i))[:2],[0,1])
        self.assertNotEqual(sorted(strict,key=lambda i:(exact[i],i))[:2],[0,1])

    def check_case(self,points,query,shift):
        maximum=max(abs(x) for p in points for x in p)
        qmax=max(map(abs,query))
        error=b.error_bound(maximum,qmax,shift)
        exact=[b.original_metric(p,query) for p in points]
        real=[sum((Fraction(q)-Fraction(p))**2 for p,q in zip(point,query)) for point in points]
        for ftz in (False,True):
            approx=[b.approximate_metric(p,query,ftz) for p in points]
            for a,d,r in zip(approx,exact,real):
                self.assertLessEqual(abs(Fraction(a)-Fraction(d)),Fraction(error))
                # The real metric is separately observed, not substituted for RN64.
                self.assertTrue(r>=0)
            ranked=sorted(range(len(points)),key=lambda i:(approx[i],i))
            limit=b.threshold(approx[ranked[1]],error)
            kept=[i for i in range(len(points)) if approx[i]<=limit]
            exact_two=sorted(range(len(points)),key=lambda i:(exact[i],i))[:2]
            filtered_two=sorted(kept,key=lambda i:(exact[i],i))[:2]
            self.assertEqual(exact_two,filtered_two)

    def test_boundary_duplicates_signedzero_subnormal_and_underflow(self):
        tiny=math.ldexp(1.,-1074)
        cases=[([[0.,0.,0.],[-0.,0.,0.],[tiny,0.,0.],[-tiny,0.,0.]],[0.,0.,0.],20),
            ([[b.MU32,0.,0.],[math.nextafter(b.MU32,0.),0.,0.],[2*b.MU32,0.,0.]],[0.,0.,0.],20),
            ([[1.,0.,0.],[math.nextafter(1.,math.inf),0.,0.],[math.nextafter(1.,0.),0.,0.]],[1.,0.,0.],0),
            ([[-1048576.,0.,0.],[-1048575.9999999,0.,0.],[-1048575.,0.,0.]],[-1048575.5,0.,0.],0),
            ([[.0625,0.,0.],[.0625,.0625,0.],[-.0625,0.,0.]],[math.nextafter(.0625,0.),0.,0.],4)]
        for points,query,shift in cases:self.check_case(points,query,shift)

    def test_random_original_neighbour_cells_not_arbitrary_radius_samples(self):
        rng=random.Random(93217)
        for shift in (0,3,5,10,20):
            width=math.ldexp(1.,-shift)
            for _ in range(30):
                cell=[rng.randrange(-1048575,1048575) for _ in range(3)]
                query=[(c+rng.random())*width for c in cell]
                points=[[(c+rng.randrange(-1,2)+rng.random())*width for c in cell] for _ in range(24)]
                self.check_case(points,query,shift)

    def test_exact_real_reference_rounding_is_included(self):
        query=[.75,.5,.25]
        points=[[.75+2.**-24,.5+2.**-26,.25+2.**-28],[.75+2.**-25,.5,.25]]
        self.check_case(points,query,0)

    def test_exact_rational_halfway_even_subnormal_and_ftz_oracle(self):
        self.assertEqual(exact_rn32(Fraction(1)+Fraction(2)**-24),1.)
        self.assertEqual(exact_rn32(Fraction(1)+3*Fraction(2)**-24),1.+2.**-22)
        self.assertEqual(exact_rn32(Fraction(2)**-150),0.)
        self.assertEqual(exact_rn32(3*Fraction(2)**-150),2.**-148)
        self.assertEqual(exact_rn32(Fraction(2)**-149,True),0.)
        points=[[1.+2.**-24,2.**-126,0.],[-1.-3*2.**-24,2.**-149,0.],[1.,0.,0.]]
        query=[1.,math.nextafter(2.**-126,0.),2.**-1074]
        error=b.error_bound(max(abs(x) for p in points for x in p),max(map(abs,query)),0)
        for ftz in (False,True):
            for p in points:self.assertLessEqual(abs(Fraction(exact_approximate(p,query,ftz))-Fraction(b.original_metric(p,query))),Fraction(error))

    def test_upward_arithmetic_never_undershoots_exact_rationals(self):
        for a,c in ((1.,2.**-53),(2.**-1022,2.**-1074),(1048576.,.1),(2.**-126,2.**-24)):
            self.assertGreaterEqual(Fraction(b.up_add(a,c)),Fraction(a)+Fraction(c))
            self.assertGreaterEqual(Fraction(b.up_mul(a,c)),Fraction(a)*Fraction(c))

    def test_inclusive_threshold_retains_distance_and_id_ties(self):
        self.check_case([[.01,0.,0.],[-.01,0.,0.],[.01,0.,0.],[.02,0.,0.]],[0.,0.,0.],5)

    def test_invalid_domain_scores_and_extreme_bounds_require_exact_fallback(self):
        for m,q,s in ((math.inf,0.,0),(0.,math.nan,0),(-1.,0.,0),(0.,0.,-1),(0.,0.,21),(1048577.,0.,0)):
            with self.assertRaises(ValueError):b.error_bound(m,q,s)
        for score in (math.inf,math.nan,-1.):
            with self.assertRaises(ValueError):b.threshold(score,1.)

    def test_f32_nearest_two_is_not_substituted_for_exact_two(self):
        query=[0.,0.,0.]
        points=[[1.+2.**-25,0.,0.],[1.,0.,0.],[1.-2.**-26,0.,0.]]
        self.assertNotEqual(sorted(range(3),key=lambda i:(b.approximate_metric(points[i],query),i))[:2],
            sorted(range(3),key=lambda i:(b.original_metric(points[i],query),i))[:2])
        self.check_case(points,query,0)


if __name__=="__main__":unittest.main()
