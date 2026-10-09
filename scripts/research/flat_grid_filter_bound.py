"""Conservative float-filter bound; no numerical imports or timing authority.

For each original 27-cell candidate |p_axis-q_axis| < 2*w, so D=3*w
is a generous enclosure. All bound operations round upward in binary64.
u32=2^-24; mu32=2^-126 covers gradual underflow AND input/output FTZ.
Each float cast differs by <=u*|x|+mu. Subtraction gets four mu to
cover two flushed inputs, rounding and a flushed output. Squaring a
subnormal input has real result <mu^2; four mu is again generous.
Nonnegative addition has at most two flushed inputs and one output.
The double reference bound separately covers its original subtract,
three squares and left-associated two additions. No FMA is allowed.

If |A_i-D_i|<=E uniformly, the exact first/two distinct IDs satisfy
D_i<=A2+E, hence A_i<=A2+2E. Inclusive threshold preserves every tie.
The original exact FP64 metric and lexicographic distinct-ID reduction
are applied to all survivors; fewer than two approximate IDs falls back.
"""
from __future__ import annotations
import math
import struct
from fractions import Fraction

U32=2.**-24
MU32=2.**-126
U64=2.**-53
MU64=2.**-1022
MAX_COORD=1048576.


def up_add(a,b):
    value=a+b
    if not math.isfinite(value):raise ValueError("Nonfinite bound")
    return math.nextafter(value,math.inf) if Fraction(value)<Fraction(a)+Fraction(b) else value


def up_mul(a,b):
    value=a*b
    if not math.isfinite(value):raise ValueError("Nonfinite bound")
    return math.nextafter(value,math.inf) if Fraction(value)<Fraction(a)*Fraction(b) else value


def metric_bound(d,delta,u,mu):
    """Bound approximation error after coordinate discrepancy delta."""
    h=up_add(d,delta)
    h2=up_mul(h,h)
    term=up_add(up_mul(u,h2),up_mul(4.,mu))
    first=up_add(up_mul(up_mul(2.,u),up_add(h2,term)),up_mul(4.,mu))
    second=up_add(up_mul(u,up_add(up_mul(3.,up_add(h2,term)),first)),up_mul(4.,mu))
    cast=up_mul(3.,up_add(up_mul(up_mul(2.,d),delta),up_mul(delta,delta)))
    return up_add(up_add(cast,up_mul(3.,term)),up_add(first,second))


def error_bound(target_max,query_max,shift):
    if (type(shift) is not int or not 0<=shift<=20
            or type(target_max) not in (int,float) or type(query_max) not in (int,float)
            or not math.isfinite(target_max) or not math.isfinite(query_max)
            or not 0<=target_max<=MAX_COORD or not 0<=query_max<=MAX_COORD):
        raise ValueError("Bound requires supported original dyadic grid domain")
    d=3.*math.ldexp(1.,-shift)  # Exact dyadic product.
    pbound=min(float(target_max),up_add(float(query_max),d))
    c=up_add(up_mul(U32,up_add(pbound,float(query_max))),up_mul(2.,MU32))
    delta=up_add(up_add(c,up_mul(U32,up_add(d,c))),up_mul(4.,MU32))
    float_error=metric_bound(d,delta,U32,MU32)
    delta64=up_add(up_mul(U64,d),MU64)
    double_error=metric_bound(d,delta64,U64,MU64)
    return up_add(float_error,double_error)


def threshold(second,error):
    if not math.isfinite(second) or second<0 or not math.isfinite(error) or error<0:
        raise ValueError("Full exact fallback required")
    return up_add(float(second),up_mul(2.,error))


def f32(value,ftz=False):
    """Ordinary host observation only; exact-real rounding oracle is in tests."""
    result=struct.unpack("<f",struct.pack("<f",value))[0]
    return math.copysign(0.,result) if ftz and 0<abs(result)<MU32 else result


def approximate_metric(point,query,ftz=False):
    values=[]
    for p,q in zip(point,query):
        a,b=f32(p,ftz),f32(q,ftz)
        delta=f32(b-a,ftz)
        values.append(f32(delta*delta,ftz))
    return f32(f32(values[0]+values[1],ftz)+values[2],ftz)


def original_metric(point,query):
    dx,dy,dz=(q-p for p,q in zip(point,query))
    return (dx*dx+dy*dy)+dz*dz
