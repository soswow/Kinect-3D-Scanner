"""Stdlib interval/domain/source contracts; no NumPy/SciPy/native execution."""

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.research.native_corner_cpu import configuration,outward_x_interval,MAX_COORDINATE,MAX_SQUARED


def run():
    rows=[]
    def reject(name, action):
        try:
            action()
        except (ValueError,ArithmeticError):
            pass
        else:
            raise AssertionError("Invalid contract accepted: "+name)
        rows.append({"case":name,"passed":True})
    for q,d in ((True,0.),(0.,True),(float("nan"),0.),(0.,float("inf")),
            (2049.,1.),(0.,-1.),(0.,MAX_SQUARED+1)):
        reject("invalid interval domain %r,%r"%(q,d),lambda q=q,d=d:outward_x_interval(q,d))
    for cfg in ((True,65536,True),(21,65536,True),(1,0,True),(1,307201,True),(1,65536,1)):
        reject("invalid resource policy %r"%(cfg,),lambda cfg=cfg:configuration(*cfg))
    assert configuration(1,65536,True)==(1,65536,True)
    rows.append({"case":"bounded resource policy","passed":True})
    tiny=math.nextafter(0.,math.inf)
    coordinates=[0.,-0.,tiny,-tiny,2.**-1000,-2.**-1000,2.**-540,-2.**-540,
        2.**-500,-2.**-500,2.**-20,-2.**-20,.5,math.nextafter(.5,math.inf),
        math.nextafter(.5,-math.inf),1.,-1.,2048.,-2048.]
    checked=0
    # Every bounded x with original RN(dx*dx)<=candidate d2 must be enclosed,
    # including underflow ties and subtraction at opposite signed extremes.
    for q in coordinates:
        for seed in coordinates:
            delta=seed-q; squared=delta*delta
            if squared>MAX_SQUARED:
                continue
            low,high=outward_x_interval(q,squared)
            for target in coordinates:
                dx=target-q
                if dx*dx<=squared:
                    assert low<=target<=high,(q,seed,target,squared,low,high)
                    checked+=1
    rng=random.Random(314159)
    for _ in range(500):
        query=(rng.uniform(-2048,2048),rng.uniform(-2048,2048))
        points=[(rng.uniform(-2048,2048),rng.uniform(-2048,2048)) for _ in range(20)]
        points += [query,(math.nextafter(query[0],math.inf),query[1]),
            (math.nextafter(query[0],-math.inf),query[1])]
        seed=rng.choice(points)
        dx,dy=seed[0]-query[0],seed[1]-query[1]
        squared=dx*dx+dy*dy
        low,high=outward_x_interval(query[0],squared)
        for point in points:
            dx,dy=point[0]-query[0],point[1]-query[1]
            if dx*dx+dy*dy<=squared:
                assert low<=point[0]<=high
                checked+=1
    rows.append({"case":"outward interval encloses all qualifying original metric rows","passed":True,
        "qualifying_rows":checked,"random_2d_cases":500,"boundary_coordinates":len(coordinates)})
    path=ROOT/"scripts/research/native_corner_cpu.py"
    tree=ast.parse(path.read_text(encoding="utf-8"))
    imports=[node for node in tree.body if isinstance(node,(ast.Import,ast.ImportFrom))]
    names=[alias.name for node in imports for alias in node.names]
    assert not any(x.split(".")[0] in ("numpy","scipy","cupy","open3d","cv2") for x in names)
    assert "numpy" not in sys.modules and "scipy" not in sys.modules
    rows.append({"case":"module import remains stdlib only","passed":True})
    return {"kind":"native-corner-cpu-stdlib-contracts","status":"passed","cases":rows,
        "passed_cases":len(rows),"helper_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
        "scope":"Pure stdlib rounding/domain/source contracts. Candidate-tree and actual NumPy metric/binary correctness remain unexecuted, require fresh complete original CPU shadow.",
        "native_or_gpu_imports_run":False,"speed_claim":False}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve previous contract report")
    report=run()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print("PASS: %d stdlib CPU-corner contracts"%report["passed_cases"])
