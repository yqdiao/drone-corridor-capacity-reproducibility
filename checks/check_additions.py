"""Independent checks of the October 20 geometry, release, and design-space claims.

Run from the manuscript root with ``python3 checks/check_additions.py``.
The 30,000-point baseline uses the original generator's draw order and seed.
Range sensitivity changes one interval at a time around its midpoint, using
common random numbers; nonnegative physical bounds are truncated at zero.
The Latin-hypercube comparison uses the unmodified box and a separate fixed
seed. These are design-space measures, not empirical probabilities.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import scipy
import sympy as sp
from scipy.optimize import brentq
from scipy.stats import qmc

OUT = Path(__file__).resolve().parents[1] / "data" / "additional_audit"
OUT.mkdir(parents=True, exist_ok=True)
N = 30000
SEED = 20261014
RANGES = {
    "a": [8., 12.], "b": [.7, 1.3], "k": [.16, .40],
    "epsilon": [.03, .18], "cost_premium": [-.3, 1.5],
    "B": [.05, .25], "gamma": [0., .30], "S": [2.2, 3.8],
    "R": [1., 2.5],
}


def exact_checks():
    b, eps, rb, k, S, K, R, L0 = sp.symbols(
        "b eps rb k S K R L0", positive=True)
    D, xi, cdelta, pQ = sp.symbols("D xi cdelta pQ", real=True)
    B, gamma = sp.symbols("B gamma", nonnegative=True)
    u = b*eps**2
    F, H = 2*k+u, 4*k+u
    sigma = b*eps*rb*S
    phi = u/F
    zeta = b*rb*eps*D/(4*F)
    xiW, xiM = (D-sigma)/H, D/(4*F)
    gain = (D-sigma)*xi-H*xi**2/2
    dst = 8*sigma/(6+phi)
    WQ = sp.Symbol("AQ")*S-(b*rb**2+k)*S**2/2-B*S**2/K-gamma*S**3/(2*K**2)
    CM = (6*F+u)*D**2/(32*F**2)
    WM = WQ+CM-zeta*S
    z = b*rb**2+k
    checks = {
        "geometry_distance": gain-H*(xiW**2-(xi-xiW)**2)/2,
        "geometry_M_threshold_factor": gain.subs(xi, xiM)-
            D*((6*F+u)*D-8*F*sigma)/(32*F**2),
        "price_threshold_equivalence": (D-dst).subs(D,eps*(pQ+b*rb*S)-cdelta)-
            eps*(pQ-(2-phi)*b*rb*S/(6+phi)-cdelta/eps),
        "equal_split_efficient_case": gain.subs(xi, xiM).subs(D,sigma)+
            H*sigma**2/(32*F**2),
        "release_marginal_difference": sp.diff(WM-WQ,S)+zeta,
        "capacity_partial_same_at_same_release": sp.diff(WM-WQ,K),
        "capacity_relief_derivative": sp.diff(WQ,K)-
            (B*S**2/K**2+gamma*S**3/K**3),
        "relative_capacity_gap": ((L0/(R+z)-(L0-zeta)/(R+z))/(L0/(R+z)))-
            zeta/L0,
        "relative_gap_R_derivative": sp.diff(zeta/L0,R),
        "absolute_gap_R_derivative": sp.diff(zeta/(R+z),R)+zeta/(R+z)**2,
        "slope_ratio": H/(4*F)-(2-phi)/4,
    }
    for name, expr in checks.items():
        assert sp.factor(expr) == 0, (name, sp.factor(expr))
    # For positive unconstrained utilization v:
    # A=(zK+2B)v+(3gamma/2)v^2.
    v = sp.symbols("v", positive=True)
    vp = -z*v/(z*K+2*B+3*gamma*v)
    Gp = (2*B*v+3*gamma*v**2)*vp
    assert sp.factor(Gp + z*v**2*(2*B+3*gamma*v)/(z*K+2*B+3*gamma*v)) == 0
    return {"count":len(checks)+1, "names":list(checks)+["slack_capacity_marginal_derivative"],
            "status":"passed",
            "slack_net_marginal_derivative":"-z*v^2*(2*B+3*gamma*v)/(z*K+2*B+3*gamma*v)-R < 0",
            "binding_net_marginal_derivative":"-z-R < 0",
            "qualification":"Applies to the algebraic positive-release candidates. The economic conclusion requires the maintained domains along the interval of compared capacities."}


def at_release(d, S, K):
    a,b,k,e,dc,B,gam = (d[n] for n in
                         ("a","b","k","epsilon","cost_premium","B","gamma"))
    r1=1+e
    rb=1+e/2
    c1=2+dc
    D=a*e-dc
    F=2*k+b*e**2
    eta1=k+b*r1*e
    eta2=k-b*e
    xm=S/2+D/(4*F)
    C=B*S/K+gam*S**2/(2*K**2)
    XQ=rb*S
    XM=XQ+e*D/(4*F)
    y=(r1*(a-b*S)-c1-C+a-b*r1*S-2-C)/2-F*S
    end1=y-2*eta1*(S-xm)
    end2=y-2*eta2*xm
    piq1=S/2*(r1*(a-b*XQ)-c1-C-k*S/2)
    piq2=S/2*(a-b*XQ-2-C-k*S/2)
    pim1=(eta1+2*eta2)*xm**2
    pim2=(eta2+2*eta1)*(S-xm)**2
    tests=[S,K,eta1,eta2,xm,S-xm,a-b*XM,y,end1,end2,piq1,piq2]
    domain=np.logical_and.reduce([v>0 for v in tests])
    return {"domain":domain,"piQ1":piq1,"piQ2":piq2,"piM1":pim1,"piM2":pim2,
            "Q_price":a-b*XQ,"M_price":a-b*XM,"endpoint1":end1,"endpoint2":end2}


def evaluate(d):
    a,b,k,e,dc,B,gam,S,R=(d[n] for n in RANGES)
    rb=1+e/2
    D=a*e-dc
    u=b*e**2
    F=2*k+u
    g=2*b*e*rb
    z=b*rb**2+k
    L0=a*rb-(4+dc)/2-B-gam/2
    A=R+z
    zeta=g*D/(8*F)
    KQ=L0/A
    KM=(L0-zeta)/A
    CM=(6*F+u)*D**2/(32*F**2)
    fixed=at_release(d,S,S)["domain"] & (D>0)
    invest=at_release(d,KQ,KQ)["domain"] & at_release(d,KM,KM)["domain"] & (D>0)
    wf=CM-zeta*S
    nv=CM-zeta*KQ+zeta**2/(2*A)
    return {"D":D,"fixed_feasible":fixed,"investment_feasible":invest,
            "fixed_welfare_gap":wf,"optimized_net_value_gap":nv,
            "capacity_reduction":KQ-KM,"relative_capacity_reduction":zeta/L0,
            "KQ":KQ,"KM":KM,"fixed_service_gain":e*D/(4*F),
            "own_capacity_service_gain":e*D/(4*F)*(R+k)/A}


def summarize(name,ranges,d):
    v=evaluate(d)
    f=v["fixed_feasible"]
    i=v["investment_feasible"]
    ratio=v["relative_capacity_reduction"][i]
    gap=v["capacity_reduction"][i]
    assert np.all(v["fixed_service_gain"][f]>0)
    assert np.all(gap>0) and np.all(v["own_capacity_service_gain"][i]>0)
    return {"design":name,"ranges":ranges,
            "fixed_feasible_positive_D":int(f.sum()),
            "fixed_M_share":float(np.mean(v["fixed_welfare_gap"][f]>0)),
            "investment_feasible_positive_D":int(i.sum()),
            "investment_M_share":float(np.mean(v["optimized_net_value_gap"][i]>0)),
            "relative_capacity_quantiles":{str(q):float(np.quantile(ratio,q)) for q in (0,.05,.25,.5,.75,.95,1)},
            "absolute_capacity_quantiles":{str(q):float(np.quantile(gap,q)) for q in (0,.5,.95,1)}},v


def design_sensitivity():
    rng=np.random.default_rng(SEED)
    # Separate columns preserve the original sequential RNG consumption.
    uniforms={key:rng.random(N) for key in RANGES}
    def transform(box,us=uniforms):
        return {key:lo+(hi-lo)*us[key] for key,(lo,hi) in box.items()}
    baseline,base_values=summarize("uniform_baseline",RANGES,transform(RANGES))
    assert baseline["fixed_feasible_positive_D"]==18650,baseline
    assert baseline["investment_feasible_positive_D"]==21364,baseline
    records=[baseline]
    for key,(lo,hi) in RANGES.items():
        for scale in (.5,2.):
            mid=(lo+hi)/2
            half=(hi-lo)*scale/2
            bounds=[mid-half,mid+half]
            if key!="cost_premium":
                bounds[0]=max(0.,bounds[0])
            box={n:list(ab) for n,ab in RANGES.items()}
            box[key]=bounds
            row,_=summarize(f"{key}_width_x{scale:g}",box,transform(box))
            records.append(row)
    lhs=qmc.LatinHypercube(d=len(RANGES),seed=20261020).random(N)
    lu={key:lhs[:,j] for j,key in enumerate(RANGES)}
    row,_=summarize("latin_hypercube_baseline",RANGES,transform(RANGES,lu))
    records.append(row)
    cols=["design","fixed_feasible_positive_D","fixed_M_share",
          "investment_feasible_positive_D","investment_M_share"]
    with (OUT/"range_sensitivity_summary.csv").open("w",newline="") as fp:
        w=csv.DictWriter(fp,fieldnames=cols)
        w.writeheader()
        w.writerows({k:r[k] for k in cols} for r in records)
    allcols={**transform(RANGES),**base_values}
    with (OUT/"baseline_design_space_full.csv").open("w",newline="") as fp:
        w=csv.writer(fp)
        w.writerow(allcols)
        w.writerows(zip(*allcols.values()))
    return {"draws":N,"uniform_seed":SEED,"lhs_seed":20261020,
            "method":"One interval at a time has its half-width scaled around the original midpoint; nonnegative physical lower bounds are truncated at zero. The other eight intervals and common random numbers are unchanged. LHS uses the original intervals.",
            "records":records}


def release_tests():
    states=np.linspace(8,17,241)
    d={"a":states,"b":.75,"k":.16,"epsilon":.16,"cost_premium":.7,
       "B":2.2,"gamma":2.2,"S":0.,"R":1.5}
    # This release figure has c2=2.3, unlike the scan's fixed c2=2.
    rb=1.08
    z=.75*rb**2+.16
    D=states*.16-.7
    F=2*.16+.75*.16**2
    zeta=.75*rb*.16*D/(4*F)
    AQ=states*rb-2.65
    AM=AQ-zeta
    B=gam=2.2
    def at(K):
        theta=z*K+2*B
        def one(A):
            v=2*A/(theta+np.sqrt(theta**2+6*gam*A))
            v=np.minimum(1.,v)
            S=K*v
            mu=np.maximum(0.,A-z*K-2*B-1.5*gam)
            return S,B*v**2+gam*v**3+mu
        SQ,GQ=one(AQ)
        SM,GM=one(AM)
        assert np.all(SM<=SQ+1e-12)
        assert np.all(GM<=GQ+1e-12)
        return SQ,SM,GQ,GM
    SQ,SM,GQ,GM=at(5.5)
    # Actual Q participation at Q's own release and M endpoint bids at M's.
    def margins(S,K,rule):
        C=B*S/K+gam*S**2/(2*K**2)
        eta1=.16+.75*1.16*.16
        eta2=.16-.75*.16
        if rule=="Q":
            X=rb*S
            return np.minimum(1.16*(states-.75*X)-3-C-.16*S/2,
                              states-.75*X-2.3-C-.16*S/2)
        xm=S/2+D/(4*F)
        y=(1.16*(states-.75*S)-3-C+states-.75*1.16*S-2.3-C)/2-F*S
        return np.minimum.reduce([xm,S-xm,states-.75*(rb*S+.16*D/(4*F)),
                                  y,y-2*eta1*(S-xm),y-2*eta2*xm])
    assert np.all(margins(SQ,5.5,"Q")>0)
    assert np.all(margins(SM,5.5,"M")>0)
    Kgrid=np.linspace(.5,12,2001)
    nq=[];nm=[]
    for K in Kgrid:
        _,_,gq,gm=at(K)
        nq.append(float(gq.mean()-1.5*K))
        nm.append(float(gm.mean()-1.5*K))
    assert np.all(np.diff(nq)<0) and np.all(np.diff(nm)<0)
    def root(rule):
        return brentq(lambda K:float(at(K)[2 if rule=="Q" else 3].mean()-1.5*K),.5,12)
    KQ,KM=root("Q"),root("M")
    assert KM<KQ
    minQ=minM=float("inf")
    for K in np.linspace(KM,KQ,101):
        sq,sm,_,_=at(K)
        minQ=min(minQ,float(margins(sq,K,"Q").min()))
        minM=min(minM,float(margins(sm,K,"M").min()))
    assert min(minQ,minM)>0
    with (OUT/"release_capacity_single_crossing.csv").open("w",newline="") as fp:
        w=csv.writer(fp)
        w.writerow(["K","net_marginal_Q","net_marginal_M"])
        w.writerows(zip(Kgrid,nq,nm))
    with (OUT/"release_ordering.csv").open("w",newline="") as fp:
        w=csv.writer(fp)
        w.writerow(["a","SQ","SM","marginal_capacity_Q","marginal_capacity_M"])
        w.writerows(zip(states,SQ,SM,GQ,GM))
    capQ=(z*5.5+2*B+1.5*gam+2.65)/rb
    zeta_slope=.75*rb*.16*.16/(4*F)
    zeta_intercept=-.75*rb*.16*.7/(4*F)
    capM=(z*5.5+2*B+1.5*gam+2.65+zeta_intercept)/(rb-zeta_slope)
    return {"states":241,"a_range":[8,17],"K_figure":5.5,
            "figure_min_Q_participation_margin":float(margins(SQ,5.5,"Q").min()),
            "figure_min_M_domain_margin":float(margins(SM,5.5,"M").min()),
            "release_gap_max":float((SQ-SM).max()),
            "exact_Q_binding_a":float(capQ),"exact_M_binding_a":float(capM),
            "numerical_single_crossing_distribution":"Uniform discrete distribution on the same 241 a-values, introduced for this audit only; Figure itself fixes K.",
            "R":1.5,"KQ":KQ,"KM":KM,
            "interval_min_Q_participation_margin":minQ,
            "interval_min_M_domain_margin":minM,
            "capacity_grid_count":2001,"capacity_grid_range":[.5,12],
            "strict_monotonicity":"passed for both net marginal schedules",
            "scope":"Grid checks support this declared distribution. The algebraic derivative check establishes monotonicity without assuming this distribution."}


def main():
    result={"versions":{"numpy":np.__version__,"scipy":scipy.__version__,"sympy":sp.__version__},
            "exact_checks":exact_checks(),
            "release_tests":release_tests(),
            "design_sensitivity":design_sensitivity()}
    (OUT/"additional_audit_results.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="design_sensitivity"},indent=2))
    for r in result["design_sensitivity"]["records"]:
        print(r["design"],r["fixed_feasible_positive_D"],round(100*r["fixed_M_share"],3),
              r["investment_feasible_positive_D"],round(100*r["investment_M_share"],3))
    print("baseline gap distribution",result["design_sensitivity"]["records"][0]["relative_capacity_quantiles"])


if __name__=="__main__":
    main()
