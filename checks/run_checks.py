"""Run all exact type-extension checks and print the manuscript scenario.

Usage: python3 checks/run_checks.py
Optional: python3 checks/run_checks.py --json checks/results.json

Only SymPy plus the Python standard library is needed. The optional JSON is a
generated reproducibility artifact. This program does not edit manuscript files.
"""
from __future__ import annotations
import argparse
import contextlib
import io
import json
from pathlib import Path
import runpy

import sympy as sp
from check_quota import run_checks as run_quota_checks

ROOT = Path(__file__).resolve().parent


def execute(filename: str):
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        namespace = runpy.run_path(str(ROOT / filename), run_name="__main__")
    return namespace, output.getvalue()


def exact_number(value):
    exact = sp.cancel(value)
    return {"exact": str(exact), "decimal": float(exact)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="Optional generated JSON results path")
    options = parser.parse_args()
    _, auction_text = execute("check_auction.py")
    auction = json.loads(auction_text)
    n, _ = execute("check_welfare_investment.py")

    def ev(expression, release=sp.Integer(3), alpha=sp.Integer(1)):
        return sp.cancel(expression.subs(n["params"] | {n["S"]:release, n["alpha"]:alpha}))

    conditions = []

    def positive(name, value):
        assert value > 0, (name, value)
        conditions.append(name)

    def interior(name, value, release):
        positive(name + " > 0", value)
        positive(name + " < S", release-value)

    for name in ("a", "b", "k", "R", "e", "r"):
        positive("primitive " + name, ev(n[name]))
    for name in ("beta1", "beta2"):
        positive(name, ev(n[name]))

    def at_release(release, prefix):
        positive(prefix + ": S", release)
        positive(prefix + ": p at maximum feasible service", ev(n["a"]-n["b"]*n["r1"]*n["S"],release))
        rows = {}
        for label, expr in (("M", n["xM"]), ("T", n["xT"]), ("W", n["xW"])):
            xi = ev(expr, release)
            interior(prefix + ": x1_" + label, xi, release)
            output = ev(n["X"].subs(n["x"],expr), release)
            welfare = ev(n["W"].subs(n["x"],expr), release)
            service_price = ev(n["a"]-n["b"]*n["X"].subs(n["x"],expr), release)
            positive(prefix + ": service price " + label, service_price)
            rows[label] = {
                "x1":exact_number(xi), "x2":exact_number(release-xi),
                "X":exact_number(output), "W":exact_number(welfare),
                "service_price":exact_number(service_price),
            }
        for label, price in (("M",n["yM"]),("T",n["yT"])):
            yy=ev(price, release)
            positive(prefix + ": auction price " + label, yy)
            rows[label]["auction_price"]=exact_number(yy)
        end1=ev(n["A1"]-2*n["beta2"]*n["xM"]-2*n["beta1"]*n["S"],release)
        end2=ev(n["A2"]-2*n["beta1"]*(n["S"]-n["xM"])-2*n["beta2"]*n["S"],release)
        mv1=ev(n["A1"]-2*n["beta1"]*n["S"],release)
        mv2=ev(n["A2"]-2*n["beta2"]*n["S"],release)
        for label,val in (("M bid1(S)",end1),("M bid2(S)",end2),("T marginal1(S)",mv1),("T marginal2(S)",mv2)):
            positive(prefix + ": " + label,val)
        gain=ev(n["WM"]-n["WT"],release)
        assert sp.cancel(gain-ev(n["diff"],release))==0
        # Explicitly verify the displayed fixed-release service/welfare interval.
        lower=ev(2*n["u"]*n["g"]*n["S"]/(2*n["F"]+3*n["u"]),release)
        dd=ev(n["D"],release)
        upper=ev(2*n["g"]*n["S"],release)
        positive(prefix+": D-lower welfare threshold",dd-lower)
        positive(prefix+": upper welfare threshold-D",upper-dd)
        return {
            "S":exact_number(release), "allocations":rows,
            "M_bid1_at_S":exact_number(end1), "M_bid2_at_S":exact_number(end2),
            "T_marginal1_at_S":exact_number(mv1), "T_marginal2_at_S":exact_number(mv2),
            "welfare_M_minus_T":exact_number(gain),
            "service_M_minus_T":exact_number(ev(n["X"].subs(n["x"],n["xM"])-n["X"].subs(n["x"],n["xT"]),release)),
        }

    fixed=at_release(sp.Integer(3),"fixed release")
    # These are the explicitly reported rounded manuscript values, independently
    # compared against exact substitution. They are not fitted observations.
    expected={
        "M":{"x1":"2.109756098","X":"3.210975610","W":"19.605852915"},
        "T":{"x1":"1.951219512","X":"3.195121951","W":"19.565377751"},
        "W":{"x1":"2.345679012","X":"3.234567901","W":"19.628395062"},
    }
    for label,fields in expected.items():
        for field,rounded in fields.items():
            actual=sp.Rational(fixed["allocations"][label][field]["exact"])
            assert abs(actual-sp.Rational(rounded))<=sp.Rational(1,2_000_000_000),(label,field,actual,rounded)

    investment=[]
    for av in (sp.Integer(1),sp.Integer(2)):
        capacities={}
        for label,kexpr in (("M",n["KM"]),("T",n["KT"])):
            kval=ev(kexpr,alpha=av)
            positive(f"alpha={av}, K_{label}",kval)
            ww=n["WM"] if label=="M" else n["WT"]
            yy=n["yM"] if label=="M" else n["yT"]
            objective=(ww+(n["alpha"]-1)*yy*n["S"]-n["R"]*n["S"]**2/2)
            assert ev(sp.diff(objective,n["S"]),kval,av)==0
            positive(f"alpha={av}, strict full-release objective concavity {label}",-ev(sp.diff(objective,n["S"],2),kval,av))
            # Interpret Cbar=.1 as B=.1,gamma=0. At S=K, allowing release
            # separately leaves a nonnegative capacity multiplier in this example.
            state_objective=(ww+(n["alpha"]-1)*yy*n["S"]).subs(n["C"],sp.Rational(1,10)*n["S"]/n["K"])
            release_gradient=sp.diff(state_objective,n["S"]).subs(n["K"],kval)
            mu=ev(release_gradient,kval,av)
            positive(f"alpha={av}, full-release multiplier {label}",mu)
            capacities[label]={
                "K":exact_number(kval),
                "full_release_multiplier_B_0p1_gamma_0":exact_number(mu),
                "at_own_capacity":at_release(kval,f"alpha={av}, K_{label}"),
            }
        investment.append({"alpha":int(av),"capacities":capacities})
    km1=sp.Rational(investment[0]["capacities"]["M"]["K"]["exact"])
    kt1=sp.Rational(investment[0]["capacities"]["T"]["K"]["exact"])
    assert km1>kt1
    assert ev(n["KM"]-n["KT"],alpha=2)<0
    symbolic_count=len(auction["symbolic_checks_passed"])+len(n["checks"])
    result={
        "status":"passed",
        "scope":"Conditional public-type extension; T is a non-strategic marginal-profit benchmark, not an IC mechanism.",
        "congestion_convention":"COMMON PER RESOURCE: total congestion C(S,K)*S. Resource weights h_i can differ after normalizing rho_i=d_i/h_i, c_i=t_i/h_i, k_i=kappa_i/h_i^2.",
        "not_established":["Empirical calibration","Literature priority","Global equilibrium selection","Pairwise or sequence-dependent aircraft separation"],
        "symbolic_checks":symbolic_count,
        "symbolic_names":auction["symbolic_checks_passed"]+list(n["checks"]),
        "strict_domain_checks":len(conditions),
        "strict_domain_names":conditions,
        "parameters":{"a":10,"b":1,"kappa":"1/5","rho1":"11/10","rho2":1,"c1":2,"c2":2,"Cbar":"1/10","R":"3/2","fixed_S":3},
        "fixed_release":fixed,
        "investment":investment,
        "auction_additional_scenarios":auction["scenarios"],
    }
    print(f"PASS: {symbolic_count} symbolic identities; {len(conditions)} strict domain checks.")
    print("Illustrative fixed-release comparison, S=3 (not calibrated):")
    print("branch       x1            x2             X              W          access price")
    for label in ("M","T","W"):
        rr=fixed["allocations"][label]
        price=rr.get("auction_price")
        price_text=f'{price["decimal"]:.9f}' if price else "n/a"
        print(f'{label:>3}   {rr["x1"]["decimal"]:12.9f}  {rr["x2"]["decimal"]:12.9f}  {rr["X"]["decimal"]:12.9f}  {rr["W"]["decimal"]:13.9f}   {price_text}')
    print("M endpoint bids:",fixed["M_bid1_at_S"]["decimal"],fixed["M_bid2_at_S"]["decimal"])
    print("T endpoint marginal-profit values:",fixed["T_marginal1_at_S"]["decimal"],fixed["T_marginal2_at_S"]["decimal"])
    for row in investment:
        cc=row["capacities"]
        print(f'alpha={row["alpha"]}: K_M={cc["M"]["K"]["decimal"]:.12f}, K_T={cc["T"]["K"]["decimal"]:.12f}; both own-capacity domains pass')
    quota = run_quota_checks()
    result["quota_policy_comparison"] = quota
    result["combined_symbolic_checks"] = symbolic_count + quota["symbolic_checks"]
    result["combined_strict_domain_checks"] = len(conditions) + quota["strict_domain_checks"]
    print(f'Combined PASS: {result["combined_symbolic_checks"]} symbolic identities; {result["combined_strict_domain_checks"]} strict domain checks.')
    if options.json:
        options.json.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
        print(f"Generated JSON: {options.json}")
    return result


if __name__=="__main__":
    main()
