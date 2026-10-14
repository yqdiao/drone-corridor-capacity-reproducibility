"""Exact checks for an executable equal-resource reservation rule.

Q assigns each of two publicly identified operators S/2 binding resource units
at zero access charge, contingent on both accepting the offer. Participation
uses the outside option zero. These are model checks, not an assertion that
equal quotas are an industry standard. The investment comparison uses alpha=1,
the same installation cost, and no revenue-recovery constraint.

Run: python3 checks/check_quota.py [--json results_quota.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sympy as sp


def run_checks():
    a, b, k, rho, eps, c1, c2, C, S, K, R = sp.symbols(
        "a b k rho eps c1 c2 C S K R", positive=True
    )
    x = sp.symbols("x", real=True)
    rho1, rho2 = rho + eps, rho
    rb = (rho1 + rho2) / 2
    D = a * eps + c2 - c1
    u = b * eps**2
    F = 2 * k + u
    H = 4 * k + u
    g = b * eps * (rho1 + rho2)
    eta1 = k + b * rho1 * eps
    eta2 = k - b * rho2 * eps
    A1 = rho1 * (a - b * rho2 * S) - c1 - C
    A2 = rho2 * (a - b * rho1 * S) - c2 - C
    X = rho2 * S + eps * x
    W = (a * X - b * X**2 / 2 - c1 * x - c2 * (S - x)
         - k * (x**2 + (S - x)**2) - C * S)
    xQ = S / 2
    xM = S / 2 + D / (4 * F)
    XQ = X.subs(x, xQ)
    XM = X.subs(x, xM)
    WQ = sp.expand(W.subs(x, xQ))
    WM = sp.expand(W.subs(x, xM))
    yM = (A1 + A2) / 2 - F * S
    pi1Q = (rho1 * (a - b * XQ) - c1 - C - k * S / 2) * S / 2
    pi2Q = (rho2 * (a - b * XQ) - c2 - C - k * S / 2) * S / 2
    pi1M = (rho1 * (a - b * XM) - c1 - C - k * xM - yM) * xM
    pi2M = (rho2 * (a - b * XM) - c2 - C - k * (S - xM) - yM) * (S - xM)
    L0 = a * rb - (c1 + c2) / 2 - C
    z = b * rb**2 + k
    A = R + z
    lam = g * D / (8 * F)
    CM = (6 * F + u) * D**2 / (32 * F**2)
    KQ = L0 / A
    KM = (L0 - lam) / A
    JQ = WQ.subs(S, K) - R * K**2 / 2
    JM = WM.subs(S, K) - R * K**2 / 2
    VQ = sp.cancel(JQ.subs(K, KQ))
    VM = sp.cancel(JM.subs(K, KM))
    fixed_gain = D * ((6 * F + u) * D - 4 * F * g * S) / (32 * F**2)
    optimized_gain = D * ((6 * F + u + g**2 / (4 * A)) * D - 4 * F * g * KQ) / (32 * F**2)
    checks = {
        "quota_service": XQ - rb * S,
        "quota_profit1_residual_value": pi1Q - (A1 * xQ - eta1 * xQ**2),
        "quota_profit2_residual_value": pi2Q - (A2 * xQ - eta2 * xQ**2),
        "M_profit1_participation": pi1M - (eta1 + 2 * eta2) * xM**2,
        "M_profit2_participation": pi2M - (eta2 + 2 * eta1) * (S - xM)**2,
        "service_M_minus_Q": XM - XQ - eps * D / (4 * F),
        "welfare_M_minus_Q": WM - WQ - fixed_gain,
        "fixed_gain_intercept": CM - (D**2 / (4 * F) - H * D**2 / (32 * F**2)),
        "fixed_gain_affine_in_release": fixed_gain - CM + lam * S,
        "quota_full_release_welfare": WQ.subs(S, K) - (L0 * K - z * K**2 / 2),
        "M_full_release_welfare": WM.subs(S, K) - (CM + (L0 - lam) * K - z * K**2 / 2),
        "quota_investment_foc": sp.diff(JQ, K).subs(K, KQ),
        "M_investment_foc": sp.diff(JM, K).subs(K, KM),
        "quota_investment_concavity": sp.diff(JQ, K, 2) + A,
        "M_investment_concavity": sp.diff(JM, K, 2) + A,
        "capacity_M_minus_Q": KM - KQ + b * rb * eps * D / (4 * F * A),
        "service_at_respective_optimal_capacities": XM.subs(S, KM) - XQ.subs(S, KQ) - eps * D * (R + k) / (4 * F * A),
        "net_value_Q": VQ - L0**2 / (2 * A),
        "net_value_M": VM - CM - (L0 - lam)**2 / (2 * A),
        "optimized_net_value_gain": VM - VQ - optimized_gain,
        "gain_at_Q_capacity_plus_readjustment": VM - VQ - (CM - lam * KQ + lam**2 / (2 * A)),
        "gain_at_M_capacity_minus_quota_opportunity_cost": VM - VQ - (CM - lam * KM - lam**2 / (2 * A)),
        "homogeneous_service_gap": (XM - XQ).subs(eps, 0),
        "homogeneous_capacity_gap": (KM - KQ).subs(eps, 0),
        "homogeneous_welfare_gain": (WM - WQ).subs(eps, 0) - 3 * (c2 - c1)**2 / (32 * k),
        "capital_gap_equals_service_gain_channel": KM - KQ + b * rb * (XM - XQ) / A,
        "constant_price_capacity_gap": (KM - KQ).subs(b, 0),
        "constant_price_service_gain": (XM - XQ).subs(b, 0) - eps * D / (8 * k),
        "constant_price_welfare_gain": (WM - WQ).subs(b, 0) - 3 * D**2 / (32 * k),
        "offsetting_types_allocation_gap": (xM - xQ).subs(c1, a * eps + c2),
        "offsetting_types_welfare_gap": (WM - WQ).subs(c1, a * eps + c2),
        "offsetting_types_capacity_gap": (KM - KQ).subs(c1, a * eps + c2),
    }
    for name, expression in checks.items():
        assert sp.factor(expression) == 0, (name, sp.factor(expression))

    domain_names = []

    def exact(value):
        value = sp.cancel(value)
        return {"exact": str(value), "decimal": float(value)}

    def positive(name, value):
        assert value > 0, (name, value)
        domain_names.append(name)

    base = {a: 10, b: 1, k: sp.Rational(1, 5), rho: 1,
            eps: sp.Rational(1, 10), c1: 2, c2: 2,
            C: sp.Rational(1, 10), R: sp.Rational(3, 2)}

    def scenario(name, parameters, expected_fixed_sign, expected_value_sign):
        def ev(expr, release):
            return sp.cancel(expr.subs(parameters | {S: release}))

        def at_release(release, tag):
            positive(tag + ": release", release)
            # Entire feasible service interval has positive downstream prices.
            positive(tag + ": p at largest service", ev(a - b * rho1 * S, release))
            for varname, value in (("eta1", eta1), ("eta2", eta2)):
                positive(tag + ": " + varname, ev(value, release))
            xm = ev(xM, release)
            positive(tag + ": M x1", xm)
            positive(tag + ": M x2", release - xm)
            rows = {}
            for label, split, output, welfare, price, pi1, pi2 in (
                ("Q", xQ, XQ, WQ, sp.Integer(0), pi1Q, pi2Q),
                ("M", xM, XM, WM, yM, pi1M, pi2M),
            ):
                positive(tag + ": " + label + " p", ev(a - b * output, release))
                positive(tag + ": " + label + " participation1", ev(pi1, release))
                positive(tag + ": " + label + " participation2", ev(pi2, release))
                rows[label] = {"x1": exact(ev(split, release)),
                               "x2": exact(release - ev(split, release)),
                               "X": exact(ev(output, release)), "W": exact(ev(welfare, release)),
                               "access_price": exact(ev(price, release)),
                               "profit1": exact(ev(pi1, release)), "profit2": exact(ev(pi2, release))}
            end1 = ev(yM - 2 * eta1 * (S - xM), release)
            end2 = ev(yM - 2 * eta2 * xM, release)
            for label, value in (("M price", ev(yM, release)), ("M endpoint1", end1), ("M endpoint2", end2)):
                positive(tag + ": " + label, value)
            return {"S": exact(release), "allocations": rows,
                    "M_endpoint1": exact(end1), "M_endpoint2": exact(end2),
                    "welfare_M_minus_Q": exact(ev(fixed_gain, release)),
                    "service_M_minus_Q": exact(ev(XM - XQ, release)),
                    "fixed_welfare_threshold_for_D_positive": exact(ev(4 * F * g * S / (6 * F + u), release))}

        fixed = at_release(sp.Integer(3), name + ": fixed")
        dd = ev(D, sp.Integer(3))
        assert sp.sign(ev(fixed_gain, 3)) == expected_fixed_sign
        assert sp.sign(ev(VM - VQ, 3)) == expected_value_sign
        km, kq = ev(KM, 3), ev(KQ, 3)
        positive(name + ": K_M", km)
        positive(name + ": K_Q", kq)
        assert sp.sign(km - kq) == -sp.sign(dd)
        optimized_service_gap = ev(XM.subs(S, KM) - XQ.subs(S, KQ), 3)
        assert sp.sign(optimized_service_gap) == sp.sign(dd)
        inv = {"M": {"K": exact(km), "net_value": exact(ev(VM, 3)),
                     "at_own_capacity": at_release(km, name + ": K_M")},
               "Q": {"K": exact(kq), "net_value": exact(ev(VQ, 3)),
                     "at_own_capacity": at_release(kq, name + ": K_Q")}}
        # All strict-domain constraints used here are affine in S on the
        # full-release path, after dividing participation profits by S/2.
        # Checking both capacities therefore checks the whole connecting
        # interval. Verify that affine property exactly, not just by sampling.
        affine_constraints = {
            "downstream_price": a - b * rho1 * S,
            "M_x1": xM, "M_x2": S - xM,
            "M_price": yM,
            "M_endpoint1": yM - 2 * eta1 * (S - xM),
            "M_endpoint2": yM - 2 * eta2 * xM,
            "Q_participation_margin1": pi1Q / (S / 2),
            "Q_participation_margin2": pi2Q / (S / 2),
        }
        for cname, expr in affine_constraints.items():
            assert sp.factor(sp.diff(expr, S, 2)) == 0, cname
        result = {"name": name,
                  "parameters": {str(key): str(value) for key, value in parameters.items()},
                  "D": exact(dd), "fixed_release": fixed, "investment": inv,
                  "capacity_M_minus_Q": exact(km - kq),
                  "optimized_service_M_minus_Q": exact(optimized_service_gap),
                  "optimized_net_value_M_minus_Q": exact(ev(VM - VQ, 3)),
                  "optimized_welfare_threshold_for_D_positive": exact(ev(4 * F * g * KQ / (6 * F + u + g**2 / (4 * A)), 3)),
                  "joint_investment_interval_domain": "Both endpoint checks passed; all defining constraints are affine after dividing Q participation by S/2."}
        return result

    scenarios = [
        scenario("service_conversion_gain", base, 1, 1),
        scenario("cost_service_tradeoff", base | {c1: sp.Rational(14, 5)}, -1, -1),
        scenario("D_zero_boundary", base | {c1: 3}, 0, 0),
        scenario("D_negative_cost_advantage", base | {c1: sp.Rational(16, 5)}, 1, 1),
    ]
    return {"status": "passed", "symbolic_checks": len(checks), "symbolic_names": list(checks),
            "strict_domain_checks": len(domain_names), "strict_domain_names": domain_names,
            "scope": "Q is an offered equal binding-resource reservation rule with zero access fee and positive operator participation; alpha=1 investment is conditional on the common maintained domain and excludes a revenue-recovery constraint.",
            "scenarios": scenarios}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    result = run_checks()
    print(json.dumps(result, indent=2))
    if args.json:
        args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    main()
