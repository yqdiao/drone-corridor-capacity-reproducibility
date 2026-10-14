"""Exact audit of a selected fixed-supply, heterogeneous-service auction branch.

This does not establish uniqueness, novel priority, or implementation of a
truth-telling mechanism. It checks the algebra and a sufficient global-deviation
domain. Run with Python and SymPy; no files are modified by execution.
"""

import json
import sympy as sp


def main():
    A1, A2, B1, B2, S, x = sp.symbols("A1 A2 B1 B2 S x", real=True)
    M = B1 + B2
    xm1 = S / 2 + (A1 - A2) / (4 * M)
    xm2 = S - xm1
    y = (A1 + A2) / 2 - M * S
    s1, s2 = 2 * B2 * xm1, 2 * B1 * xm2
    identities = []

    def check(name, expr):
        assert sp.simplify(expr) == 0, (name, sp.factor(expr))
        identities.append(name)

    check("bidder 1 clearing", A1 - 2 * B1 * xm1 - s1 - y)
    check("bidder 2 clearing", A2 - 2 * B2 * xm2 - s2 - y)
    for i, (Ai, Bi, Bj, Aj, sj, xi) in enumerate(
        [(A1, B1, B2, A2, s2, xm1), (A2, B2, B1, A1, s1, xm2)], 1
    ):
        envelope = Ai * x - Bi * x**2 - (Aj - sj - 2 * Bj * (S - x)) * x
        check(f"bidder {i} envelope FOC", sp.diff(envelope, x).subs(x, xi))
        check(f"bidder {i} envelope SOC", sp.diff(envelope, x, 2) + 2 * (Bi + 2 * Bj))
        check(f"bidder {i} equilibrium profit", envelope.subs(x, xi) - (Bi + 2 * Bj) * xi**2)
        check(f"bidder {i} global deviation loss", envelope.subs(x, xi) - envelope - (Bi + 2 * Bj) * (x - xi)**2)

    r1, r2, b, a, c1, c2, k1, k2 = sp.symbols("r1 r2 b a c1 c2 k1 k2", real=True)
    d = r1 - r2
    subst = {
        A1: r1 * (a - b * r2 * S) - c1,
        A2: r2 * (a - b * r1 * S) - c2,
        B1: b * r1 * d + k1,
        B2: -b * r2 * d + k2,
    }
    mm = b * d**2 + k1 + k2
    H = b * d**2 + 2 * (k1 + k2)
    D = a * d - (c1 - c2)
    xmm = S / 2 + D / (4 * mm)
    xw = (D - b * d * r2 * S + 2 * k2 * S) / H
    xt = (D + 2 * (k2 - b * r2 * d) * S) / (2 * mm)
    X = r2 * S + d * x
    W = a * X - b * X**2 / 2 - c1 * x - c2 * (S - x) - k1 * x**2 - k2 * (S - x)**2
    check("heterogeneous candidate reduction", xm1.subs(subst) - xmm)
    check("welfare FOC", sp.diff(W, x).subs(x, xw))
    check("welfare curvature", sp.diff(W, x, 2) + H)
    check("welfare loss identity", W.subs(x, xw) - W - H * (x - xw)**2 / 2)
    check("M vs NS welfare identity", W.subs(x, xmm) - W.subs(x, xt) - H * ((xt - xw)**2 - (xmm - xw)**2) / 2)
    gate = D * (3 * b * d**2 + 2 * (k1 + k2)) / (4 * mm) - S * (k1 - k2 + b * (r1**2 - r2**2) / 2)
    check("direction of M allocation distortion", sp.diff(W, x).subs(x, xmm) - gate)

    def scenario(name, data):
        A = [subst[A1].subs(data), subst[A2].subs(data)]
        B = [subst[B1].subs(data), subst[B2].subs(data)]
        sd = {A1: A[0], A2: A[1], B1: B[0], B2: B[1], S: data[S]}
        q = [xm1.subs(sd), xm2.subs(sd)]
        price = y.subs(sd)
        endpoints = [price - 2 * B[0] * q[1], price - 2 * B[1] * q[0]]
        efficient = xw.subs(data)
        naive = xt.subs(data)
        welfare_gap = (W.subs(x, xmm) - W.subs(x, xt)).subs(data)
        assert all(v > 0 for v in B + q + endpoints + [price])
        assert 0 < efficient < data[S] and 0 < naive < data[S]
        return {
            "name": name,
            "x1M": str(q[0]), "x2M": str(q[1]), "priceM": str(price),
            "beta1": str(B[0]), "beta2": str(B[1]),
            "bid1_at_S": str(endpoints[0]), "bid2_at_S": str(endpoints[1]),
            "x1W": str(efficient), "x1NS": str(naive),
            "welfare_M_minus_NS": str(welfare_gap),
            "welfare_M_minus_NS_float": float(welfare_gap),
        }

    scenarios = [
        scenario("moderate productivity heterogeneity in the strict sufficient domain", {
            r1: sp.Rational(11, 10), r2: 1, b: 1, a: 10,
            c1: 2, c2: 2, k1: 1, k2: 1, S: 2}),
        scenario("strategic bidding closer to welfare allocation than marginal-profit bidding", {
            r1: sp.Rational(6, 5), r2: 1, b: 1, a: 10,
            c1: sp.Rational(14, 5), c2: 1, k1: 1, k2: 1, S: 1}),
        scenario("strategic bidding farther from welfare allocation than marginal-profit bidding", {
            r1: sp.Rational(6, 5), r2: 1, b: 1, a: 5,
            c1: 1, c2: 1, k1: 1, k2: 1, S: 1}),
    ]
    print(json.dumps({"symbolic_checks_passed": identities, "scenarios": scenarios}, indent=2))


if __name__ == "__main__":
    main()
