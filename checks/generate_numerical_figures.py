"""Reproduce the numerical experiments and current-model figures.

The script evaluates only the heterogeneous resource-productivity model in the
main paper.  It does not reuse the legacy homogeneous-model figures.  All
parameter scans have a fixed seed and write their machine-readable outputs to
``checks/numerical_results``.

Run from the manuscript root:

    python3 checks/generate_numerical_figures.py
"""

from __future__ import annotations

import csv
import json
import os
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "corridor-mplconfig"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "sections" / "generated"
RESULT_DIR = ROOT / "checks" / "numerical_results"
RANDOM_SEED = 20261014
DESIGN_DRAWS = 30000
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

BLUE = "#1f5a94"
ORANGE = "#d9782d"
GREEN = "#2f855a"
PURPLE = "#7561a8"
GREY = "#707070"
LIGHT_GREY = "#e7e7e7"

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 160,
        "savefig.bbox": "tight",
    }
)


def save_pdf(fig: plt.Figure, filename: str) -> None:
    """Write a PDF without clock-dependent metadata.

    Removing creation and modification timestamps makes repeated runs with the
    pinned environment byte-for-byte stable on the same platform.
    """
    fig.savefig(
        FIGURE_DIR / filename,
        metadata={"Creator": "generate_numerical_figures.py", "CreationDate": None, "ModDate": None},
    )


@dataclass(frozen=True)
class Parameters:
    a: float = 10.0
    b: float = 1.0
    k: float = 0.2
    rho1: float = 1.1
    rho2: float = 1.0
    c1: float = 2.0
    c2: float = 2.0
    B: float = 0.1
    gamma: float = 0.0
    S: float = 3.0
    K: float = 3.0
    R: float = 1.5


def quantities(p: Parameters) -> dict[str, float]:
    eps = p.rho1 - p.rho2
    rb = (p.rho1 + p.rho2) / 2
    d = p.a * eps + p.c2 - p.c1
    u = p.b * eps**2
    f = 2 * p.k + u
    h = 4 * p.k + u
    g = p.b * eps * (p.rho1 + p.rho2)
    eta1 = p.k + p.b * p.rho1 * eps
    eta2 = p.k - p.b * p.rho2 * eps
    congestion = p.B * p.S / p.K + p.gamma * p.S**2 / (2 * p.K**2)

    xq = p.S / 2
    xm = p.S / 2 + d / (4 * f)
    xt = p.S / 2 + (d - g * p.S) / (2 * f)
    xw = (d + (2 * p.k - p.b * eps * p.rho2) * p.S) / h

    def service(x: float) -> float:
        return p.rho2 * p.S + eps * x

    def welfare(x: float) -> float:
        X = service(x)
        return (
            p.a * X
            - p.b * X**2 / 2
            - p.c1 * x
            - p.c2 * (p.S - x)
            - p.k * (x**2 + (p.S - x) ** 2)
            - congestion * p.S
        )

    a1 = p.rho1 * (p.a - p.b * p.rho2 * p.S) - p.c1 - congestion
    a2 = p.rho2 * (p.a - p.b * p.rho1 * p.S) - p.c2 - congestion
    ym = (a1 + a2) / 2 - f * p.S
    end1 = ym - 2 * eta1 * (p.S - xm)
    end2 = ym - 2 * eta2 * xm
    xq_service = service(xq)
    piq1 = p.S / 2 * (
        p.rho1 * (p.a - p.b * xq_service)
        - p.c1
        - congestion
        - p.k * p.S / 2
    )
    piq2 = p.S / 2 * (
        p.rho2 * (p.a - p.b * xq_service)
        - p.c2
        - congestion
        - p.k * p.S / 2
    )
    fixed_domain = bool(
        min(
            eta1,
            eta2,
            xm,
            p.S - xm,
            p.a - p.b * service(xm),
            ym,
            end1,
            end2,
            piq1,
            piq2,
        )
        > 0
    )
    comparison_domain = bool(
        fixed_domain
        and min(xt, p.S - xt, xw, p.S - xw, p.a - p.b * service(xt)) > 0
    )
    threshold = 4 * f * g * p.S / (6 * f + u)
    return {
        "epsilon": eps,
        "rho_bar": rb,
        "D": d,
        "u": u,
        "F": f,
        "H": h,
        "g": g,
        "eta1": eta1,
        "eta2": eta2,
        "xQ": xq,
        "xM": xm,
        "xT": xt,
        "xW": xw,
        "XQ": service(xq),
        "XM": service(xm),
        "XT": service(xt),
        "XW": service(xw),
        "WQ": welfare(xq),
        "WM": welfare(xm),
        "WT": welfare(xt),
        "WW": welfare(xw),
        "yM": ym,
        "endpoint1": end1,
        "endpoint2": end2,
        "piQ1": piq1,
        "piQ2": piq2,
        "fixed_domain": fixed_domain,
        "comparison_domain": comparison_domain,
        "fixed_threshold": threshold,
    }


def investment(p: Parameters) -> dict[str, float | bool]:
    q = quantities(p)
    eps, rb, d, u, f, g = (
        q["epsilon"],
        q["rho_bar"],
        q["D"],
        q["u"],
        q["F"],
        q["g"],
    )
    cbar = (p.c1 + p.c2) / 2
    congestion_at_full_release = p.B + p.gamma / 2
    l0 = p.a * rb - cbar - congestion_at_full_release
    z = p.b * rb**2 + p.k
    curvature = p.R + z
    zeta = g * d / (8 * f)
    cm = (6 * f + u) * d**2 / (32 * f**2)
    kq = l0 / curvature
    km = (l0 - zeta) / curvature
    vq = l0**2 / (2 * curvature)
    vm = cm + (l0 - zeta) ** 2 / (2 * curvature)
    xq = rb * kq
    xm = rb * km + eps * d / (4 * f)

    def at_capacity(capacity: float) -> bool:
        if capacity <= 0:
            return False
        return quantities(
            Parameters(
                **{
                    **asdict(p),
                    "S": float(capacity),
                    "K": float(capacity),
                }
            )
        )["fixed_domain"]

    feasible = at_capacity(kq) and at_capacity(km)
    net_threshold = 4 * f * g * kq / (6 * f + u + g**2 / (4 * curvature))
    return {
        "KQ": kq,
        "KM": km,
        "XQ_own": xq,
        "XM_own": xm,
        "VQ": vq,
        "VM": vm,
        "capacity_gap": km - kq,
        "service_gap": xm - xq,
        "net_value_gap": vm - vq,
        "net_threshold": net_threshold,
        "feasible": feasible,
    }


def write_rows(path: Path, rows: list[dict[str, float | int | bool]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def policy_regions() -> dict[str, float]:
    base = Parameters()
    ds = np.linspace(-0.45, 1.45, 321)
    rows: list[dict[str, float | bool]] = []
    for d in ds:
        c1 = base.a * (base.rho1 - base.rho2) + base.c2 - d
        p = Parameters(**{**asdict(base), "c1": float(c1)})
        q = quantities(p)
        rows.append(
            {
                "D": d,
                "c1_minus_c2": c1 - base.c2,
                "service_M_minus_Q": q["XM"] - q["XQ"],
                "welfare_M_minus_Q": q["WM"] - q["WQ"],
                "fixed_threshold": q["fixed_threshold"],
                "fixed_domain": q["fixed_domain"],
            }
        )
    write_rows(RESULT_DIR / "fixed_release_sweep.csv", rows)

    eps_grid = np.linspace(0.025, 0.19, 150)
    premium_grid = np.linspace(-0.4, 1.8, 170)
    regime = np.zeros((len(premium_grid), len(eps_grid)), dtype=int)
    for yi, premium in enumerate(premium_grid):
        for xi, eps in enumerate(eps_grid):
            p = Parameters(
                **{
                    **asdict(base),
                    "rho1": base.rho2 + float(eps),
                    "k": 0.22,
                    "c1": base.c2 + float(premium),
                }
            )
            q = quantities(p)
            if not q["fixed_domain"]:
                regime[yi, xi] = 0
            elif q["WM"] >= q["WQ"]:
                regime[yi, xi] = 2
            else:
                regime[yi, xi] = 1

    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.45))
    ax = axes[0]
    ax.axhline(0, color="black", linewidth=0.7)
    ax.plot(ds, [r["service_M_minus_Q"] for r in rows], color=BLUE, linewidth=1.8)
    ax.axvline(0, color=GREY, linestyle="--", linewidth=0.8)
    ax.set(xlabel=r"Service--cost advantage $D$", ylabel=r"$X_M-X_Q$", title="(a) Service effect")

    ax = axes[1]
    welfare = np.array([r["welfare_M_minus_Q"] for r in rows])
    threshold = float(rows[0]["fixed_threshold"])
    ax.axhline(0, color="black", linewidth=0.7)
    ax.plot(ds, welfare, color=PURPLE, linewidth=1.8)
    ax.axvline(0, color=GREY, linestyle="--", linewidth=0.8)
    ax.axvline(threshold, color=ORANGE, linestyle="--", linewidth=1.1)
    ax.text(threshold + 0.035, min(welfare) * 0.82, "welfare\nthreshold", color=ORANGE, fontsize=7)
    ax.set(xlabel=r"Service--cost advantage $D$", ylabel=r"$W_M-W_Q$", title="(b) Welfare effect")

    ax = axes[2]
    cmap = mcolors.ListedColormap([LIGHT_GREY, "#f3c48d", "#91bad6"])
    ax.imshow(
        regime,
        origin="lower",
        aspect="auto",
        extent=[eps_grid.min(), eps_grid.max(), premium_grid.min(), premium_grid.max()],
        cmap=cmap,
        vmin=0,
        vmax=2,
        interpolation="nearest",
    )
    ax.plot(eps_grid, base.a * eps_grid, color="white", linewidth=0.9, linestyle=":", label=r"$D=0$")
    ax.set(
        xlabel=r"Productivity gap $\varepsilon$",
        ylabel=r"Cost premium $c_1-c_2$",
        title="(c) Policy regions",
    )
    handles = [
        plt.Line2D([0], [0], color="#91bad6", lw=6, label=r"$M$ higher welfare"),
        plt.Line2D([0], [0], color="#f3c48d", lw=6, label=r"$Q$ higher welfare"),
        plt.Line2D([0], [0], color=LIGHT_GREY, lw=6, label="outside domain"),
    ]
    ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=6.5)
    fig.tight_layout(w_pad=1.2)
    save_pdf(fig, "fig_policy_regions.pdf")
    plt.close(fig)
    return {"fixed_threshold_baseline": threshold}


def mechanism_comparison() -> None:
    base = Parameters()
    ds = np.linspace(-0.35, 1.4, 300)
    rows: list[dict[str, float | bool]] = []
    for d in ds:
        c1 = base.a * (base.rho1 - base.rho2) + base.c2 - d
        p = Parameters(**{**asdict(base), "c1": float(c1)})
        q = quantities(p)
        rows.append(
            {
                "D": d,
                "share_Q": q["xQ"] / p.S,
                "share_M": q["xM"] / p.S,
                "share_T": q["xT"] / p.S,
                "share_W": q["xW"] / p.S,
                "loss_Q": q["WW"] - q["WQ"],
                "loss_M": q["WW"] - q["WM"],
                "loss_T": q["WW"] - q["WT"],
                "comparison_domain": q["comparison_domain"],
            }
        )
    write_rows(RESULT_DIR / "mechanism_comparison_sweep.csv", rows)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.65))
    ax = axes[0]
    ax.plot(ds, [r["share_W"] for r in rows], color=GREEN, lw=2, label=r"Efficient $W$")
    ax.plot(ds, [r["share_M"] for r in rows], color=BLUE, lw=1.7, label=r"Auction $M$")
    ax.plot(ds, [r["share_T"] for r in rows], color=PURPLE, lw=1.5, label=r"Unshifted bids $T$")
    ax.plot(ds, [r["share_Q"] for r in rows], color=ORANGE, lw=1.5, linestyle="--", label=r"Equal reservations $Q$")
    ax.set(
        xlabel=r"Service--cost advantage $D$",
        ylabel="Operator 1 resource share",
        title="(a) Allocation paths",
        ylim=(0, 1),
    )
    ax.legend(frameon=False, ncol=2, loc="upper left")

    ax = axes[1]
    ax.plot(ds, [r["loss_Q"] for r in rows], color=ORANGE, lw=1.7, linestyle="--", label=r"$W_W-W_Q$")
    ax.plot(ds, [r["loss_M"] for r in rows], color=BLUE, lw=1.7, label=r"$W_W-W_M$")
    ax.plot(ds, [r["loss_T"] for r in rows], color=PURPLE, lw=1.5, label=r"$W_W-W_T$")
    ax.set(
        xlabel=r"Service--cost advantage $D$",
        ylabel="Welfare loss from efficient split",
        title="(b) Distance from the allocation target",
    )
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    fig.tight_layout(w_pad=1.3)
    save_pdf(fig, "fig_mechanism_comparison.pdf")
    plt.close(fig)


def investment_sweep() -> dict[str, float]:
    base = Parameters()
    ds = np.linspace(-0.7, 1.7, 360)
    rows: list[dict[str, float | bool]] = []
    for d in ds:
        c1 = base.a * (base.rho1 - base.rho2) + base.c2 - d
        p = Parameters(**{**asdict(base), "c1": float(c1)})
        inv = investment(p)
        rows.append({"D": d, **inv})
    write_rows(RESULT_DIR / "investment_sweep.csv", rows)

    fig, axes = plt.subplots(1, 3, figsize=(7.35, 2.45))
    ax = axes[0]
    ax.plot(ds, [r["KQ"] for r in rows], color=ORANGE, linestyle="--", lw=1.7, label=r"$K_Q$")
    ax.plot(ds, [r["KM"] for r in rows], color=BLUE, lw=1.7, label=r"$K_M$")
    ax.set(xlabel=r"Service--cost advantage $D$", ylabel="Installed capacity", title="(a) Capacity choice")
    ax.legend(frameon=False)

    ax = axes[1]
    ax.plot(ds, [r["XQ_own"] for r in rows], color=ORANGE, linestyle="--", lw=1.7, label=r"$X_Q(K_Q)$")
    ax.plot(ds, [r["XM_own"] for r in rows], color=BLUE, lw=1.7, label=r"$X_M(K_M)$")
    ax.set(xlabel=r"Service--cost advantage $D$", ylabel="Delivered service", title="(b) Service at own capacity")
    ax.legend(frameon=False)

    ax = axes[2]
    gap = np.array([r["net_value_gap"] for r in rows], dtype=float)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.plot(ds, gap, color=GREEN, lw=1.8)
    ax.fill_between(ds, 0, gap, where=gap >= 0, color="#91c7a8", alpha=0.45)
    ax.fill_between(ds, 0, gap, where=gap < 0, color="#f3c48d", alpha=0.5)
    ax.set(xlabel=r"Service--cost advantage $D$", ylabel=r"$\mathcal{V}_M-\mathcal{V}_Q$", title="(c) Optimized net value")
    fig.tight_layout(w_pad=1.15)
    save_pdf(fig, "fig_investment_substitution.pdf")
    plt.close(fig)

    # Linear interpolation of the positive crossing is used only for reporting.
    positive_crossings = []
    for left, right in zip(rows[:-1], rows[1:]):
        if left["D"] > 0 and left["net_value_gap"] * right["net_value_gap"] <= 0:
            x0, x1 = left["D"], right["D"]
            y0, y1 = left["net_value_gap"], right["net_value_gap"]
            positive_crossings.append(x0 - y0 * (x1 - x0) / (y1 - y0))
    return {"positive_net_value_crossing": positive_crossings[0] if positive_crossings else float("nan")}


def release_for_state(p: Parameters) -> dict[str, float | bool]:
    eps = p.rho1 - p.rho2
    rb = (p.rho1 + p.rho2) / 2
    d = p.a * eps + p.c2 - p.c1
    f = 2 * p.k + p.b * eps**2
    g = p.b * eps * (p.rho1 + p.rho2)
    z = p.b * rb**2 + p.k
    zeta = g * d / (8 * f)
    aq = p.a * rb - (p.c1 + p.c2) / 2
    am = aq - zeta

    def optimum(intercept: float) -> tuple[float, float]:
        # F_S = intercept - z S - 2 B S/K - 3 gamma S^2/(2K^2).
        coeff = [1.5 * p.gamma / p.K**2, z + 2 * p.B / p.K, -intercept]
        if abs(coeff[0]) < 1e-12:
            unconstrained = max(0.0, intercept / coeff[1])
        else:
            roots = np.roots(coeff)
            feasible = [float(r.real) for r in roots if abs(r.imag) < 1e-9 and r.real >= 0]
            unconstrained = max(feasible) if feasible else 0.0
        release = min(p.K, unconstrained)
        derivative_at_release = (
            intercept
            - z * release
            - 2 * p.B * release / p.K
            - 1.5 * p.gamma * release**2 / p.K**2
        )
        return release, max(0.0, derivative_at_release) if release >= p.K - 1e-9 else 0.0

    sq, muq = optimum(aq)
    sm, mum = optimum(am)
    pm = Parameters(**{**asdict(p), "S": sm})
    congestion_relief = p.B * sm**2 / p.K**2 + p.gamma * sm**3 / p.K**3
    return {
        "SQ": sq,
        "SM": sm,
        "muQ": muq,
        "muM": mum,
        "congestion_relief_M": congestion_relief,
        "M_domain": quantities(pm)["fixed_domain"],
    }


def state_release() -> dict[str, float]:
    base = Parameters(
        a=8.0,
        b=0.75,
        k=0.16,
        rho1=1.16,
        rho2=1.0,
        c1=3.0,
        c2=2.3,
        B=2.2,
        gamma=2.2,
        S=2.8,
        K=5.5,
        R=1.5,
    )
    demand = np.linspace(8, 17, 241)
    rows: list[dict[str, float | bool]] = []
    for a in demand:
        p = Parameters(**{**asdict(base), "a": float(a)})
        rows.append({"a": a, **release_for_state(p)})
    write_rows(RESULT_DIR / "state_release_sweep.csv", rows)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.65))
    ax = axes[0]
    ax.axhline(base.K, color=GREY, linestyle=":", lw=1.0, label=r"Installed $K$")
    ax.plot(demand, [r["SQ"] for r in rows], color=ORANGE, linestyle="--", lw=1.7, label=r"$S_Q(a)$")
    ax.plot(demand, [r["SM"] for r in rows], color=BLUE, lw=1.8, label=r"$S_M(a)$")
    ax.fill_between(demand, [r["SM"] for r in rows], base.K, color="#c9dcec", alpha=0.4, label="idle installed capacity")
    ax.set(xlabel=r"Demand intercept $a_\omega$", ylabel="Released resource", title="(a) State-contingent release")
    ax.legend(frameon=False, loc="lower right")

    ax = axes[1]
    relief = np.array([r["congestion_relief_M"] for r in rows])
    scarcity = np.array([r["muM"] for r in rows])
    ax.stackplot(demand, relief, scarcity, colors=["#91c7a8", "#9f8ac4"], labels=["congestion relief", "binding-cap scarcity value"], alpha=0.9)
    ax.set(xlabel=r"Demand intercept $a_\omega$", ylabel="Marginal value of capacity", title="(b) Two investment margins")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout(w_pad=1.4)
    save_pdf(fig, "fig_state_release.pdf")
    plt.close(fig)

    first_binding = next(r["a"] for r in rows if r["SM"] >= base.K - 1e-8)
    assert all(bool(r["M_domain"]) for r in rows)
    return {"state_release_first_binding_a": float(first_binding)}


def design_scan() -> dict[str, float | int | dict[str, list[float]]]:
    rng = np.random.default_rng(RANDOM_SEED)
    n = DESIGN_DRAWS
    ranges = {
        "a": [8.0, 12.0],
        "b": [0.7, 1.3],
        "k": [0.16, 0.40],
        "epsilon": [0.03, 0.18],
        "cost_premium": [-0.3, 1.5],
        "B": [0.05, 0.25],
        "gamma": [0.0, 0.30],
        "S": [2.2, 3.8],
        "R": [1.0, 2.5],
    }
    draws = {key: rng.uniform(low, high, n) for key, (low, high) in ranges.items()}
    rows: list[dict[str, float | bool]] = []
    for idx in range(n):
        p = Parameters(
            a=float(draws["a"][idx]),
            b=float(draws["b"][idx]),
            k=float(draws["k"][idx]),
            rho1=1.0 + float(draws["epsilon"][idx]),
            rho2=1.0,
            c1=2.0 + float(draws["cost_premium"][idx]),
            c2=2.0,
            B=float(draws["B"][idx]),
            gamma=float(draws["gamma"][idx]),
            S=float(draws["S"][idx]),
            K=float(draws["S"][idx]),
            R=float(draws["R"][idx]),
        )
        q = quantities(p)
        inv = investment(p)
        rows.append(
            {
                "D": q["D"],
                "epsilon": q["epsilon"],
                "cost_premium": p.c1 - p.c2,
                "fixed_domain": q["fixed_domain"],
                "investment_domain": inv["feasible"],
                "service_gap_fixed": q["XM"] - q["XQ"],
                "welfare_gap_fixed": q["WM"] - q["WQ"],
                "fixed_threshold": q["fixed_threshold"],
                "capacity_reduction": inv["KQ"] - inv["KM"],
                "service_gap_own_capacity": inv["service_gap"],
                "net_value_gap": inv["net_value_gap"],
                "net_threshold": inv["net_threshold"],
            }
        )
    write_rows(RESULT_DIR / "design_space_scan.csv", rows)

    fixed = [r for r in rows if r["fixed_domain"] and r["D"] > 0]
    invest = [r for r in rows if r["investment_domain"] and r["D"] > 0]
    assert all(r["service_gap_fixed"] > 0 for r in fixed)
    assert all(r["capacity_reduction"] > 0 and r["service_gap_own_capacity"] > 0 for r in invest)

    summary: dict[str, float | int | dict[str, list[float]]] = {
        "seed": RANDOM_SEED,
        "draws": n,
        "ranges": ranges,
        "fixed_domain_positive_D": len(fixed),
        "fixed_domain_positive_D_share_of_draws": len(fixed) / n,
        "auction_higher_fixed_welfare_share": sum(r["welfare_gap_fixed"] > 0 for r in fixed) / len(fixed),
        "quota_higher_fixed_welfare_share": sum(r["welfare_gap_fixed"] < 0 for r in fixed) / len(fixed),
        "investment_domain_positive_D": len(invest),
        "investment_domain_positive_D_share_of_draws": len(invest) / n,
        "auction_higher_optimized_net_value_share": sum(r["net_value_gap"] > 0 for r in invest) / len(invest),
        "quota_higher_optimized_net_value_share": sum(r["net_value_gap"] < 0 for r in invest) / len(invest),
    }
    (RESULT_DIR / "design_space_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    # Plot a deterministic subsample for legibility while retaining every draw in CSV.
    fixed_plot = fixed[:: max(1, len(fixed) // 3500)]
    invest_plot = invest[:: max(1, len(invest) // 3500)]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    ax = axes[0]
    colors = np.where(np.array([r["welfare_gap_fixed"] for r in fixed_plot]) > 0, BLUE, ORANGE)
    ax.scatter(
        [r["service_gap_fixed"] for r in fixed_plot],
        [r["welfare_gap_fixed"] for r in fixed_plot],
        c=colors,
        s=7,
        alpha=0.35,
        linewidths=0,
    )
    ax.axhline(0, color="black", lw=0.7)
    ax.set(xlabel=r"Fixed-release service gain $X_M-X_Q$", ylabel=r"Fixed-release welfare gap $W_M-W_Q$", title="(a) Allocation trade-off")

    ax = axes[1]
    colors = np.where(np.array([r["net_value_gap"] for r in invest_plot]) > 0, GREEN, ORANGE)
    ax.scatter(
        [r["capacity_reduction"] for r in invest_plot],
        [r["net_value_gap"] for r in invest_plot],
        c=colors,
        s=7,
        alpha=0.35,
        linewidths=0,
    )
    ax.axhline(0, color="black", lw=0.7)
    ax.set(xlabel=r"Capacity reduction $K_Q-K_M$", ylabel=r"Optimized net-value gap $\mathcal{V}_M-\mathcal{V}_Q$", title="(b) Capital substitution")
    fig.tight_layout(w_pad=1.35)
    save_pdf(fig, "fig_design_space_scan.pdf")
    plt.close(fig)
    return summary


def main() -> None:
    summary: dict[str, object] = {
        "baseline": asdict(Parameters()),
        **policy_regions(),
        **investment_sweep(),
        **state_release(),
    }
    mechanism_comparison()
    summary["design_space"] = design_scan()
    (RESULT_DIR / "numerical_experiment_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
