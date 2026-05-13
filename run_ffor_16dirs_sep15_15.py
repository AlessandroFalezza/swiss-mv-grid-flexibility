"""
run_ffor_16dirs_sep15_15.py
===========================

Standalone runner for T=1 FFOR with all DERs, evaluated at:

    September 15, 15:00
    16 fixed directions
    angular step = 22.5 degrees
    start angle = 0 degrees

This script DOES NOT save any output files.
It prints the 16 optimized boundary points and visualizes the FFOR with plt.show().

Requirements:
    - config.py
    - grid_loader_mveq.py
    - linear_grid_model_fixed.py
    - ffor_t1_all_ders.py

Assumptions:
    - PF_PV = 0.95
    - PF_HP = 0.95
    - SOC_min = 0.40
    - SOC_max = 0.80
    - SOC_init = 0.60
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from config import GEO_MV, GRID_ID
from grid_loader_mveq import Grid20Loader
from linear_grid_model_fixed import LinearGridModel

from ffor_t1_all_ders import (
    S_BASE_MVA,
    T1DERData,
    baseline_from_loader,
    build_t1_lp,
    describe_devices,
    ffor_area,
    solve_direction_t1,
)


DATA_YEAR = 2030
T_COL = "09-15 15:00:00"

N_DIRS = 16
START_DEG = 0.0

PF_PV = 0.95
PF_HP = 0.95

SOC_MIN = 0.40
SOC_MAX = 0.80
SOC_INIT = 0.60

U_MIN = 0.95
U_MAX = 1.05

N_POLY_LINE = 16
N_POLY_BESS = 16


def sort_ccw(vertices: np.ndarray) -> np.ndarray:
    """Sort 2D vertices counter-clockwise around centroid."""
    if len(vertices) < 3:
        return vertices
    c = vertices.mean(axis=0)
    angles = np.arctan2(vertices[:, 1] - c[1], vertices[:, 0] - c[0])
    return vertices[np.argsort(angles)]


def compute_fixed_direction_points(lp, n_dirs: int = 16, start_deg: float = 0.0):
    """
    Solve fixed-direction OPFs.

    For angle theta:
        maximize cos(theta)*dP_PCC + sin(theta)*dQ_PCC
    """
    vertices = []
    results = []

    print("\n=== FFOR fixed directions ===")
    print(f"{'angle':>8} | {'alpha':>9} | {'beta':>9} | {'dP [MW]':>12} | {'dQ [MVAr]':>12}")
    print("-" * 66)

    for k in range(n_dirs):
        deg = start_deg + k * 360.0 / n_dirs
        rad = np.deg2rad(deg)

        alpha = float(np.cos(rad))
        beta = float(np.sin(rad))

        v, z, obj = solve_direction_t1(alpha, beta, lp)

        if v is None:
            print(f"{deg:8.1f} | infeasible")
            continue

        vertices.append(v)
        results.append({
            "angle_deg": deg,
            "alpha": alpha,
            "beta": beta,
            "vertex": v,
            "dispatch": z,
            "objective": obj,
        })

        print(
            f"{deg:8.1f} | "
            f"{alpha:+9.5f} | "
            f"{beta:+9.5f} | "
            f"{v[0] * S_BASE_MVA:+12.5f} | "
            f"{v[1] * S_BASE_MVA:+12.5f}"
        )

    if not vertices:
        raise RuntimeError("No feasible direction found. Check LP/device/network constraints.")

    vertices = np.array(vertices)
    vertices_sorted = sort_ccw(vertices)
    return vertices_sorted, results


def plot_ffor(vertices: np.ndarray, results: list, area: float, title_time: str):
    """Visualize FFOR polygon without saving a file."""
    verts = vertices * S_BASE_MVA
    verts_closed = np.vstack([verts, verts[0]])

    plt.figure(figsize=(8, 8))

    plt.fill(
        verts_closed[:, 0],
        verts_closed[:, 1],
        alpha=0.25,
        label="FFOR - 16 fixed directions",
    )

    plt.plot(
        verts_closed[:, 0],
        verts_closed[:, 1],
        linewidth=2,
    )

    plt.scatter(
        verts[:, 0],
        verts[:, 1],
        s=55,
        zorder=3,
    )

    for res in results:
        v = res["vertex"] * S_BASE_MVA
        deg = res["angle_deg"]
        plt.text(
            v[0],
            v[1],
            f"{deg:.1f} deg",
            fontsize=8,
            ha="center",
            va="bottom",
        )

    plt.axhline(0, linestyle="--", linewidth=0.8)
    plt.axvline(0, linestyle="--", linewidth=0.8)

    plt.xlabel(r"$\Delta P_{PCC}^{flex}$ [MW], Hug positive flexibility")
    plt.ylabel(r"$\Delta Q_{PCC}^{flex}$ [MVAr], Hug convention")

    plt.title(
        f"FFOR - T=1, all DERs\n"
        f"{title_time}, 16 directions, area = {area * S_BASE_MVA**2:.5f} MW*MVAr"
    )

    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.show()


def print_dispatch_for_extremes(lp, results):
    """
    Print nonzero device dispatches for four useful extremes:
        max dP, min dP, max dQ, min dQ.
    """
    if not results:
        return

    labels_and_indices = [
        ("max dP", int(np.argmax([r["vertex"][0] for r in results]))),
        ("min dP", int(np.argmin([r["vertex"][0] for r in results]))),
        ("max dQ", int(np.argmax([r["vertex"][1] for r in results]))),
        ("min dQ", int(np.argmin([r["vertex"][1] for r in results]))),
    ]

    print("\n=== Nonzero dispatches at key vertices ===")
    for label, idx in labels_and_indices:
        res = results[idx]
        v = res["vertex"]
        z = res["dispatch"]

        print(
            f"\n--- {label} "
            f"(angle={res['angle_deg']:.1f} deg, "
            f"dP={v[0]*S_BASE_MVA:+.5f} MW, "
            f"dQ={v[1]*S_BASE_MVA:+.5f} MVAr) ---"
        )

        any_nonzero = False
        for name, val in zip(lp.var_names, z):
            if abs(val) > 1e-6:
                any_nonzero = True
                print(f"{name:35s} {val*S_BASE_MVA:+.6f}")

        if not any_nonzero:
            print("(all device variables approximately zero)")


def main():
    print("=" * 72)
    print("FFOR T=1 - all DERs - 16 fixed directions")
    print("=" * 72)
    print(f"Grid          : {GRID_ID}")
    print(f"Data year     : {DATA_YEAR}")
    print(f"Timestep      : {T_COL}")
    print(f"Directions    : {N_DIRS}, start={START_DEG} deg, step={360/N_DIRS:.1f} deg")
    print(f"PF_PV / PF_HP : {PF_PV:.2f} / {PF_HP:.2f}")
    print(f"SOC range     : [{SOC_MIN:.2f}, {SOC_MAX:.2f}], init={SOC_INIT:.2f}")
    print("=" * 72)

    loader = Grid20Loader(grid_name=GRID_ID, data_year=DATA_YEAR)
    loader.build_all_profiles()

    if getattr(loader, "net_baseline_by_mv_node", None) is None:
        loader.build_net_baseline_by_node()

    model = LinearGridModel.from_geojson(GEO_MV, GRID_ID)

    baseline = baseline_from_loader(loader, model, T_COL)

    print(
        f"\nBaseline PCC at {T_COL}: "
        f"P={baseline['P_PCC'] * S_BASE_MVA:+.5f} MW, "
        f"Q={baseline['Q_PCC'] * S_BASE_MVA:+.5f} MVAr"
    )

    print(
        f"Baseline voltage range: "
        f"Umin={baseline['U'].min():.5f} p.u., "
        f"Umax={baseline['U'].max():.5f} p.u."
    )

    devices = T1DERData.from_loader(
        loader,
        model,
        t=T_COL,
        pf_pv=PF_PV,
        pf_hp=PF_HP,
        soc_min=SOC_MIN,
        soc_max=SOC_MAX,
        soc_init=SOC_INIT,
    )

    print("\n=== DER summary ===")
    print(describe_devices(devices))

    lp = build_t1_lp(
        model,
        baseline,
        devices,
        u_min=U_MIN,
        u_max=U_MAX,
        n_poly_line=N_POLY_LINE,
        n_poly_bess=N_POLY_BESS,
    )

    # PCC sign test.
    # model.K_P maps nodal injection to slack-bus injection response.
    # Hug/Brandle positive flexibility means:
    #   increase generation OR decrease consumption.
    # Therefore, if K_P is negative for +1 kW nodal injection,
    # we flip the PCC objective signs below.
    print("\n=== PCC sign test ===")

    test_node = 133
    flip_pcc_sign = False

    if test_node not in model._fidx:
        print(f"Node {test_node} not found in model.non_pcc, skipping sign test.")
        print("Defaulting to flip PCC sign for Hug/FFOR convention.")
        flip_pcc_sign = True
    else:
        x_node = np.zeros(2 * model.n_free)
        i = model._fidx[test_node]

        # +1 kW injection at node 133.
        # Since S_BASE_MVA = 1, 1 kW = 0.001 MW = 0.001 p.u.
        x_node[i] = 0.001

        dP_pcc = model.K_P @ x_node
        dQ_pcc = model.K_Q @ x_node

        print(
            f"+1 kW injection at node {test_node} gives "
            f"K_P response = {dP_pcc * S_BASE_MVA * 1000:+.3f} kW"
        )
        print(
            f"+1 kW injection at node {test_node} gives "
            f"K_Q response = {dQ_pcc * S_BASE_MVA * 1000:+.3f} kVAr"
        )

        if dP_pcc > 0:
            print("Interpretation: K_P already has the Hug/FFOR sign.")
            flip_pcc_sign = False
        else:
            print("Interpretation: K_P is slack-injection sign. Flipping P and Q for FFOR.")
            flip_pcc_sign = True

    if flip_pcc_sign:
        lp.kP_dev = -lp.kP_dev
        lp.kQ_dev = -lp.kQ_dev
        print("Applied: lp.kP_dev = -lp.kP_dev and lp.kQ_dev = -lp.kQ_dev")
    else:
        print("Applied: no PCC sign flip")

    print("\n=== LP summary ===")
    print(f"Variables       : {len(lp.var_names)}")
    print(f"Inequalities    : {lp.A_ub.shape[0]}")
    print(f"Voltage limits  : [{U_MIN}, {U_MAX}] p.u.")
    print(f"Line polygon    : {N_POLY_LINE} sides")
    print(f"BESS polygon    : {N_POLY_BESS} sides")

    vertices, results = compute_fixed_direction_points(
        lp,
        n_dirs=N_DIRS,
        start_deg=START_DEG,
    )

    area = ffor_area(vertices)

    print("\n=== Ordered polygon vertices [MW, MVAr] ===")
    for p, q in vertices * S_BASE_MVA:
        print(f"{p:+.6f}, {q:+.6f}")

    print(f"\nArea from 16-direction polygon: {area * S_BASE_MVA**2:.6f} MW*MVAr")

    print_dispatch_for_extremes(lp, results)

    plot_ffor(vertices, results, area, T_COL)


if __name__ == "__main__":
    main()