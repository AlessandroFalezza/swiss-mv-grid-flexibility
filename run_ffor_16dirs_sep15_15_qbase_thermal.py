"""
run_ffor_16dirs_sep15_15_qbase_thermal.py
=========================================

Standalone runner for T=1 FFOR with all DERs, evaluated at:

    September 15, 15:00
    16 fixed support directions
    angular step = 22.5 degrees
    start angle = 0 degrees

This version includes:
    1. Q baseline with pf_load = 0.95 and pf_hp = 0.95
    2. PV pf = 0.95
    3. HP pf = 0.95
    4. SOC_min = 0.40, SOC_max = 0.80, SOC_init = 0.60
    5. One-step HP thermal comfort constraint, in the same spirit as your
       colleague's notebook:
           T_next = T_set - (p_hp_flex_inj / P_HP_base) * q_heat * dt
           T_min <= T_next <= T_max

Important:
    This is still T=1.
    It does not save any output files.
    It visualizes the FFOR with plt.show().

Requirements:
    - config.py
    - grid_loader_mveq.py
    - linear_grid_model_fixed.py
    - ffor_t1_all_ders.py
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import GEO_MV, GRID_ID
try:
    from config import ZAPPAROLI
except Exception:
    ZAPPAROLI = None

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


# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------

DATA_YEAR = 2030
T_COL = "12-15 15:00:00"

N_DIRS = 16
START_DEG = 0.0

PF_LOAD = 0.95
PF_PV = 0.95
PF_HP = 0.95

SOC_MIN = 0.40
SOC_MAX = 0.80
SOC_INIT = 0.60

U_MIN = 0.95
U_MAX = 1.05

N_POLY_LINE = 16
N_POLY_BESS = 16

# HP thermal comfort
T_SET_C = 20.0
T_MIN_C = 19.0
T_MAX_C = 21.0
DT_H = 1.0

# If True, HP nodes with no valid P_base/q_heat thermal data are forced to no flex.
# To mimic your colleague's notebook more closely, keep this False.
FORCE_ZERO_HP_IF_NO_THERMAL_DATA = False


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------

def tan_from_pf(pf: float) -> float:
    return float(np.tan(np.arccos(pf)))


def sort_ccw(vertices: np.ndarray) -> np.ndarray:
    """Sort 2D vertices counter-clockwise around centroid."""
    if len(vertices) < 3:
        return vertices
    c = vertices.mean(axis=0)
    angles = np.arctan2(vertices[:, 1] - c[1], vertices[:, 0] - c[0])
    return vertices[np.argsort(angles)]


def as_node_indexed(df: pd.DataFrame, node_col: str = "MV_osmid") -> pd.DataFrame:
    out = df.copy()
    if node_col in out.columns:
        out[node_col] = out[node_col].astype(int)
        out = out.set_index(node_col)
    else:
        out.index = out.index.astype(int)
    return out


def resolve_hp_year_dir() -> Path:
    """
    Resolve the directory containing HP CSVs for DATA_YEAR.
    Tries both:
        03_HP/<year>
        <ZAPPAROLI>/03_HP/<year>
    """
    candidates = [Path("03_HP") / str(DATA_YEAR)]

    if ZAPPAROLI is not None:
        candidates.append(Path(ZAPPAROLI) / "03_HP" / str(DATA_YEAR))

    # Also try parent of current file, useful when running from project folder.
    here = Path(__file__).resolve().parent
    candidates.append(here / "03_HP" / str(DATA_YEAR))

    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError(
        "Could not find HP data directory. Tried:\n"
        + "\n".join(str(c) for c in candidates)
    )


# -----------------------------------------------------------------------------
# Q baseline
# -----------------------------------------------------------------------------

def build_q_baseline_from_loader(
    loader,
    t_col: str,
    pf_load: float = PF_LOAD,
    pf_hp: float = PF_HP,
) -> dict[int, float]:
    """
    Build Q baseline in consumption-positive convention, in MVAr.

    Q_base_cons =
        P_demand_base * tan(acos(pf_load))
      + P_HP_base     * tan(acos(pf_hp))
      + Q_PV_base

    with Q_PV_base = 0 because PV baseline is unity power factor in the paper.

    Returns:
        dict {MV_osmid: Q_cons_MVAr}
    """
    m_load = tan_from_pf(pf_load)
    m_hp = tan_from_pf(pf_hp)

    demand_df = getattr(loader, "baseline_demand_by_mv_node", None)
    hp_df = getattr(loader, "hp_baseline_by_mv_node", None)

    if demand_df is None:
        raise AttributeError("loader.baseline_demand_by_mv_node is missing.")
    if hp_df is None:
        raise AttributeError("loader.hp_baseline_by_mv_node is missing.")

    demand = as_node_indexed(demand_df)
    hp = as_node_indexed(hp_df)

    if t_col not in demand.columns:
        raise ValueError(f"Demand baseline does not contain time column {t_col!r}")
    if t_col not in hp.columns:
        raise ValueError(f"HP baseline does not contain time column {t_col!r}")

    all_nodes = demand.index.union(hp.index)
    q_net_mvar = {}

    for node in all_nodes:
        p_dem_kw = float(demand.loc[node, t_col]) if node in demand.index else 0.0
        p_hp_kw = float(hp.loc[node, t_col]) if node in hp.index else 0.0

        q_cons_kvar = p_dem_kw * m_load + p_hp_kw * m_hp
        q_net_mvar[int(node)] = q_cons_kvar / 1000.0

    return q_net_mvar


# -----------------------------------------------------------------------------
# HP thermal data
# -----------------------------------------------------------------------------

def build_hp_thermal_data_by_node(loader, model, t_col: str) -> dict[int, dict]:
    """
    Reconstruct HP one-step thermal data aggregated to MV-equivalent nodes.

    Follows the same logic as your colleague's notebook:
        q_heat = Thermal_conductivity_kW/K * max(T_set - T_out, 0) / Thermal_capacitance_KWh/K

    For each individual HP allocation row:
        p_base = loader._hp_consumption_for_row(row, temperature_profiles)[t_col]
        q_heat_K_per_h = ...
    Then aggregate to MV node.

    Since our T=1 LP has one aggregate HP variable per MV node, we use a
    P_base-weighted average q_heat per node:
        q_eff = sum(p_base_i * q_heat_i) / sum(p_base_i)

    Returns:
        {node: {"p_base_kw": ..., "q_heat_K_per_h": ..., "n_assets": ...}}
    """
    hp_year_dir = resolve_hp_year_dir()

    if not hasattr(loader, "_load_temperature_profiles"):
        raise AttributeError("loader has no _load_temperature_profiles method.")
    if not hasattr(loader, "_hp_consumption_for_row"):
        raise AttributeError("loader has no _hp_consumption_for_row method.")

    temperature_profiles = loader._load_temperature_profiles(hp_year_dir)

    connected_lv = set()
    if hasattr(loader, "mv_to_lv"):
        connected_lv = {str(v) for v in loader.mv_to_lv.values() if str(v) != "-1"}

    lv_to_mv = getattr(loader, "lv_to_mv", {})

    # Accumulators per MV node.
    pbase_sum = {}
    q_weighted_sum = {}
    count = {}

    files = [
        (hp_year_dir / "MV_heat_pump_allocation.csv", "MV"),
        (hp_year_dir / "LV_heat_pump_allocation.csv", "LV"),
    ]

    for path, source in files:
        if not path.exists():
            continue

        hp = pd.read_csv(path)

        if source == "MV":
            if "MV_grid" not in hp.columns:
                continue
            hp = hp[hp["MV_grid"].astype(str).eq(str(GRID_ID))].copy()
        else:
            if "LV_grid" not in hp.columns:
                continue
            hp["LV_grid"] = hp["LV_grid"].astype(str)
            hp = hp[hp["LV_grid"].isin(connected_lv)].copy()

        for _, row in hp.iterrows():
            if source == "MV":
                if "MV_osmid" not in row:
                    continue
                mv_osmid = int(row["MV_osmid"])
            else:
                lv_grid = str(row["LV_grid"])
                if lv_grid not in lv_to_mv:
                    continue
                mv_osmid = int(lv_to_mv[lv_grid])

            if mv_osmid not in model._fidx:
                continue

            try:
                profile = loader._hp_consumption_for_row(row, temperature_profiles)
            except Exception:
                continue

            if t_col not in profile.index:
                continue

            p_base_kw = float(profile[t_col])
            if p_base_kw <= 1e-9:
                # No heating baseline at this time for this HP.
                # It contributes no thermal information.
                continue

            profile_name = row.get("Temperature_profile_name", None)
            if profile_name is not None and profile_name in temperature_profiles.index:
                t_out = float(temperature_profiles.loc[profile_name, t_col])
            else:
                t_out = T_SET_C

            try:
                H = float(row["Thermal_conductivity_kW/K"])
                Cth = float(row["Thermal_capacitance_KWh/K"])
            except Exception:
                continue

            if Cth <= 1e-12:
                continue

            q_heat = H * max(T_SET_C - t_out, 0.0) / Cth  # K/h
            if q_heat <= 1e-12:
                continue

            pbase_sum[mv_osmid] = pbase_sum.get(mv_osmid, 0.0) + p_base_kw
            q_weighted_sum[mv_osmid] = q_weighted_sum.get(mv_osmid, 0.0) + p_base_kw * q_heat
            count[mv_osmid] = count.get(mv_osmid, 0) + 1

    out = {}
    for node, p_sum in pbase_sum.items():
        if p_sum <= 1e-9:
            continue
        out[node] = {
            "p_base_kw": p_sum,
            "q_heat_K_per_h": q_weighted_sum[node] / p_sum,
            "n_assets": count[node],
        }
    return out


def add_hp_thermal_constraints_to_lp(lp, thermal_by_node: dict[int, dict]):
    """
    Add one-step HP thermal constraints to an existing T=1 LP.

    Our HP LP variable is:
        p_hp_inj = delta P_HP_injection
    where:
        p_hp_inj > 0 means HP consumes less
        p_hp_inj < 0 means HP consumes more

    Thermal model:
        T_next = T_set - (p_hp_inj / P_HP_base) * q_heat * DT_H

    Comfort:
        T_MIN <= T_next <= T_MAX

    This gives:
        -coeff * p_hp_inj <= T_MAX - T_SET
         coeff * p_hp_inj <= T_SET - T_MIN

    where:
        coeff = q_heat * DT_H / P_HP_base
    """
    n_var = len(lp.var_names)
    name_to_idx = {name: i for i, name in enumerate(lp.var_names)}

    A_new = []
    b_new = []

    added = 0
    skipped = 0
    forced_zero = 0

    for hp_dev in lp.devices.hp:
        node = int(hp_dev.node)
        var_name = f"hp[{node}].p_inj"

        if var_name not in name_to_idx:
            skipped += 1
            continue

        j = name_to_idx[var_name]

        if node not in thermal_by_node:
            skipped += 1
            if FORCE_ZERO_HP_IF_NO_THERMAL_DATA:
                # Force p_hp_inj = 0 through two inequalities.
                row = np.zeros(n_var)
                row[j] = 1.0
                A_new.append(row)
                b_new.append(0.0)

                row = np.zeros(n_var)
                row[j] = -1.0
                A_new.append(row)
                b_new.append(0.0)
                forced_zero += 1
            continue

        p_base_kw = float(thermal_by_node[node]["p_base_kw"])
        q_heat = float(thermal_by_node[node]["q_heat_K_per_h"])

        p_base_pu = p_base_kw / (S_BASE_MVA * 1000.0)
        if p_base_pu <= 1e-12 or q_heat <= 1e-12:
            skipped += 1
            continue

        coeff = q_heat * DT_H / p_base_pu

        # -coeff * p_hp_inj <= T_MAX - T_SET
        row = np.zeros(n_var)
        row[j] = -coeff
        A_new.append(row)
        b_new.append(T_MAX_C - T_SET_C)

        # coeff * p_hp_inj <= T_SET - T_MIN
        row = np.zeros(n_var)
        row[j] = coeff
        A_new.append(row)
        b_new.append(T_SET_C - T_MIN_C)

        added += 1

    if A_new:
        lp.A_ub = np.vstack([lp.A_ub, np.vstack(A_new)])
        lp.b_ub = np.concatenate([lp.b_ub, np.array(b_new, dtype=float)])

    return {
        "added_nodes": added,
        "skipped_nodes": skipped,
        "forced_zero_nodes": forced_zero,
        "added_inequalities": len(A_new),
    }


# -----------------------------------------------------------------------------
# Direction solve and plotting
# -----------------------------------------------------------------------------

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
        f"FFOR - T=1, all DERs, Q baseline + HP thermal\n"
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


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    print("=" * 78)
    print("FFOR T=1 - all DERs - 16 fixed directions - Q baseline + HP thermal")
    print("=" * 78)
    print(f"Grid          : {GRID_ID}")
    print(f"Data year     : {DATA_YEAR}")
    print(f"Timestep      : {T_COL}")
    print(f"Directions    : {N_DIRS}, start={START_DEG} deg, step={360/N_DIRS:.1f} deg")
    print(f"PF load/PV/HP : {PF_LOAD:.2f} / {PF_PV:.2f} / {PF_HP:.2f}")
    print(f"SOC range     : [{SOC_MIN:.2f}, {SOC_MAX:.2f}], init={SOC_INIT:.2f}")
    print(f"HP comfort    : [{T_MIN_C:.1f}, {T_MAX_C:.1f}] C, T_set={T_SET_C:.1f} C")
    print("=" * 78)

    loader = Grid20Loader(grid_name=GRID_ID, data_year=DATA_YEAR)
    loader.build_all_profiles()

    if getattr(loader, "net_baseline_by_mv_node", None) is None:
        loader.build_net_baseline_by_node()

    model = LinearGridModel.from_geojson(GEO_MV, GRID_ID)

    # Build Q baseline with pf_load=0.95 and pf_hp=0.95.
    q_net_mvar = build_q_baseline_from_loader(
        loader,
        T_COL,
        pf_load=PF_LOAD,
        pf_hp=PF_HP,
    )

    baseline = baseline_from_loader(loader, model, T_COL, q_net_mvar=q_net_mvar)

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

    # Add HP thermal constraints.
    print("\n=== HP thermal constraints ===")
    thermal_by_node = build_hp_thermal_data_by_node(loader, model, T_COL)
    print(f"Thermal data available for {len(thermal_by_node)} MV-equivalent HP nodes.")

    thermal_info = add_hp_thermal_constraints_to_lp(lp, thermal_by_node)
    print(
        f"Added thermal constraints for {thermal_info['added_nodes']} HP nodes "
        f"({thermal_info['added_inequalities']} inequalities). "
        f"Skipped {thermal_info['skipped_nodes']} nodes."
    )
    if thermal_info["forced_zero_nodes"] > 0:
        print(f"Forced zero HP flex for {thermal_info['forced_zero_nodes']} nodes.")

    # PCC sign test and flip to Hug/FFOR convention if needed.
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
