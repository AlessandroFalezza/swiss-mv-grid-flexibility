"""
ffor_t1_all_ders.py
===================

Single-timestep FFOR computation with device-level DER variables.

This module builds a T=1 LP with BESS, PV curtailment and HP flexibility
as separate device-level variables, maps them to nodal injection deviations,
and then maps those deviations to PCC flexibility through the linearized
network model.

Conventions
-----------
All optimization variables use injection-positive convention:
    ΔP_inj > 0  = more injection into the grid
    ΔP_inj < 0  = more absorption from the grid

Default assumptions requested:
    PF_PV   = 0.95
    PF_HP   = 0.95
    SOC_MIN = 0.40
    SOC_MAX = 0.80
    SOC_INIT = 0.60

This file implements T=1 only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linprog

S_BASE_MVA = 1.0
U_MIN = 0.95
U_MAX = 1.05
PF_PV = 0.95
PF_HP = 0.95
SOC_MIN = 0.40
SOC_MAX = 0.80
SOC_INIT = 0.50
H_STEP_H = 1.0
N_POLY_LINE = 16
N_POLY_BESS = 16


def tan_from_pf(pf: float) -> float:
    if not (0 < pf <= 1):
        raise ValueError(f"Power factor must be in (0, 1], got {pf}")
    return float(np.tan(np.arccos(pf)))


def polygon_normals(n_sides: int) -> np.ndarray:
    angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])


def _as_node_indexed(df: pd.DataFrame, node_col: str = "MV_osmid") -> pd.DataFrame:
    out = df.copy()
    if node_col in out.columns:
        out[node_col] = out[node_col].astype(int)
        out = out.set_index(node_col)
    else:
        out.index = out.index.astype(int)
    return out


def _find_col(df: pd.DataFrame, candidates: List[str], contains_all: Optional[List[str]] = None) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    if contains_all:
        for c in df.columns:
            lc = c.lower()
            if all(s.lower() in lc for s in contains_all):
                return c
    return None


def _time_col_from_input(loader: Any, t: str | int) -> str:
    time_cols = list(getattr(loader, "time_columns", []))
    if isinstance(t, int):
        if not time_cols:
            raise ValueError("loader.time_columns is empty; cannot resolve integer timestep.")
        return time_cols[t]
    return t


@dataclass
class BESSDevice:
    node: int
    p_nom_pu: float
    s_max_pu: float
    c_puh: float
    eta_ch: float
    eta_dis: float
    soc_min: float = SOC_MIN
    soc_max: float = SOC_MAX
    soc_init: float = SOC_INIT

    @property
    def p_ch_max_pu(self) -> float:
        energy_bound = (self.soc_max - self.soc_init) * self.c_puh / max(self.eta_ch, 1e-9) / H_STEP_H
        return float(min(self.p_nom_pu, max(0.0, energy_bound)))

    @property
    def p_dis_max_pu(self) -> float:
        energy_bound = (self.soc_init - self.soc_min) * self.c_puh * self.eta_dis / H_STEP_H
        return float(min(self.p_nom_pu, max(0.0, energy_bound)))


@dataclass
class PVDevice:
    node: int
    p_avail_pu: float
    pf: float = PF_PV

    @property
    def q_max_pu(self) -> float:
        return float(self.p_avail_pu * tan_from_pf(self.pf))


@dataclass
class HPDevice:
    node: int
    p_base_pu: float
    p_nom_pu: float
    pf: float = PF_HP

    @property
    def p_inj_lb_pu(self) -> float:
        # HP consumption actual in [0, P_nom]; baseline consumption = P_base.
        # ΔP_inj = -ΔP_consumption, so ΔP_inj in [P_base-P_nom, P_base].
        return float(self.p_base_pu - self.p_nom_pu)

    @property
    def p_inj_ub_pu(self) -> float:
        return float(self.p_base_pu)

    @property
    def q_slope(self) -> float:
        # ΔQ_hp_inj = tan(phi) * ΔP_hp_inj.
        return tan_from_pf(self.pf)


@dataclass
class T1DERData:
    bess: List[BESSDevice] = field(default_factory=list)
    pv: List[PVDevice] = field(default_factory=list)
    hp: List[HPDevice] = field(default_factory=list)

    @classmethod
    def from_loader(
        cls,
        loader: Any,
        model: Any,
        t: str | int,
        pf_pv: float = PF_PV,
        pf_hp: float = PF_HP,
        soc_min: float = SOC_MIN,
        soc_max: float = SOC_MAX,
        soc_init: float = SOC_INIT,
    ) -> "T1DERData":
        t_col = _time_col_from_input(loader, t)
        fidx = model._fidx
        bess_devices: List[BESSDevice] = []
        pv_devices: List[PVDevice] = []
        hp_devices: List[HPDevice] = []

        # BESS
        bess_df = getattr(loader, "bess_allocation", None)
        if bess_df is None:
            bess_df = getattr(loader, "bess_allocation_mveq", None)
        if bess_df is not None and len(bess_df) > 0:
            bdf = bess_df.copy()
            if "MV_osmid" in bdf.columns:
                bdf["MV_osmid"] = bdf["MV_osmid"].astype(int)
                bdf = bdf.set_index("MV_osmid")
            else:
                bdf.index = bdf.index.astype(int)
            for node, row in bdf.iterrows():
                node = int(node)
                if node not in fidx:
                    continue
                p_nom_kw = float(row.get("Nominal_power_kW", row.get("nominal_power_kW", 0.0)))
                c_kwh = float(row.get("Battery_capacity_kWh", row.get("battery_capacity_kWh", 0.0)))
                eta_ch = float(row.get("Charging_efficiency", row.get("eta_ch", 0.9219544457292888)))
                eta_dis = float(row.get("Discharging_efficiency", row.get("eta_dis", eta_ch)))
                if p_nom_kw <= 0 or c_kwh <= 0:
                    continue
                p_nom_pu = p_nom_kw / (S_BASE_MVA * 1000.0)
                c_puh = c_kwh / (S_BASE_MVA * 1000.0)
                bess_devices.append(BESSDevice(
                    node=node,
                    p_nom_pu=p_nom_pu,
                    s_max_pu=p_nom_pu,
                    c_puh=c_puh,
                    eta_ch=eta_ch,
                    eta_dis=eta_dis,
                    soc_min=soc_min,
                    soc_max=soc_max,
                    soc_init=soc_init,
                ))

        # PV
        pv_df = getattr(loader, "pv_generation_by_mv_node", None)
        if pv_df is not None and len(pv_df) > 0:
            pdf = _as_node_indexed(pv_df)
            if t_col not in pdf.columns:
                raise ValueError(f"PV profile does not contain time column {t_col!r}")
            for node, row in pdf.iterrows():
                node = int(node)
                if node not in fidx:
                    continue
                p_kw = float(row[t_col])
                if p_kw <= 1e-9:
                    continue
                pv_devices.append(PVDevice(node=node, p_avail_pu=p_kw / (S_BASE_MVA * 1000.0), pf=pf_pv))

        # HP
        hp_base_df = getattr(loader, "hp_baseline_by_mv_node", None)
        cap_df = getattr(loader, "node_capacity_summary", None)
        if hp_base_df is not None and cap_df is not None and len(hp_base_df) > 0:
            hbase = _as_node_indexed(hp_base_df)
            caps = _as_node_indexed(cap_df)
            if t_col not in hbase.columns:
                raise ValueError(f"HP baseline does not contain time column {t_col!r}")
            hp_nom_col = _find_col(
                caps,
                candidates=[
                    "hp_nominal_power_kW",
                    "HP_nominal_power_kW",
                    "Nominal_power_kW",
                    "hp_p_nom_kW",
                    "hp_p_installed_kW",
                ],
                contains_all=["hp", "nominal"],
            )
            if hp_nom_col is None:
                hp_nom_col = _find_col(caps, candidates=[], contains_all=["hp", "power"])
            if hp_nom_col is not None:
                for node, row in hbase.iterrows():
                    node = int(node)
                    if node not in fidx or node not in caps.index:
                        continue
                    p_base_kw = float(row[t_col])
                    p_nom_kw = float(caps.loc[node, hp_nom_col])
                    if p_nom_kw <= 0:
                        continue
                    hp_devices.append(HPDevice(
                        node=node,
                        p_base_pu=p_base_kw / (S_BASE_MVA * 1000.0),
                        p_nom_pu=p_nom_kw / (S_BASE_MVA * 1000.0),
                        pf=pf_hp,
                    ))
        return cls(bess=bess_devices, pv=pv_devices, hp=hp_devices)


@dataclass
class T1LP:
    A_ub: np.ndarray
    b_ub: np.ndarray
    bounds: List[Tuple[Optional[float], Optional[float]]]
    kP_dev: np.ndarray
    kQ_dev: np.ndarray
    A_node: np.ndarray
    var_names: List[str]
    devices: T1DERData
    baseline: dict
    model: Any


def baseline_from_loader(loader: Any, model: Any, t: str | int, q_net_mvar: Optional[Dict[int, float]] = None) -> dict:
    """Build baseline state from loader.net_baseline_by_mv_node in kW, consumption-positive."""
    t_col = _time_col_from_input(loader, t)
    net_df = getattr(loader, "net_baseline_by_mv_node", None)
    if net_df is None:
        if hasattr(loader, "build_net_baseline_by_node"):
            net_df = loader.build_net_baseline_by_node()
        else:
            raise AttributeError("loader has no net_baseline_by_mv_node and no build_net_baseline_by_node().")
    ndf = _as_node_indexed(net_df)
    if t_col not in ndf.columns:
        raise ValueError(f"net baseline does not contain time column {t_col!r}")
    p_net_mw = {int(node): float(row[t_col]) / 1000.0 for node, row in ndf.iterrows()}
    if hasattr(model, "consumption_baseline_to_pu"):
        P_inj_pu, Q_inj_pu = model.consumption_baseline_to_pu(p_net_mw, q_net_mvar)
    else:
        P_inj_mw = {k: -v for k, v in p_net_mw.items()}
        Q_inj_mvar = None if q_net_mvar is None else {k: -v for k, v in q_net_mvar.items()}
        P_inj_pu, Q_inj_pu = model.baseline_to_pu(P_inj_mw, Q_inj_mvar)
    return model.compute_baseline_state(P_inj_pu, Q_inj_pu)


def build_t1_lp(
    model: Any,
    baseline: dict,
    devices: T1DERData,
    u_min: float = U_MIN,
    u_max: float = U_MAX,
    n_poly_line: int = N_POLY_LINE,
    n_poly_bess: int = N_POLY_BESS,
) -> T1LP:
    """Build a T=1 linear program with all DERs as device-level variables."""
    var_names: List[str] = []
    bounds: List[Tuple[Optional[float], Optional[float]]] = []
    idx: Dict[Tuple[str, int, str], int] = {}

    def add_var(key: Tuple[str, int, str], name: str, lb: Optional[float], ub: Optional[float]) -> int:
        j = len(var_names)
        idx[key] = j
        var_names.append(name)
        bounds.append((lb, ub))
        return j

    for b, dev in enumerate(devices.bess):
        add_var(("bess", b, "p_ch"), f"bess[{dev.node}].p_ch", 0.0, dev.p_ch_max_pu)
        add_var(("bess", b, "p_dis"), f"bess[{dev.node}].p_dis", 0.0, dev.p_dis_max_pu)
        add_var(("bess", b, "q"), f"bess[{dev.node}].q", -dev.s_max_pu, dev.s_max_pu)
    for p, dev in enumerate(devices.pv):
        add_var(("pv", p, "p_curt"), f"pv[{dev.node}].p_curt", 0.0, dev.p_avail_pu)
        add_var(("pv", p, "q"), f"pv[{dev.node}].q", -dev.q_max_pu, dev.q_max_pu)
    for h, dev in enumerate(devices.hp):
        add_var(("hp", h, "p_inj"), f"hp[{dev.node}].p_inj", dev.p_inj_lb_pu, dev.p_inj_ub_pu)

    n_var = len(var_names)
    n_free = model.n_free
    A_node = np.zeros((2 * n_free, n_var))

    def node_i(node: int) -> Optional[int]:
        return model._fidx.get(int(node))

    for b, dev in enumerate(devices.bess):
        i = node_i(dev.node)
        if i is None:
            continue
        A_node[i, idx[("bess", b, "p_ch")]] -= 1.0
        A_node[i, idx[("bess", b, "p_dis")]] += 1.0
        A_node[n_free + i, idx[("bess", b, "q")]] += 1.0

    for p, dev in enumerate(devices.pv):
        i = node_i(dev.node)
        if i is None:
            continue
        A_node[i, idx[("pv", p, "p_curt")]] -= 1.0
        A_node[n_free + i, idx[("pv", p, "q")]] += 1.0

    for h, dev in enumerate(devices.hp):
        i = node_i(dev.node)
        if i is None:
            continue
        j = idx[("hp", h, "p_inj")]
        A_node[i, j] += 1.0
        A_node[n_free + i, j] += dev.q_slope

    kP_dev = model.K_P @ A_node
    kQ_dev = model.K_Q @ A_node

    A_rows: List[np.ndarray] = []
    b_rows: List[float] = []

    # Voltage constraints
    MU_dev = model.M_U @ A_node
    dU_base = baseline["dU"]
    for i in range(n_free):
        A_rows.append(MU_dev[i, :])
        b_rows.append((u_max - 1.0) - dU_base[i])
        A_rows.append(-MU_dev[i, :])
        b_rows.append((1.0 - u_min) + dU_base[i])

    # Line flow constraints, both directions, no double-counting of Q_ref.
    line_normals = polygon_normals(n_poly_line)
    for Fp, Fq, P_base, Q_base in [
        (model.F_P_from, model.F_Q_from, baseline["P_from"], baseline["Q_from"]),
        (model.F_P_to, model.F_Q_to, baseline["P_to"], baseline["Q_to"]),
    ]:
        FP_dev = Fp @ A_node
        FQ_dev = Fq @ A_node
        for l in range(len(model.branches)):
            smax = model.s_nom_pu[l]
            for cphi, sphi in line_normals:
                A_rows.append(cphi * FP_dev[l, :] + sphi * FQ_dev[l, :])
                b_rows.append(float(smax - cphi * P_base[l] - sphi * Q_base[l]))

    # BESS apparent-power polygon.
    bess_normals = polygon_normals(n_poly_bess)
    for b, dev in enumerate(devices.bess):
        j_ch = idx[("bess", b, "p_ch")]
        j_dis = idx[("bess", b, "p_dis")]
        j_q = idx[("bess", b, "q")]
        for cphi, sphi in bess_normals:
            row = np.zeros(n_var)
            row[j_dis] += cphi
            row[j_ch] -= cphi
            row[j_q] += sphi
            A_rows.append(row)
            b_rows.append(dev.s_max_pu)

                # PV device wedge: reactive capability decreases with curtailment.
    #
    # P_pv_actual = P_avail - p_curt
    # |q_pv| <= tan(phi_pv) * P_pv_actual
    #
    # q_pv <= m * (P_avail - p_curt)
    # -q_pv <= m * (P_avail - p_curt)
    #
    # Equivalent linear inequalities:
    # q_pv  + m*p_curt <= m*P_avail
    # -q_pv + m*p_curt <= m*P_avail

    m_pv = tan_from_pf(PF_PV)

    for p, dev in enumerate(devices.pv):
        j_curt = idx[("pv", p, "p_curt")]
        j_q    = idx[("pv", p, "q")]

        # q_pv + m*p_curt <= m*P_avail
        row = np.zeros(n_var)
        row[j_q] = 1.0
        row[j_curt] = m_pv
        A_rows.append(row)
        b_rows.append(m_pv * dev.p_avail_pu)

        # -q_pv + m*p_curt <= m*P_avail
        row = np.zeros(n_var)
        row[j_q] = -1.0
        row[j_curt] = m_pv
        A_rows.append(row)
        b_rows.append(m_pv * dev.p_avail_pu)

    A_ub = np.vstack(A_rows) if A_rows else np.zeros((0, n_var))
    b_ub = np.array(b_rows, dtype=float)
    return T1LP(A_ub=A_ub, b_ub=b_ub, bounds=bounds, kP_dev=kP_dev, kQ_dev=kQ_dev,
                A_node=A_node, var_names=var_names, devices=devices, baseline=baseline, model=model)


def solve_direction_t1(alpha: float, beta: float, lp: T1LP, method: str = "highs") -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
    """Solve max alpha*ΔP_PCC + beta*ΔQ_PCC."""
    c = -(alpha * lp.kP_dev + beta * lp.kQ_dev)
    result = linprog(c, A_ub=lp.A_ub, b_ub=lp.b_ub, bounds=lp.bounds, method=method,
                     options={"presolve": True, "disp": False})
    if result.status != 0:
        return None, None, None
    z = result.x
    dP = float(lp.kP_dev @ z)
    dQ = float(lp.kQ_dev @ z)
    obj = float(alpha * dP + beta * dQ)
    return np.array([dP, dQ]), z, obj


def _sort_ccw(vertices: List[np.ndarray]) -> List[np.ndarray]:
    if len(vertices) < 3:
        return vertices
    arr = np.array(vertices)
    c = arr.mean(axis=0)
    angles = np.arctan2(arr[:, 1] - c[1], arr[:, 0] - c[0])
    return [vertices[i] for i in np.argsort(angles)]


def _already_present(v: np.ndarray, vertices: List[np.ndarray], tol: float = 1e-6) -> bool:
    return any(np.linalg.norm(v - u) <= tol for u in vertices)


def _triangle_area(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))


def _outward_normal(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = b - a
    n = np.array([-d[1], d[0]], dtype=float)
    norm = np.linalg.norm(n)
    if norm <= 1e-12:
        return np.array([1.0, 0.0])
    n /= norm
    mid = 0.5 * (a + b)
    if np.dot(n, mid) < 0:
        n = -n
    return n


def compute_ffor_t1(lp: T1LP, eps_tol: float = 1e-5, max_iter: int = 200,
                    verbose: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Compute T=1 FFOR vertices with QuickFlex."""
    vertices: List[np.ndarray] = []
    dispatches: List[np.ndarray] = []

    def solve_and_add(alpha: float, beta: float):
        v, z, _ = solve_direction_t1(alpha, beta, lp)
        if v is not None and not _already_present(v, vertices):
            vertices.append(v)
            dispatches.append(z)

    # Initial extremes: max P, max Q, min P, min Q
    for a, b in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
        solve_and_add(a, b)

    if len(vertices) < 2:
        raise RuntimeError("FFOR appears empty or infeasible: fewer than 2 initial vertices.")

    vertices = _sort_ccw(vertices)
    edges = [(np.array(vertices[i]), np.array(vertices[(i + 1) % len(vertices)])) for i in range(len(vertices))]

    iteration = 0
    while edges and iteration < max_iter:
        iteration += 1
        vi, vj = edges.pop(0)
        n = _outward_normal(vi, vj)
        v_new, z_new, _ = solve_direction_t1(float(n[0]), float(n[1]), lp)
        if v_new is None:
            if verbose:
                print(f"iter {iteration:03d}: infeasible direction {n}")
            continue
        dA = _triangle_area(vi, vj, v_new)
        if verbose:
            print(f"iter {iteration:03d}: dA={dA:.8f}, v=({v_new[0]:+.5f},{v_new[1]:+.5f})")
        if dA <= eps_tol or _already_present(v_new, vertices):
            continue
        vertices.append(v_new)
        dispatches.append(z_new)
        vertices = _sort_ccw(vertices)
        edges.append((np.array(vi), np.array(v_new)))
        edges.append((np.array(v_new), np.array(vj)))

    vertices_arr = np.array(_sort_ccw(vertices))
    info = {
        "iterations": iteration,
        "n_vertices": len(vertices_arr),
        "dispatches": dispatches,
        "var_names": lp.var_names,
        "area_pu2": ffor_area(vertices_arr),
    }
    return vertices_arr, info


def ffor_area(vertices: np.ndarray) -> float:
    if len(vertices) < 3:
        return 0.0
    x = vertices[:, 0]
    y = vertices[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def describe_devices(devices: T1DERData) -> str:
    lines = []
    lines.append(f"BESS devices: {len(devices.bess)}")
    for d in devices.bess:
        lines.append(
            f"  node {d.node}: Pnom={d.p_nom_pu*S_BASE_MVA:.4f} MW, "
            f"Smax={d.s_max_pu*S_BASE_MVA:.4f} MVA, C={d.c_puh*S_BASE_MVA:.4f} MWh, "
            f"p_ch_max={d.p_ch_max_pu*S_BASE_MVA:.4f} MW, "
            f"p_dis_max={d.p_dis_max_pu*S_BASE_MVA:.4f} MW"
        )
    lines.append(f"PV devices: {len(devices.pv)}")
    total_pv = sum(d.p_avail_pu for d in devices.pv)
    lines.append(f"  total P_avail={total_pv*S_BASE_MVA:.4f} MW, pf={PF_PV}")
    lines.append(f"HP devices: {len(devices.hp)}")
    total_hp_base = sum(d.p_base_pu for d in devices.hp)
    total_hp_nom = sum(d.p_nom_pu for d in devices.hp)
    lines.append(f"  total P_base={total_hp_base*S_BASE_MVA:.4f} MW, "
                 f"total P_nom={total_hp_nom*S_BASE_MVA:.4f} MW, pf={PF_HP}")
    return "\n".join(lines)


def compute_ffor_t1_from_loader(
    loader: Any,
    model: Any,
    t: str | int,
    pf_pv: float = PF_PV,
    pf_hp: float = PF_HP,
    soc_min: float = SOC_MIN,
    soc_max: float = SOC_MAX,
    soc_init: float = SOC_INIT,
    eps_tol: float = 1e-5,
    verbose: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    baseline = baseline_from_loader(loader, model, t)
    devices = T1DERData.from_loader(loader, model, t, pf_pv=pf_pv, pf_hp=pf_hp,
                                    soc_min=soc_min, soc_max=soc_max, soc_init=soc_init)
    if verbose:
        print("=== T=1 DER summary ===")
        print(describe_devices(devices))
        print(f"Baseline PCC: P={baseline['P_PCC']*S_BASE_MVA:+.4f} MW, "
              f"Q={baseline['Q_PCC']*S_BASE_MVA:+.4f} MVAr")
    lp = build_t1_lp(model, baseline, devices)
    vertices, info = compute_ffor_t1(lp, eps_tol=eps_tol, verbose=verbose)
    info.update({"baseline": baseline, "devices": devices, "lp": lp})
    return vertices, info


if __name__ == "__main__":
    from config import GEO_MV, GRID_ID
    from grid_loader_mveq import Grid20Loader
    from linear_grid_model_fixed import LinearGridModel

    t_col = "07-15 12:00:00"
    loader = Grid20Loader(grid_name=GRID_ID, data_year=2030)
    loader.build_all_profiles()
    if getattr(loader, "net_baseline_by_mv_node", None) is None:
        loader.build_net_baseline_by_node()
    model = LinearGridModel.from_geojson(GEO_MV, GRID_ID)
    vertices, info = compute_ffor_t1_from_loader(
        loader, model, t=t_col, pf_pv=0.95, pf_hp=0.95,
        soc_min=0.40, soc_max=0.80, soc_init=0.60,
        eps_tol=1e-5, verbose=True,
    )
    print("\n=== FFOR vertices [MW, MVAr] ===")
    for p, q in vertices * S_BASE_MVA:
        print(f"{p:+.5f}, {q:+.5f}")
    print(f"\nArea: {info['area_pu2'] * S_BASE_MVA**2:.6f} MW·MVAr")


