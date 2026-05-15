"""
unified_ffor_t1_mveq_sbase100.py
=================================

Unified single-file T=1 FFOR runner for grid 20_0, keeping only
``grid_loader_mveq.py`` as an external project module.

Purpose
-------
This script merges the modular pipeline developed around:
    - grid_loader_mveq.py
    - linear_grid_model_fixed.py
    - ffor_t1_all_ders.py
    - run_ffor_16dirs_*.py

with the useful practical features of Pierpollo's notebook:
    - one executable file for the FFOR computation;
    - asset-level PV and HP variables from raw MV/LV Zapparoli files;
    - no artificial HP start-up when HP baseline is zero;
    - one-step HP thermal comfort constraints;
    - structured output folder with boundary points, dispatches, line flows,
      voltage summaries and plots.

Only ``grid_loader_mveq.py`` stays separate, because it is the official data
loader for the MV-equivalent baseline.

Important conventions
---------------------
1. grid_loader_mveq uses consumption-positive convention:
       P_net > 0  means load / absorption from the upstream grid.

2. This linear OPF uses injection-positive variables internally:
       P_inj > 0  means generation / injection into the grid.

3. The reported FFOR boundary is converted to the Hug/Brändle flexibility
   convention by an automatic PCC sign test:
       positive ΔP_flex means more injection or less consumption at the PCC.

4. Per-unit base is intentionally set to:
       S_BASE_MVA = 100 MVA
   as discovered from the MATPOWER / colleague setup.

Main modeled resources
----------------------
Baseline:
    P_base = demand_MVeq + HP_base_MVeq - PV_available_MVeq
    BESS baseline = 0

Flexibility variables:
    - BESS charge/discharge/Q
    - PV curtailment and inverter Q
    - HP active-power flexibility with fixed power factor and one-step thermal
      comfort constraint

EVs are not included in this file. They should be added later as a separate,
explicit scenario:
    baseline + EV_base, then optionally EV_flex.

Typical use
-----------
    python unified_ffor_t1_mveq_sbase100.py

or import and call:
    run_unified_ffor(...)
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linprog

from grid_loader_mveq import Grid20Loader

try:
    from config import GRID_ID, DATA_YEAR, ZAPPAROLI, SWISS_PDGS
except Exception:  # allows syntax checking outside the project folder
    GRID_ID = "20_0"
    DATA_YEAR = 2030
    ZAPPAROLI = "."
    SWISS_PDGS = "."

try:
    from config import GEO_MV
except Exception:
    GEO_MV = Path(SWISS_PDGS) / "grids" / "geojson_data" / "MV"


# =============================================================================
# Global settings
# =============================================================================

GRID_NAME = GRID_ID
YEAR = DATA_YEAR
T_COL = "09-15 15:00:00"

# The discovered common MATPOWER / colleague base.
S_BASE_MVA = 100.0
S_BASE_KW = S_BASE_MVA * 1000.0
S_BASE_KVAR = S_BASE_MVA * 1000.0

V_BASE_KV = 20.0
Z_BASE_OHM = V_BASE_KV**2 / S_BASE_MVA
Y_BASE_S = 1.0 / Z_BASE_OHM

# Baseline and device assumptions.
PF_LOAD = 0.95
PF_PV = 0.95
PF_HP = 0.95

SOC_MIN = 0.40
SOC_MAX = 0.80
SOC_INIT = 0.60
DT_H = 1.0

T_SET_C = 20.0
T_MIN_C = 19.0
T_MAX_C = 21.0

# If False, HP rows with zero baseline consumption are skipped.
# This avoids creating flexibility by artificially starting heating when the
# baseline says the HP is off.
ALLOW_HP_START_IF_BASE_ZERO = False
HP_BASE_EPS_KW = 1e-6

U_MIN = 0.95
U_MAX = 1.05

N_DIRS = 16
START_DEG = 0.0
N_POLY_LINE = 16
N_POLY_BESS = 16

# Q baseline options:
#   "pf"       -> Q_load from fixed PF_LOAD, Q_HP from PF_HP
#   "matpower" -> conventional demand Q/P from 20_0_bus_data.csv if found,
#                 Q_HP from PF_HP
#   "zero"     -> Q baseline = 0
Q_BASE_MODE = "pf"

# Use asset-level PV and HP from raw MV/LV files, while BESS comes from the
# loader's MV-equivalent allocation.
USE_ASSET_LEVEL_PV = True
USE_ASSET_LEVEL_HP = True

# Output behavior.
SAVE_OUTPUTS = True
SHOW_PLOT = False
OUTPUT_ROOT = Path("outputs") / "ffor_unified_sbase100"


# =============================================================================
# Utility functions
# =============================================================================


def tan_from_pf(pf: float) -> float:
    if not (0.0 < pf <= 1.0):
        raise ValueError(f"Power factor must be in (0, 1], got {pf}")
    return float(math.tan(math.acos(pf)))


def polygon_normals(n_sides: int) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, int(n_sides), endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def parse_time_col(value: str) -> datetime:
    raw = str(value).strip()
    if len(raw) == len("01-01 00:00:00"):
        return datetime.strptime(f"2030-{raw}", "%Y-%m-%d %H:%M:%S")
    return datetime.strptime(raw, "%Y-%m-%d %H:%M:%S")


def representative_pv_col(t_col: str) -> str:
    """Map any hourly column to the monthly representative day column."""
    month = str(t_col)[:2]
    hour = str(t_col).split()[1]
    return f"{month}-15 {hour}"


def row_time_value(row: pd.Series, t_col: str, default: float = 0.0) -> float:
    """Return row[t_col], or monthly representative PV value if needed."""
    if t_col in row.index:
        return safe_float(row[t_col], default)
    rep = representative_pv_col(t_col)
    if rep in row.index:
        return safe_float(row[rep], default)
    return default


def as_node_indexed(df: pd.DataFrame, node_col: str = "MV_osmid") -> pd.DataFrame:
    out = df.copy()
    if node_col in out.columns:
        out[node_col] = out[node_col].astype(int)
        out = out.set_index(node_col)
    else:
        out.index = out.index.astype(int)
    return out


def sort_ccw(vertices: np.ndarray) -> np.ndarray:
    if len(vertices) < 3:
        return vertices
    c = vertices.mean(axis=0)
    angles = np.arctan2(vertices[:, 1] - c[1], vertices[:, 0] - c[0])
    return vertices[np.argsort(angles)]


def ffor_area(vertices: np.ndarray) -> float:
    if len(vertices) < 3:
        return 0.0
    x = vertices[:, 0]
    y = vertices[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def year_dir(folder_name: str, year: int = YEAR) -> Path:
    """Resolve Zapparoli year folder robustly."""
    candidates = [
        Path(ZAPPAROLI) / folder_name / str(year),
        Path(folder_name) / str(year),
        Path(__file__).resolve().parent / folder_name / str(year),
    ]
    for c in candidates:
        if c.exists():
            return c
    # Return first candidate for clearer downstream error messages.
    return candidates[0]


def find_existing_file(candidates: Iterable[Path]) -> Optional[Path]:
    for p in candidates:
        if p.exists():
            return p
    return None


# =============================================================================
# LinearGridModel, integrated from linear_grid_model_fixed.py with S_BASE=100 MVA
# =============================================================================


class LinearGridModel:
    """
    Linearized AC power-flow model around flat start.

    Internal convention:
        positive P/Q = injection into the network.
    """

    def __init__(self, nodes_df: pd.DataFrame, edges_df: pd.DataFrame):
        self.nodes_df = nodes_df.copy()
        self.edges_df = edges_df.copy()

        self.nodes_df["osmid"] = self.nodes_df["osmid"].astype(int)
        self.edges_df["u"] = self.edges_df["u"].astype(int)
        self.edges_df["v"] = self.edges_df["v"].astype(int)

        pcc_mask = self.nodes_df["source"].astype(str).str.lower().eq("true")
        if pcc_mask.sum() != 1:
            raise ValueError(f"Expected exactly one source/PCC node, found {pcc_mask.sum()}.")

        self.pcc_id = int(self.nodes_df.loc[pcc_mask, "osmid"].iloc[0])
        self.node_ids = sorted(self.nodes_df["osmid"].astype(int).tolist())
        self.non_pcc = [n for n in self.node_ids if n != self.pcc_id]
        self.n = len(self.node_ids)
        self.n_free = len(self.non_pcc)
        self._idx = {n: i for i, n in enumerate(self.node_ids)}
        self._fidx = {n: i for i, n in enumerate(self.non_pcc)}

        self.branches = self._build_branches()
        J = self._build_jacobian()
        self._invert_and_build_sensitivities(J)
        self._build_branch_flow_matrices()

    @classmethod
    def from_geojson(cls, geo_mv_path: str | Path, grid_id: str) -> "LinearGridModel":
        geo_mv_path = Path(geo_mv_path)

        def load_geojson(path: Path) -> pd.DataFrame:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return pd.DataFrame([feat["properties"] for feat in data["features"]])

        nodes = load_geojson(geo_mv_path / f"{grid_id}_nodes")
        edges = load_geojson(geo_mv_path / f"{grid_id}_edges")
        return cls(nodes, edges)

    def _build_branches(self) -> list[dict]:
        branches = []
        for _, row in self.edges_df.iterrows():
            r_ohm = float(row["r"])
            x_ohm = float(row["x"])
            b_s = float(row.get("b", 0.0) or 0.0)
            s_nom_mva = float(row["s_nom"])

            r_pu = r_ohm / Z_BASE_OHM
            x_pu = x_ohm / Z_BASE_OHM
            b_total_pu = b_s * Z_BASE_OHM
            b_sh_pu = b_total_pu / 2.0
            s_nom_pu = s_nom_mva / S_BASE_MVA

            z2 = r_pu**2 + x_pu**2
            if z2 <= 0.0:
                raise ValueError(f"Zero impedance branch {row['u']} - {row['v']}.")
            g_pu = r_pu / z2
            b_ser_pu = -x_pu / z2

            branches.append({
                "u": int(row["u"]),
                "v": int(row["v"]),
                "r_pu": r_pu,
                "x_pu": x_pu,
                "g_pu": g_pu,
                "b_pu": b_ser_pu,
                "b_sh_pu": b_sh_pu,
                "s_nom_pu": s_nom_pu,
            })

        self.s_nom_pu = np.array([br["s_nom_pu"] for br in branches], dtype=float)

        self.P_ref_free = np.zeros(self.n_free)
        self.Q_ref_free = np.zeros(self.n_free)
        self.P_ref_pcc = 0.0
        self.Q_ref_pcc = 0.0
        for br in branches:
            bsh = br["b_sh_pu"]
            for bus in (br["u"], br["v"]):
                if bus == self.pcc_id:
                    self.Q_ref_pcc -= bsh
                else:
                    self.Q_ref_free[self._fidx[bus]] -= bsh
        return branches

    def _build_jacobian(self) -> np.ndarray:
        n = self.n
        J_Pth = np.zeros((n, n))
        J_PU = np.zeros((n, n))
        J_Qth = np.zeros((n, n))
        J_QU = np.zeros((n, n))

        for br in self.branches:
            i = self._idx[br["u"]]
            k = self._idx[br["v"]]
            g = br["g_pu"]
            b = br["b_pu"]
            bsh = br["b_sh_pu"]

            J_Pth[i, k] = b
            J_Pth[k, i] = b
            J_PU[i, k] = -g
            J_PU[k, i] = -g
            J_Qth[i, k] = g
            J_Qth[k, i] = g
            J_QU[i, k] = b
            J_QU[k, i] = b

            J_Pth[i, i] -= b
            J_Pth[k, k] -= b
            J_PU[i, i] += g
            J_PU[k, k] += g
            J_Qth[i, i] -= g
            J_Qth[k, k] -= g
            J_QU[i, i] -= (b + 2.0 * bsh)
            J_QU[k, k] -= (b + 2.0 * bsh)

        return np.vstack([
            np.hstack([J_Pth, J_PU]),
            np.hstack([J_Qth, J_QU]),
        ])

    def _invert_and_build_sensitivities(self, J: np.ndarray) -> None:
        n = self.n
        n_free = self.n_free
        pcc_i = self._idx[self.pcc_id]

        th_free = [i for i in range(n) if i != pcc_i]
        u_free = [i + n for i in range(n) if i != pcc_i]
        free_idx = th_free + u_free

        J_red = J[np.ix_(free_idx, free_idx)]
        cond = np.linalg.cond(J_red)
        if cond > 1e12:
            raise RuntimeError(f"Reduced Jacobian nearly singular, cond={cond:.2e}.")

        J_red_inv = np.linalg.inv(J_red)
        self.J_red_inv = J_red_inv
        self._cond_J_red = cond

        self.M_TH = J_red_inv[:n_free, :]
        self.M_U = J_red_inv[n_free:, :]

        pcc_th_row = J[pcc_i][free_idx]
        pcc_u_row = J[pcc_i + n][free_idx]
        self.K_P = pcc_th_row @ J_red_inv
        self.K_Q = pcc_u_row @ J_red_inv

    def _build_branch_flow_matrices(self) -> None:
        n_free = self.n_free
        n_br = len(self.branches)
        F_P_from = np.zeros((n_br, 2 * n_free))
        F_Q_from = np.zeros((n_br, 2 * n_free))
        F_P_to = np.zeros((n_br, 2 * n_free))
        F_Q_to = np.zeros((n_br, 2 * n_free))
        Q_ref_from = np.zeros(n_br)
        Q_ref_to = np.zeros(n_br)

        def th_idx(osmid: int) -> Optional[int]:
            return None if osmid == self.pcc_id else self._fidx[osmid]

        def u_idx(osmid: int) -> Optional[int]:
            return None if osmid == self.pcc_id else self._fidx[osmid]

        def fill_direction(l: int, i_bus: int, k_bus: int, Fp: np.ndarray, Fq: np.ndarray, Qref: np.ndarray, br: dict) -> None:
            g = br["g_pu"]
            b = br["b_pu"]
            bsh = br["b_sh_pu"]

            i_th = th_idx(i_bus)
            k_th = th_idx(k_bus)
            i_u = u_idx(i_bus)
            k_u = u_idx(k_bus)

            c_th_p = np.zeros(n_free)
            c_u_p = np.zeros(n_free)
            if i_th is not None:
                c_th_p[i_th] = -b
            if k_th is not None:
                c_th_p[k_th] = b
            if i_u is not None:
                c_u_p[i_u] = g
            if k_u is not None:
                c_u_p[k_u] = -g
            Fp[l, :] = c_th_p @ self.M_TH + c_u_p @ self.M_U

            c_th_q = np.zeros(n_free)
            c_u_q = np.zeros(n_free)
            if i_th is not None:
                c_th_q[i_th] = -g
            if k_th is not None:
                c_th_q[k_th] = g
            if i_u is not None:
                c_u_q[i_u] = -(b + 2.0 * bsh)
            if k_u is not None:
                c_u_q[k_u] = b
            Qref[l] = -bsh
            Fq[l, :] = c_th_q @ self.M_TH + c_u_q @ self.M_U

        for l, br in enumerate(self.branches):
            fill_direction(l, br["u"], br["v"], F_P_from, F_Q_from, Q_ref_from, br)
            fill_direction(l, br["v"], br["u"], F_P_to, F_Q_to, Q_ref_to, br)

        self.F_P_from = F_P_from
        self.F_Q_from = F_Q_from
        self.F_P_to = F_P_to
        self.F_Q_to = F_Q_to
        self.Q_ref_from = Q_ref_from
        self.Q_ref_to = Q_ref_to

    def consumption_baseline_to_pu(
        self,
        P_net_mw: Dict[int, float],
        Q_net_mvar: Optional[Dict[int, float]] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        P_pu = np.array([-P_net_mw.get(n, 0.0) / S_BASE_MVA for n in self.non_pcc], dtype=float)
        if Q_net_mvar is None:
            Q_pu = np.zeros(self.n_free)
        else:
            Q_pu = np.array([-Q_net_mvar.get(n, 0.0) / S_BASE_MVA for n in self.non_pcc], dtype=float)
        return P_pu, Q_pu

    def compute_baseline_state(self, P_inj_pu: np.ndarray, Q_inj_pu: Optional[np.ndarray] = None) -> dict:
        if Q_inj_pu is None:
            Q_inj_pu = np.zeros(self.n_free)
        rhs = np.concatenate([P_inj_pu - self.P_ref_free, Q_inj_pu - self.Q_ref_free])
        theta = self.M_TH @ rhs
        dU = self.M_U @ rhs
        P_from = self.F_P_from @ rhs
        Q_from = self.Q_ref_from + self.F_Q_from @ rhs
        P_to = self.F_P_to @ rhs
        Q_to = self.Q_ref_to + self.F_Q_to @ rhs
        P_PCC = self.P_ref_pcc + self.K_P @ rhs
        Q_PCC = self.Q_ref_pcc + self.K_Q @ rhs
        return {
            "theta": theta,
            "dU": dU,
            "U": 1.0 + dU,
            "P_from": P_from,
            "Q_from": Q_from,
            "P_to": P_to,
            "Q_to": Q_to,
            "P_PCC": P_PCC,
            "Q_PCC": Q_PCC,
            "rhs": rhs,
        }

    def summary(self) -> str:
        return (
            f"LinearGridModel(grid={GRID_NAME}, buses={self.n}, pcc={self.pcc_id}, "
            f"branches={len(self.branches)}, Sbase={S_BASE_MVA} MVA, "
            f"Zbase={Z_BASE_OHM:.6g} ohm, cond={self._cond_J_red:.2e})"
        )


# =============================================================================
# Device data structures
# =============================================================================


@dataclass
class BESSDevice:
    node: int
    p_nom_pu: float
    s_max_pu: float
    c_puh: float
    eta_ch: float
    eta_dis: float
    soc_min: float
    soc_max: float
    soc_init: float
    source: str = ""

    @property
    def p_ch_max_pu(self) -> float:
        energy_bound = (self.soc_max - self.soc_init) * self.c_puh / max(self.eta_ch, 1e-12) / DT_H
        return float(min(self.p_nom_pu, max(0.0, energy_bound)))

    @property
    def p_dis_max_pu(self) -> float:
        energy_bound = (self.soc_init - self.soc_min) * self.c_puh * self.eta_dis / DT_H
        return float(min(self.p_nom_pu, max(0.0, energy_bound)))


@dataclass
class PVDevice:
    node: int
    p_avail_pu: float
    pf: float = PF_PV
    source: str = ""
    source_id: str = ""

    @property
    def q_max_pu(self) -> float:
        return float(self.p_avail_pu * tan_from_pf(self.pf))


@dataclass
class HPDevice:
    node: int
    p_base_pu: float
    p_nom_pu: float
    pf: float = PF_HP
    q_heat_K_per_h: Optional[float] = None
    source: str = ""
    source_id: str = ""

    @property
    def p_inj_lb_pu(self) -> float:
        # Actual HP consumption in [0, P_nom].
        # Variable is injection deviation: p_inj = -delta_consumption.
        return float(self.p_base_pu - self.p_nom_pu)

    @property
    def p_inj_ub_pu(self) -> float:
        return float(self.p_base_pu)

    @property
    def q_slope(self) -> float:
        return tan_from_pf(self.pf)


@dataclass
class DERData:
    bess: List[BESSDevice] = field(default_factory=list)
    pv: List[PVDevice] = field(default_factory=list)
    hp: List[HPDevice] = field(default_factory=list)
    hp_skipped_zero_baseline: int = 0


# =============================================================================
# Baseline and asset extraction
# =============================================================================


def build_q_baseline(
    loader: Grid20Loader,
    model: LinearGridModel,
    t_col: str,
    mode: str = Q_BASE_MODE,
    pf_load: float = PF_LOAD,
    pf_hp: float = PF_HP,
) -> Dict[int, float]:
    """Return Q consumption baseline in MVAr by MV_osmid."""
    if mode == "zero":
        return {int(n): 0.0 for n in model.non_pcc}

    if loader.baseline_demand_by_mv_node is None:
        loader.build_baseline_demand()
    if loader.hp_baseline_by_mv_node is None:
        loader.build_hp_baseline()

    demand = as_node_indexed(loader.baseline_demand_by_mv_node)
    hp = as_node_indexed(loader.hp_baseline_by_mv_node)
    all_nodes = demand.index.union(hp.index)

    hp_tan = tan_from_pf(pf_hp)
    load_tan_default = tan_from_pf(pf_load)

    q_over_p_by_node: Dict[int, float] = {}
    if mode == "matpower":
        bus_path = find_existing_file([
            Path("06_Grids") / f"{GRID_NAME}_bus_data.csv",
            Path(f"{GRID_NAME}_bus_data.csv"),
            Path(__file__).resolve().parent / f"{GRID_NAME}_bus_data.csv",
        ])
        if bus_path is not None:
            bus_data = pd.read_csv(bus_path)
            if {"BUS_I", "Pd", "Qd"}.issubset(bus_data.columns):
                for _, row in bus_data.iterrows():
                    bus_i = int(row["BUS_I"])
                    osmid = bus_i - 1
                    pd_mw = safe_float(row["Pd"])
                    qd_mvar = safe_float(row["Qd"])
                    q_over_p_by_node[int(osmid)] = qd_mvar / pd_mw if abs(pd_mw) > 1e-12 else 0.0
        if not q_over_p_by_node:
            print("MATPOWER Q baseline requested, but bus_data was not found/usable. Falling back to PF mode.")
            mode = "pf"

    q_net_mvar: Dict[int, float] = {}
    for node in all_nodes:
        p_dem_kw = float(demand.loc[node, t_col]) if node in demand.index and t_col in demand.columns else 0.0
        p_hp_kw = float(hp.loc[node, t_col]) if node in hp.index and t_col in hp.columns else 0.0
        if mode == "matpower":
            load_tan = q_over_p_by_node.get(int(node), load_tan_default)
        else:
            load_tan = load_tan_default
        q_kvar = p_dem_kw * load_tan + p_hp_kw * hp_tan
        q_net_mvar[int(node)] = q_kvar / 1000.0
    return q_net_mvar


def build_baseline_from_loader(
    loader: Grid20Loader,
    model: LinearGridModel,
    t_col: str,
    q_net_mvar: Optional[Dict[int, float]] = None,
) -> dict:
    """Build baseline state from loader.net_baseline_by_mv_node."""
    if loader.net_baseline_by_mv_node is None:
        loader.build_net_baseline_by_node()
    net = as_node_indexed(loader.net_baseline_by_mv_node)
    if t_col not in net.columns:
        raise ValueError(f"net_baseline_by_mv_node does not contain {t_col!r}.")
    p_net_mw = {int(node): float(row[t_col]) / 1000.0 for node, row in net.iterrows()}
    P_inj_pu, Q_inj_pu = model.consumption_baseline_to_pu(p_net_mw, q_net_mvar)
    return model.compute_baseline_state(P_inj_pu, Q_inj_pu)


def extract_bess_devices(loader: Grid20Loader, model: LinearGridModel) -> List[BESSDevice]:
    if loader.bess_allocation is None:
        loader.load_bess_data()
    df = loader.bess_allocation
    devices: List[BESSDevice] = []
    if df is None or df.empty:
        return devices

    for _, row in df.iterrows():
        node = int(row["MV_osmid"])
        if node not in model._fidx:
            continue
        p_nom_kw = safe_float(row.get("Nominal_power_kW"))
        c_kwh = safe_float(row.get("Battery_capacity_kWh"))
        if p_nom_kw <= 0.0 or c_kwh <= 0.0:
            continue
        devices.append(BESSDevice(
            node=node,
            p_nom_pu=p_nom_kw / S_BASE_KW,
            s_max_pu=p_nom_kw / S_BASE_KW,
            c_puh=c_kwh / S_BASE_KW,
            eta_ch=safe_float(row.get("Charging_efficiency"), 1.0),
            eta_dis=safe_float(row.get("Discharging_efficiency"), safe_float(row.get("Charging_efficiency"), 1.0)),
            soc_min=SOC_MIN,
            soc_max=SOC_MAX,
            soc_init=SOC_INIT,
            source=str(row.get("source_component", "")),
        ))
    return devices


def extract_pv_devices_asset_level(loader: Grid20Loader, model: LinearGridModel, t_col: str, pf: float = PF_PV) -> List[PVDevice]:
    """Extract PV devices from raw MV_generation and LV_generation, mapped to MV nodes."""
    devices: List[PVDevice] = []
    pv_dir = year_dir("01_PV", loader.data_year)

    # MV PV rows.
    mv_path = pv_dir / "MV_generation.csv"
    if mv_path.exists():
        mv = pd.read_csv(mv_path)
        mv = mv[mv["MV_grid"].astype(str).eq(str(loader.grid_name))].copy()
        for idx, row in mv.iterrows():
            node = int(row["MV_osmid"])
            if node not in model._fidx:
                continue
            p_kw = row_time_value(row, t_col, 0.0)
            if p_kw > 1e-9:
                devices.append(PVDevice(
                    node=node,
                    p_avail_pu=p_kw / S_BASE_KW,
                    pf=pf,
                    source="MV_direct",
                    source_id=f"MV:{idx}",
                ))

    # LV PV rows, aggregated only by network mapping but kept asset-level.
    lv_path = pv_dir / "LV_generation.csv"
    connected_lv = {str(v) for v in loader.mv_to_lv.values()}
    if lv_path.exists() and connected_lv:
        lv = pd.read_csv(lv_path)
        lv["LV_grid"] = lv["LV_grid"].astype(str)
        lv = lv[lv["LV_grid"].isin(connected_lv)].copy()
        for idx, row in lv.iterrows():
            lv_grid = str(row["LV_grid"])
            mv_osmid = loader.lv_to_mv.get(lv_grid)
            if mv_osmid is None:
                continue
            node = int(mv_osmid)
            if node not in model._fidx:
                continue
            p_kw = row_time_value(row, t_col, 0.0)
            if p_kw > 1e-9:
                devices.append(PVDevice(
                    node=node,
                    p_avail_pu=p_kw / S_BASE_KW,
                    pf=pf,
                    source="LV_aggregated_asset",
                    source_id=f"{lv_grid}:{row.get('LV_osmid', idx)}",
                ))
    return devices


def extract_pv_devices_from_loader(loader: Grid20Loader, model: LinearGridModel, t_col: str, pf: float = PF_PV) -> List[PVDevice]:
    devices: List[PVDevice] = []
    if loader.pv_generation_by_mv_node is None:
        loader.build_pv_generation()
    pdf = as_node_indexed(loader.pv_generation_by_mv_node)
    if t_col not in pdf.columns:
        raise ValueError(f"PV profile does not contain {t_col!r}.")
    for node, row in pdf.iterrows():
        node = int(node)
        if node not in model._fidx:
            continue
        p_kw = safe_float(row[t_col])
        if p_kw > 1e-9:
            devices.append(PVDevice(node=node, p_avail_pu=p_kw / S_BASE_KW, pf=pf, source="MVeq_node"))
    return devices


def extract_hp_devices_asset_level(
    loader: Grid20Loader,
    model: LinearGridModel,
    t_col: str,
    pf: float = PF_HP,
    allow_start_if_base_zero: bool = ALLOW_HP_START_IF_BASE_ZERO,
) -> tuple[List[HPDevice], int]:
    """Extract HP rows from raw MV/LV allocation and create asset-level HP variables."""
    hp_dir = year_dir("03_HP", loader.data_year)
    temperature_profiles = loader._load_temperature_profiles(hp_dir)
    connected_lv = {str(v) for v in loader.mv_to_lv.values()}

    devices: List[HPDevice] = []
    skipped_zero = 0

    files = [
        (hp_dir / "MV_heat_pump_allocation.csv", "MV_direct"),
        (hp_dir / "LV_heat_pump_allocation.csv", "LV_aggregated_asset"),
    ]
    for path, source in files:
        if not path.exists():
            continue
        hp = pd.read_csv(path)
        if source == "MV_direct":
            hp = hp[hp["MV_grid"].astype(str).eq(str(loader.grid_name))].copy()
        else:
            hp["LV_grid"] = hp["LV_grid"].astype(str)
            hp = hp[hp["LV_grid"].isin(connected_lv)].copy()

        for idx, row in hp.iterrows():
            if source == "MV_direct":
                node = int(row["MV_osmid"])
                source_id = f"MV:{idx}"
            else:
                lv_grid = str(row["LV_grid"])
                mv_osmid = loader.lv_to_mv.get(lv_grid)
                if mv_osmid is None:
                    continue
                node = int(mv_osmid)
                source_id = f"{lv_grid}:{row.get('LV_osmid', idx)}"

            if node not in model._fidx:
                continue

            profile = loader._hp_consumption_for_row(row, temperature_profiles)
            if t_col not in profile.index:
                continue
            p_base_kw = safe_float(profile[t_col])
            p_nom_kw = safe_float(row.get("Nominal_power_kW"))
            if p_nom_kw <= 0.0:
                continue

            if p_base_kw <= HP_BASE_EPS_KW and not allow_start_if_base_zero:
                skipped_zero += 1
                continue

            # Thermal coefficient for one-step comfort constraint.
            profile_name = row.get("Temperature_profile_name", None)
            if profile_name is not None and profile_name in temperature_profiles.index and t_col in temperature_profiles.columns:
                t_out = safe_float(temperature_profiles.loc[profile_name, t_col], T_SET_C)
            else:
                t_out = T_SET_C
            H = safe_float(row.get("Thermal_conductivity_kW/K"))
            Cth = safe_float(row.get("Thermal_capacitance_KWh/K"))
            q_heat = None
            if Cth > 1e-12:
                q_heat = H * max(T_SET_C - t_out, 0.0) / Cth  # K/h

            devices.append(HPDevice(
                node=node,
                p_base_pu=p_base_kw / S_BASE_KW,
                p_nom_pu=p_nom_kw / S_BASE_KW,
                pf=pf,
                q_heat_K_per_h=q_heat,
                source=source,
                source_id=source_id,
            ))
    return devices, skipped_zero


def extract_hp_devices_from_loader(loader: Grid20Loader, model: LinearGridModel, t_col: str, pf: float = PF_HP) -> tuple[List[HPDevice], int]:
    """Fallback node-level HP devices from loader totals."""
    if loader.hp_baseline_by_mv_node is None:
        loader.build_hp_baseline()
    if loader.node_capacity_summary is None:
        loader.build_node_capacity_summary()
    hbase = as_node_indexed(loader.hp_baseline_by_mv_node)
    caps = as_node_indexed(loader.node_capacity_summary)
    devices: List[HPDevice] = []
    skipped_zero = 0
    for node, row in hbase.iterrows():
        node = int(node)
        if node not in model._fidx or node not in caps.index:
            continue
        p_base_kw = safe_float(row[t_col]) if t_col in hbase.columns else 0.0
        p_nom_kw = safe_float(caps.loc[node, "hp_nominal_power_kW"])
        if p_nom_kw <= 0.0:
            continue
        if p_base_kw <= HP_BASE_EPS_KW and not ALLOW_HP_START_IF_BASE_ZERO:
            skipped_zero += 1
            continue
        devices.append(HPDevice(node=node, p_base_pu=p_base_kw / S_BASE_KW, p_nom_pu=p_nom_kw / S_BASE_KW, pf=pf, source="MVeq_node"))
    return devices, skipped_zero


def build_der_data(loader: Grid20Loader, model: LinearGridModel, t_col: str) -> DERData:
    bess = extract_bess_devices(loader, model)
    pv = extract_pv_devices_asset_level(loader, model, t_col, PF_PV) if USE_ASSET_LEVEL_PV else extract_pv_devices_from_loader(loader, model, t_col, PF_PV)
    if USE_ASSET_LEVEL_HP:
        hp, skipped_zero = extract_hp_devices_asset_level(loader, model, t_col, PF_HP, ALLOW_HP_START_IF_BASE_ZERO)
    else:
        hp, skipped_zero = extract_hp_devices_from_loader(loader, model, t_col, PF_HP)
    return DERData(bess=bess, pv=pv, hp=hp, hp_skipped_zero_baseline=skipped_zero)


# =============================================================================
# LP construction and solving
# =============================================================================


@dataclass
class T1LP:
    A_ub: np.ndarray
    b_ub: np.ndarray
    bounds: List[Tuple[Optional[float], Optional[float]]]
    kP_dev: np.ndarray
    kQ_dev: np.ndarray
    A_node: np.ndarray
    var_names: List[str]
    devices: DERData
    baseline: dict
    model: LinearGridModel
    sign_flipped_to_hug: bool = False


def build_t1_lp(
    model: LinearGridModel,
    baseline: dict,
    devices: DERData,
    u_min: float = U_MIN,
    u_max: float = U_MAX,
    n_poly_line: int = N_POLY_LINE,
    n_poly_bess: int = N_POLY_BESS,
) -> T1LP:
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
        add_var(("bess", b, "p_ch"), f"bess[{dev.node},{b}].p_ch", 0.0, dev.p_ch_max_pu)
        add_var(("bess", b, "p_dis"), f"bess[{dev.node},{b}].p_dis", 0.0, dev.p_dis_max_pu)
        add_var(("bess", b, "q"), f"bess[{dev.node},{b}].q", -dev.s_max_pu, dev.s_max_pu)

    for p, dev in enumerate(devices.pv):
        add_var(("pv", p, "p_curt"), f"pv[{dev.node},{p}].p_curt", 0.0, dev.p_avail_pu)
        add_var(("pv", p, "q"), f"pv[{dev.node},{p}].q", -dev.q_max_pu, dev.q_max_pu)

    for h, dev in enumerate(devices.hp):
        add_var(("hp", h, "p_inj"), f"hp[{dev.node},{h}].p_inj", dev.p_inj_lb_pu, dev.p_inj_ub_pu)

    n_var = len(var_names)
    n_free = model.n_free
    A_node = np.zeros((2 * n_free, n_var))

    def node_i(node: int) -> Optional[int]:
        return model._fidx.get(int(node))

    # BESS mapping to nodal injection.
    for b, dev in enumerate(devices.bess):
        i = node_i(dev.node)
        if i is None:
            continue
        A_node[i, idx[("bess", b, "p_ch")]] -= 1.0
        A_node[i, idx[("bess", b, "p_dis")]] += 1.0
        A_node[n_free + i, idx[("bess", b, "q")]] += 1.0

    # PV curtailment: curtailment reduces active injection.
    for p, dev in enumerate(devices.pv):
        i = node_i(dev.node)
        if i is None:
            continue
        A_node[i, idx[("pv", p, "p_curt")]] -= 1.0
        A_node[n_free + i, idx[("pv", p, "q")]] += 1.0

    # HP: p_inj is active injection deviation; Q follows fixed PF.
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

    # Voltage constraints around baseline.
    MU_dev = model.M_U @ A_node
    dU_base = baseline["dU"]
    for i in range(n_free):
        A_rows.append(MU_dev[i, :])
        b_rows.append((u_max - 1.0) - dU_base[i])
        A_rows.append(-MU_dev[i, :])
        b_rows.append((1.0 - u_min) + dU_base[i])

    # Line apparent-power polygon, both directions.
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

    # BESS P/Q apparent-power polygon.
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

    # PV inverter wedge: |q| <= tan(phi) * (P_avail - p_curt).
    # Uses dev.pf, not the global constant, so scenario PF changes are respected.
    for p, dev in enumerate(devices.pv):
        j_curt = idx[("pv", p, "p_curt")]
        j_q = idx[("pv", p, "q")]
        m = tan_from_pf(dev.pf)

        row = np.zeros(n_var)
        row[j_q] = 1.0
        row[j_curt] = m
        A_rows.append(row)
        b_rows.append(m * dev.p_avail_pu)

        row = np.zeros(n_var)
        row[j_q] = -1.0
        row[j_curt] = m
        A_rows.append(row)
        b_rows.append(m * dev.p_avail_pu)

    # One-step HP thermal comfort constraints, asset-level when available.
    # T_next = T_set - (p_inj / P_base) * q_heat * DT
    for h, dev in enumerate(devices.hp):
        if dev.q_heat_K_per_h is None:
            continue
        if dev.p_base_pu <= HP_BASE_EPS_KW / S_BASE_KW:
            continue
        if dev.q_heat_K_per_h <= 1e-12:
            continue
        j = idx[("hp", h, "p_inj")]
        coeff = dev.q_heat_K_per_h * DT_H / dev.p_base_pu

        # T_next <= T_MAX -> -coeff*p_inj <= T_MAX - T_SET
        row = np.zeros(n_var)
        row[j] = -coeff
        A_rows.append(row)
        b_rows.append(T_MAX_C - T_SET_C)

        # T_next >= T_MIN -> coeff*p_inj <= T_SET - T_MIN
        row = np.zeros(n_var)
        row[j] = coeff
        A_rows.append(row)
        b_rows.append(T_SET_C - T_MIN_C)

    A_ub = np.vstack(A_rows) if A_rows else np.zeros((0, n_var))
    b_ub = np.array(b_rows, dtype=float)
    return T1LP(
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=bounds,
        kP_dev=kP_dev,
        kQ_dev=kQ_dev,
        A_node=A_node,
        var_names=var_names,
        devices=devices,
        baseline=baseline,
        model=model,
    )


def apply_hug_pcc_sign_convention(lp: T1LP, preferred_test_node: int = 133) -> str:
    """Flip PCC deviation vectors if model.K uses slack-injection sign."""
    model = lp.model
    test_node = preferred_test_node if preferred_test_node in model._fidx else None
    if test_node is None:
        # Prefer a node with a device, otherwise first non-PCC node.
        candidates = [d.node for d in lp.devices.bess] + [d.node for d in lp.devices.pv] + [d.node for d in lp.devices.hp]
        for n in candidates:
            if n in model._fidx:
                test_node = int(n)
                break
    if test_node is None:
        test_node = model.non_pcc[0]

    x_node = np.zeros(2 * model.n_free)
    x_node[model._fidx[test_node]] = 0.001 / S_BASE_MVA  # +1 kW injection in p.u.
    dP_pcc = float(model.K_P @ x_node)
    dQ_pcc = float(model.K_Q @ x_node)

    # Hug positive flexibility: + nodal injection should be positive PCC flexibility.
    if dP_pcc < 0.0:
        lp.kP_dev = -lp.kP_dev
        lp.kQ_dev = -lp.kQ_dev
        lp.sign_flipped_to_hug = True
        action = "flipped"
    else:
        action = "not_flipped"

    return (
        f"PCC sign test at node {test_node}: +1 kW injection -> "
        f"dP_PCC={dP_pcc*S_BASE_MVA*1000:+.6f} kW, "
        f"dQ_PCC={dQ_pcc*S_BASE_MVA*1000:+.6f} kvar; action={action}."
    )


def solve_direction(alpha: float, beta: float, lp: T1LP) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float], str]:
    c = -(alpha * lp.kP_dev + beta * lp.kQ_dev)
    result = linprog(
        c,
        A_ub=lp.A_ub,
        b_ub=lp.b_ub,
        bounds=lp.bounds,
        method="highs",
        options={"presolve": True, "disp": False},
    )
    if result.status != 0:
        return None, None, None, result.message
    z = result.x
    dP = float(lp.kP_dev @ z)
    dQ = float(lp.kQ_dev @ z)
    obj = float(alpha * dP + beta * dQ)
    return np.array([dP, dQ]), z, obj, "optimal"


def compute_fixed_direction_points(lp: T1LP, n_dirs: int = N_DIRS, start_deg: float = START_DEG) -> tuple[np.ndarray, list[dict]]:
    vertices: List[np.ndarray] = []
    results: List[dict] = []

    for k in range(n_dirs):
        deg = start_deg + k * 360.0 / n_dirs
        rad = math.radians(deg)
        alpha = math.cos(rad)
        beta = math.sin(rad)
        v, z, obj, status = solve_direction(alpha, beta, lp)
        res = {
            "angle_deg": deg,
            "alpha": alpha,
            "beta": beta,
            "status": status,
            "vertex": v,
            "dispatch": z,
            "objective": obj,
        }
        results.append(res)
        if v is not None:
            vertices.append(v)

    if not vertices:
        raise RuntimeError("No feasible FFOR direction found.")
    return sort_ccw(np.array(vertices)), results


# =============================================================================
# Output and diagnostics
# =============================================================================


def device_summary_frame(devices: DERData) -> pd.DataFrame:
    rows: List[dict] = []
    for i, d in enumerate(devices.bess):
        rows.append({
            "type": "BESS",
            "idx": i,
            "node": d.node,
            "source": d.source,
            "p_nom_MW": d.p_nom_pu * S_BASE_MVA,
            "s_max_MVA": d.s_max_pu * S_BASE_MVA,
            "capacity_MWh": d.c_puh * S_BASE_MVA,
            "p_ch_max_MW": d.p_ch_max_pu * S_BASE_MVA,
            "p_dis_max_MW": d.p_dis_max_pu * S_BASE_MVA,
            "p_base_MW": np.nan,
            "q_heat_K_per_h": np.nan,
        })
    for i, d in enumerate(devices.pv):
        rows.append({
            "type": "PV",
            "idx": i,
            "node": d.node,
            "source": d.source,
            "source_id": d.source_id,
            "p_avail_MW": d.p_avail_pu * S_BASE_MVA,
            "q_max_MVAr": d.q_max_pu * S_BASE_MVA,
            "pf": d.pf,
        })
    for i, d in enumerate(devices.hp):
        rows.append({
            "type": "HP",
            "idx": i,
            "node": d.node,
            "source": d.source,
            "source_id": d.source_id,
            "p_base_MW": d.p_base_pu * S_BASE_MVA,
            "p_nom_MW": d.p_nom_pu * S_BASE_MVA,
            "p_inj_lb_MW": d.p_inj_lb_pu * S_BASE_MVA,
            "p_inj_ub_MW": d.p_inj_ub_pu * S_BASE_MVA,
            "pf": d.pf,
            "q_heat_K_per_h": d.q_heat_K_per_h,
        })
    return pd.DataFrame(rows)


def baseline_summary_frame(loader: Grid20Loader, baseline: dict, q_net_mvar: Dict[int, float], t_col: str) -> pd.DataFrame:
    demand = loader.baseline_demand_total.loc[t_col] if loader.baseline_demand_total is not None else np.nan
    hp = loader.hp_baseline_total.loc[t_col] if loader.hp_baseline_total is not None else np.nan
    pv = loader.pv_generation_total.loc[t_col] if loader.pv_generation_total is not None else np.nan
    net = loader.net_baseline_total.loc[t_col] if loader.net_baseline_total is not None else np.nan
    return pd.DataFrame([{
        "grid": loader.grid_name,
        "data_year": loader.data_year,
        "t_col": t_col,
        "S_BASE_MVA": S_BASE_MVA,
        "demand_total_MW": float(demand) / 1000.0,
        "hp_baseline_total_MW": float(hp) / 1000.0,
        "pv_generation_total_MW": float(pv) / 1000.0,
        "net_baseline_total_MW_consumption_positive": float(net) / 1000.0,
        "q_baseline_total_MVAr_consumption_positive": sum(q_net_mvar.values()),
        "linear_P_PCC_raw_pu": baseline["P_PCC"],
        "linear_Q_PCC_raw_pu": baseline["Q_PCC"],
        "linear_P_PCC_raw_MW": baseline["P_PCC"] * S_BASE_MVA,
        "linear_Q_PCC_raw_MVAr": baseline["Q_PCC"] * S_BASE_MVA,
        "U_min_pu": float(np.min(baseline["U"])),
        "U_max_pu": float(np.max(baseline["U"])),
    }])


def line_flow_records(lp: T1LP, results: list[dict]) -> pd.DataFrame:
    records: List[dict] = []
    model = lp.model
    for res in results:
        if res["dispatch"] is None:
            continue
        z = res["dispatch"]
        dx = lp.A_node @ z
        P_from = lp.baseline["P_from"] + model.F_P_from @ dx
        Q_from = lp.baseline["Q_from"] + model.F_Q_from @ dx
        P_to = lp.baseline["P_to"] + model.F_P_to @ dx
        Q_to = lp.baseline["Q_to"] + model.F_Q_to @ dx
        for direction, P, Q in [("from", P_from, Q_from), ("to", P_to, Q_to)]:
            for l, br in enumerate(model.branches):
                if direction == "from":
                    from_node, to_node = br["u"], br["v"]
                else:
                    from_node, to_node = br["v"], br["u"]
                s_pu = math.hypot(float(P[l]), float(Q[l]))
                records.append({
                    "angle_deg": res["angle_deg"],
                    "branch_idx": l,
                    "from_node": from_node,
                    "to_node": to_node,
                    "P_MW": float(P[l]) * S_BASE_MVA,
                    "Q_MVAr": float(Q[l]) * S_BASE_MVA,
                    "S_MVA": s_pu * S_BASE_MVA,
                    "Smax_MVA": model.s_nom_pu[l] * S_BASE_MVA,
                    "loading_pct": 100.0 * s_pu / max(model.s_nom_pu[l], 1e-12),
                })
    return pd.DataFrame(records)


def voltage_summary_records(lp: T1LP, results: list[dict]) -> pd.DataFrame:
    rows: List[dict] = []
    model = lp.model
    for res in results:
        if res["dispatch"] is None:
            continue
        z = res["dispatch"]
        dx = lp.A_node @ z
        U = lp.baseline["U"] + model.M_U @ dx
        i_min = int(np.argmin(U))
        i_max = int(np.argmax(U))
        rows.append({
            "angle_deg": res["angle_deg"],
            "U_min_pu": float(U[i_min]),
            "U_min_node": model.non_pcc[i_min],
            "U_max_pu": float(U[i_max]),
            "U_max_node": model.non_pcc[i_max],
        })
    return pd.DataFrame(rows)


def dispatch_records(lp: T1LP, results: list[dict]) -> pd.DataFrame:
    rows: List[dict] = []
    for res in results:
        z = res["dispatch"]
        if z is None:
            continue
        for name, val in zip(lp.var_names, z):
            rows.append({
                "angle_deg": res["angle_deg"],
                "variable": name,
                "value_pu": float(val),
                "value_MW_or_MVAr": float(val) * S_BASE_MVA,
            })
    return pd.DataFrame(rows)


def boundary_frame(results: list[dict]) -> pd.DataFrame:
    rows: List[dict] = []
    for res in results:
        v = res["vertex"]
        rows.append({
            "angle_deg": res["angle_deg"],
            "alpha": res["alpha"],
            "beta": res["beta"],
            "status": res["status"],
            "dP_pu": np.nan if v is None else float(v[0]),
            "dQ_pu": np.nan if v is None else float(v[1]),
            "dP_MW": np.nan if v is None else float(v[0]) * S_BASE_MVA,
            "dQ_MVAr": np.nan if v is None else float(v[1]) * S_BASE_MVA,
            "objective_pu": res["objective"],
        })
    return pd.DataFrame(rows)


def polygon_frame(vertices: np.ndarray) -> pd.DataFrame:
    rows = []
    for i, v in enumerate(vertices):
        rows.append({"vertex_order": i, "dP_pu": v[0], "dQ_pu": v[1], "dP_MW": v[0] * S_BASE_MVA, "dQ_MVAr": v[1] * S_BASE_MVA})
    if len(vertices) > 0:
        v = vertices[0]
        rows.append({"vertex_order": len(vertices), "dP_pu": v[0], "dQ_pu": v[1], "dP_MW": v[0] * S_BASE_MVA, "dQ_MVAr": v[1] * S_BASE_MVA})
    return pd.DataFrame(rows)


def plot_ffor(vertices: np.ndarray, results: list[dict], area_pu2: float, output_path: Optional[Path] = None, show: bool = False) -> None:
    verts = vertices * S_BASE_MVA
    if len(verts) == 0:
        return
    closed = np.vstack([verts, verts[0]])

    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    ax.fill(closed[:, 0], closed[:, 1], alpha=0.25, label="FFOR polygon")
    ax.plot(closed[:, 0], closed[:, 1], linewidth=1.8)
    ax.scatter(verts[:, 0], verts[:, 1], s=45, zorder=3)
    for res in results:
        if res["vertex"] is None:
            continue
        v = res["vertex"] * S_BASE_MVA
        ax.annotate(f"{res['angle_deg']:.1f}°", (v[0], v[1]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.axhline(0.0, linewidth=0.8, linestyle="--")
    ax.axvline(0.0, linewidth=0.8, linestyle="--")
    ax.set_xlabel(r"$\Delta P_{PCC}^{flex}$ [MW], Hug convention")
    ax.set_ylabel(r"$\Delta Q_{PCC}^{flex}$ [MVAr], Hug convention")
    ax.set_title(f"FFOR T=1, {GRID_NAME}, {T_COL}\nSbase={S_BASE_MVA:.0f} MVA, area={area_pu2*S_BASE_MVA**2:.5f} MW·MVAr")
    ax.grid(True, alpha=0.3)
    ax.axis("equal")
    ax.legend(loc="best")
    if output_path is not None:
        fig.savefig(output_path, dpi=180)
    if show:
        plt.show()
    plt.close(fig)


def write_manifest(
    output_dir: Path,
    model: LinearGridModel,
    lp: T1LP,
    sign_message: str,
    devices: DERData,
    area_pu2: float,
) -> None:
    lines = [
        "Unified FFOR T=1 run",
        f"Grid: {GRID_NAME}",
        f"Data year: {YEAR}",
        f"Timestep: {T_COL}",
        f"S_BASE_MVA: {S_BASE_MVA}",
        f"V_BASE_KV: {V_BASE_KV}",
        f"Z_BASE_OHM: {Z_BASE_OHM}",
        f"Q_BASE_MODE: {Q_BASE_MODE}",
        f"PF_LOAD/PF_PV/PF_HP: {PF_LOAD}/{PF_PV}/{PF_HP}",
        f"SOC_MIN/SOC_MAX/SOC_INIT: {SOC_MIN}/{SOC_MAX}/{SOC_INIT}",
        f"HP comfort: [{T_MIN_C}, {T_MAX_C}] °C, setpoint={T_SET_C} °C",
        f"ALLOW_HP_START_IF_BASE_ZERO: {ALLOW_HP_START_IF_BASE_ZERO}",
        f"USE_ASSET_LEVEL_PV: {USE_ASSET_LEVEL_PV}",
        f"USE_ASSET_LEVEL_HP: {USE_ASSET_LEVEL_HP}",
        f"Directions: {N_DIRS}, start_deg={START_DEG}, step={360/N_DIRS}",
        f"Line polygon sides: {N_POLY_LINE}",
        f"BESS polygon sides: {N_POLY_BESS}",
        f"Model: {model.summary()}",
        sign_message,
        f"Variables: {len(lp.var_names)}",
        f"Inequalities: {lp.A_ub.shape[0]}",
        f"BESS devices: {len(devices.bess)}",
        f"PV devices: {len(devices.pv)}",
        f"HP devices: {len(devices.hp)}",
        f"HP skipped due to zero baseline: {devices.hp_skipped_zero_baseline}",
        f"Area_pu2: {area_pu2}",
        f"Area_MW_MVAr: {area_pu2 * S_BASE_MVA**2}",
        "Baseline formula from loader: P_base = demand_MVeq + HP_MVeq - PV_MVeq",
        "BESS baseline: 0",
        "Internal LP convention: injection-positive",
        "Reported boundary convention: Hug/Brandle positive flexibility after PCC sign test",
    ]
    with open(output_dir / "00_manifest.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# =============================================================================
# Main run function
# =============================================================================


def run_unified_ffor(
    grid_name: str = GRID_NAME,
    data_year: int = YEAR,
    t_col: str = T_COL,
    output_root: Path = OUTPUT_ROOT,
    save_outputs: bool = SAVE_OUTPUTS,
    show_plot: bool = SHOW_PLOT,
) -> dict:
    global GRID_NAME, YEAR, T_COL
    GRID_NAME = grid_name
    YEAR = data_year
    T_COL = t_col

    print("=" * 80)
    print("Unified FFOR T=1 — MV-equivalent, Sbase=100 MVA")
    print("=" * 80)
    print(f"Grid      : {grid_name}")
    print(f"Year      : {data_year}")
    print(f"Timestep  : {t_col}")
    print(f"Sbase     : {S_BASE_MVA} MVA")
    print(f"Q baseline: {Q_BASE_MODE}")
    print("=" * 80)

    loader = Grid20Loader(grid_name=grid_name, data_year=data_year, setpoint_c=T_SET_C)
    loader.build_all_profiles()

    geo_mv = Path(GEO_MV)
    model = LinearGridModel.from_geojson(geo_mv, grid_name)
    print(model.summary())

    q_net_mvar = build_q_baseline(loader, model, t_col, mode=Q_BASE_MODE, pf_load=PF_LOAD, pf_hp=PF_HP)
    baseline = build_baseline_from_loader(loader, model, t_col, q_net_mvar=q_net_mvar)

    devices = build_der_data(loader, model, t_col)
    print(f"BESS devices: {len(devices.bess)}")
    print(f"PV devices  : {len(devices.pv)}")
    print(f"HP devices  : {len(devices.hp)}")
    print(f"HP skipped zero baseline: {devices.hp_skipped_zero_baseline}")

    lp = build_t1_lp(model, baseline, devices, u_min=U_MIN, u_max=U_MAX, n_poly_line=N_POLY_LINE, n_poly_bess=N_POLY_BESS)
    sign_message = apply_hug_pcc_sign_convention(lp)
    print(sign_message)
    print(f"LP variables: {len(lp.var_names)}, inequalities: {lp.A_ub.shape[0]}")

    vertices, results = compute_fixed_direction_points(lp, n_dirs=N_DIRS, start_deg=START_DEG)
    area_pu2 = ffor_area(vertices)

    boundary = boundary_frame(results)
    print("\n=== Boundary points [MW, MVAr] ===")
    for _, row in boundary.iterrows():
        if row["status"] == "optimal":
            print(f"{row['angle_deg']:7.1f}° : dP={row['dP_MW']:+.6f} MW, dQ={row['dQ_MVAr']:+.6f} MVAr")
        else:
            print(f"{row['angle_deg']:7.1f}° : {row['status']}")
    print(f"\nArea: {area_pu2 * S_BASE_MVA**2:.6f} MW·MVAr")

    output_dir: Optional[Path] = None
    if save_outputs:
        stamp = t_col.replace(" ", "_").replace(":", "")
        output_dir = output_root / f"{grid_name}_{data_year}_{stamp}"
        output_dir.mkdir(parents=True, exist_ok=True)

        baseline_summary_frame(loader, baseline, q_net_mvar, t_col).to_csv(output_dir / "baseline_summary.csv", index=False)
        device_summary_frame(devices).to_csv(output_dir / "device_summary.csv", index=False)
        boundary.to_csv(output_dir / "ffor_boundary_points_by_direction.csv", index=False)
        polygon_frame(vertices).to_csv(output_dir / "ffor_polygon_sorted_closed.csv", index=False)
        dispatch_records(lp, results).to_csv(output_dir / "dispatch_by_direction.csv", index=False)
        line_flow_records(lp, results).to_csv(output_dir / "line_flows_by_direction.csv", index=False)
        voltage_summary_records(lp, results).to_csv(output_dir / "voltage_summary_by_direction.csv", index=False)
        plot_ffor(vertices, results, area_pu2, output_path=output_dir / "ffor_polygon.png", show=show_plot)
        write_manifest(output_dir, model, lp, sign_message, devices, area_pu2)
        print(f"\nSaved outputs to: {output_dir}")
    else:
        plot_ffor(vertices, results, area_pu2, output_path=None, show=show_plot)

    return {
        "loader": loader,
        "model": model,
        "baseline": baseline,
        "q_net_mvar": q_net_mvar,
        "devices": devices,
        "lp": lp,
        "vertices": vertices,
        "results": results,
        "area_pu2": area_pu2,
        "output_dir": output_dir,
    }


if __name__ == "__main__":
    run_unified_ffor()
