"""
linear_grid_model.py
====================
Linearized AC power flow model (first-order Taylor expansion around flat start)
for the FFOR OPF pipeline, following Brändle, Rousseau, Nahata & Hug (2025).

Methodology
-----------
The model linearises the AC power flow around the flat operating point
(U_i = 1 p.u., θ_i = 0 ∀i), which is the reference point used in the paper.

At flat start the power injections are:

    [P]   [P_ref]   [J_Pθ  J_PU ] [θ  ]
    [Q] = [Q_ref] + [J_Qθ  J_QU ] [ΔU ]

where P_ref = 0 and Q_ref[i] = -∑_k b_sh_ik (≈ 0 for this 20 kV network).

After fixing the PCC (slack bus: θ_PCC = 0, ΔU_PCC = 0) and reducing the
Jacobian, the model is inverted to give:

    [θ  ]            [ΔP - ΔP_ref]
    [ΔU ] = J_red⁻¹ [ΔQ - ΔQ_ref]

This yields the sensitivity matrices used by the OPF to enforce:
  - Voltage limits  : |ΔU_i| ≤ 0.05 p.u.
  - Line flow limits: √(P_ij² + Q_ij²) ≤ s_nom_ij  (polygon approximation)
  - PCC power       : ΔP_PCC = K_P · [ΔP; ΔQ]  (objective of FFOR OPF)

SIGN CONVENTION  ← read carefully before interfacing with grid_loader_mveq
-----------------
This model uses the **injection-positive** (generation) convention:
    P_inj > 0  →  bus injects power into the network  (generator / discharging BESS)
    P_inj < 0  →  bus absorbs power from the network  (load / charging BESS)

The grid_loader_mveq uses the **consumption-positive** (load) convention:
    P_net > 0  →  bus consumes power  (load + HP - PV)

Conversion when calling this model:
    P_inj_k = −P_net_k          (flip sign for all buses)
    ΔP_inj_flex = −ΔP_flex      (flip sign for device deviations too)

FFOR USAGE PATTERN
------------------
Step 1 – compute baseline state once per timestep t:
    P_inj_base = -P_net_base(t)   # flip sign from loader convention
    Q_inj_base = -Q_net_base(t)   # (or 0 if Q_net not available)
    baseline   = model.compute_baseline_state(P_inj_base, Q_inj_base)

Step 2 – inside the LP loop over directions (α, β):
    # decision variables: ΔP_inj_flex, ΔQ_inj_flex (injection convention)
    delta = model.compute_flex_response(delta_P_inj, delta_Q_inj)
    dU    = baseline['dU'] + delta['dU']       # for voltage constraint
    P_br  = baseline['P_from'] + delta['P_from']  # for line constraint
    DP_PCC = delta['dP_PCC']                   # FFOR objective
    DQ_PCC = delta['dQ_PCC']

Per-unit base
-------------
  V_base = 20 kV,  S_base = 1 MVA  →  Z_base = 400 Ω,  Y_base = 0.0025 S

Line parameters (from GeoJSON edges)
  r_pu  = r_Ω  / 400
  x_pu  = x_Ω  / 400
  b_sh_pu = (b_S / 2) * 400   (half-shunt per terminal in π model)
  s_nom_pu = s_nom_MVA         (with S_base = 1 MVA)

  Series admittance:
  g_ik = r_pu / (r_pu² + x_pu²)    > 0
  b_ik = −x_pu / (r_pu² + x_pu²)   < 0

Jacobian elements at flat start (standard textbook formulas, Y-bus convention)
  Off-diagonal (i ≠ k, line between i and k):
    J_Pθ(i,k) =  b_ik
    J_PU(i,k) = -g_ik
    J_Qθ(i,k) =  g_ik
    J_QU(i,k) =  b_ik

  Diagonal:
    J_Pθ(i,i) = -∑_{k} b_ik                           >  0
    J_PU(i,i) =  ∑_{k} g_ik                            >  0
    J_Qθ(i,i) = -∑_{k} g_ik                            <  0
    J_QU(i,i) = -∑_{k} (b_ik + 2*b_sh_ik)  ≈ ∑_{k}|b_ik|  >  0

Linearised branch flows (sending end i → k)
  P_ik ≈  g_ik*(ΔU_i − ΔU_k) − b_ik*(θ_i − θ_k)
  Q_ik ≈ -b_sh_ik − (b_ik + 2*b_sh_ik)*ΔU_i + b_ik*ΔU_k − g_ik*(θ_i − θ_k)

Usage
-----
    from linear_grid_model import LinearGridModel
    from config import GEO_MV, GRID_ID

    model = LinearGridModel.from_geojson(GEO_MV, GRID_ID)
    model.summary()

    # Sensitivity matrices for OPF
    # M_U[i, :] = row that maps [ΔP; ΔQ] → ΔU_i
    M_U = model.M_U          # shape (n_non_pcc, 2*n_non_pcc)

    # Branch flow coefficients
    # FP[l, :] · [ΔP; ΔQ] = P_ij on branch l
    FP = model.F_P            # shape (n_branches, 2*n_non_pcc)
    FQ = model.F_Q

    # PCC power coefficients (for objective function)
    # K_P · [ΔP; ΔQ] = ΔP_PCC
    K_P = model.K_P           # shape (2*n_non_pcc,)
    K_Q = model.K_Q
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Per-unit base (20 kV, 1 MVA)
# ─────────────────────────────────────────────────────────────────────────────
V_BASE_KV  = 20.0          # kV  — MV bus nominal voltage
S_BASE_MVA =  1.0          # MVA — standard distribution base
Z_BASE     = V_BASE_KV**2 / S_BASE_MVA   # = 400 Ω
Y_BASE     = 1.0 / Z_BASE                 # = 0.0025 S


class LinearGridModel:
    """
    Linearized AC power flow model for the 20_0 MV network.

    Attributes
    ----------
    nodes_df   : DataFrame   — GeoJSON nodes (osmid, source, lv_grid, …)
    edges_df   : DataFrame   — GeoJSON edges (u, v, r, x, b, s_nom, …)
    node_ids   : list[int]   — all osmids, sorted
    non_pcc    : list[int]   — osmids of non-slack buses
    pcc_id     : int         — osmid of the PCC (slack bus)
    n          : int         — total number of buses
    n_free     : int         — number of non-PCC buses (= n − 1)
    branches   : list[dict]  — branch parameters in p.u.

    Sensitivity matrices (indexed by non-PCC bus ordering in self.non_pcc)
    M_U  : ndarray (n_free, 2*n_free)  — ΔU   = M_U  @ [ΔP; ΔQ]
    M_TH : ndarray (n_free, 2*n_free)  — Δθ   = M_TH @ [ΔP; ΔQ]
    F_P  : ndarray (n_br, 2*n_free)    — P_ij = F_P  @ [ΔP; ΔQ]
    F_Q  : ndarray (n_br, 2*n_free)    — Q_ij = F_Q  @ [ΔP; ΔQ]
    K_P  : ndarray (2*n_free,)         — ΔP_PCC = K_P @ [ΔP; ΔQ]
    K_Q  : ndarray (2*n_free,)         — ΔQ_PCC = K_Q @ [ΔP; ΔQ]
    s_nom_pu : ndarray (n_br,)         — line ratings in p.u.
    """

    def __init__(self, nodes_df: pd.DataFrame, edges_df: pd.DataFrame):
        self.nodes_df = nodes_df.copy()
        self.edges_df = edges_df.copy()

        # ── Cast types ───────────────────────────────────────────────────────
        self.nodes_df['osmid'] = self.nodes_df['osmid'].astype(int)
        self.edges_df['u']     = self.edges_df['u'].astype(int)
        self.edges_df['v']     = self.edges_df['v'].astype(int)

        # ── Identify PCC (slack bus) ─────────────────────────────────────────
        pcc_mask = self.nodes_df['source'].astype(str).str.lower() == 'true'
        if pcc_mask.sum() != 1:
            raise ValueError(
                f"Expected exactly 1 source node, found {pcc_mask.sum()}."
            )
        self.pcc_id   = int(self.nodes_df.loc[pcc_mask, 'osmid'].values[0])
        self.node_ids = sorted(self.nodes_df['osmid'].tolist())
        self.non_pcc  = [n for n in self.node_ids if n != self.pcc_id]
        self.n        = len(self.node_ids)
        self.n_free   = len(self.non_pcc)

        # Index maps: osmid → matrix index
        self._idx  = {n: i for i, n in enumerate(self.node_ids)}
        self._fidx = {n: i for i, n in enumerate(self.non_pcc)}

        # ── Build branch parameters in p.u. ─────────────────────────────────
        self.branches = self._build_branches()

        # ── Build Jacobian and sensitivity matrices ──────────────────────────
        J = self._build_jacobian()
        self._invert_and_build_sensitivities(J)

        # ── Build branch flow matrices ───────────────────────────────────────
        self._build_branch_flow_matrices()

    # ─────────────────────────────────────────────────────────────────────────
    # CLASS METHOD: build from GeoJSON files
    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def from_geojson(cls, geo_mv_path, grid_id: str) -> 'LinearGridModel':
        """
        Load nodes and edges from Swiss-PDGs GeoJSON and build the model.

        Parameters
        ----------
        geo_mv_path : str or Path  — directory containing the GeoJSON files
        grid_id     : str          — grid identifier, e.g. '20_0'
        """
        geo_mv_path = Path(geo_mv_path)

        def load_geojson(path):
            with open(path, 'r') as f:
                data = json.load(f)
            return pd.DataFrame(
                [feat['properties'] for feat in data['features']]
            )

        nodes = load_geojson(geo_mv_path / f'{grid_id}_nodes')
        edges = load_geojson(geo_mv_path / f'{grid_id}_edges')
        return cls(nodes, edges)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1 — Branch parameters in p.u.
    # ─────────────────────────────────────────────────────────────────────────

    def _build_branches(self) -> list:
        """
        Convert GeoJSON edge parameters to p.u. and compute series admittances.

        Returns a list of dicts, one per branch, with keys:
          u, v          : osmid of the two endpoints
          r_pu, x_pu   : series resistance and reactance [p.u.]
          g_pu, b_pu   : series conductance and susceptance [p.u.]
          b_sh_pu      : shunt susceptance per terminal [p.u.]  (≈ 0)
          s_nom_pu     : apparent power rating [p.u.]
        """
        branches = []
        for _, row in self.edges_df.iterrows():
            r = float(row['r'])          # Ω (total line)
            x = float(row['x'])          # Ω (total line)
            b = float(row['b'])          # S (total shunt)
            s_nom = float(row['s_nom'])  # MVA

            # Convert to p.u.
            r_pu   = r / Z_BASE
            x_pu   = x / Z_BASE
            b_pu   = b * Z_BASE          # total shunt in p.u.
            b_sh_pu = b_pu / 2.0         # half-shunt per π terminal
            s_nom_pu = s_nom / S_BASE_MVA

            # Series admittance y = g + jb
            z2    = r_pu**2 + x_pu**2
            g_pu  =  r_pu / z2   # > 0
            b_ser = -x_pu / z2   # < 0 (inductive line)

            branches.append({
                'u'       : int(row['u']),
                'v'       : int(row['v']),
                'r_pu'    : r_pu,
                'x_pu'    : x_pu,
                'g_pu'    : g_pu,
                'b_pu'    : b_ser,     # series susceptance (< 0)
                'b_sh_pu' : b_sh_pu,   # shunt per terminal (> 0, ≈ 0)
                's_nom_pu': s_nom_pu,
            })
        self.s_nom_pu = np.array([br['s_nom_pu'] for br in branches])

        # ── Reference injections from line shunts ────────────────────────────
        # Paper formulation at flat start:
        #   P_ref_i = 0
        #   Q_ref_i = -∑_k b_sh_ik
        # Store both the reduced non-PCC references and the PCC reference.
        # The shunt terms are small here, but keeping them avoids hidden offsets
        # when comparing absolute Q values. They cancel for pure flexibility
        # deviations.
        self.P_ref_free = np.zeros(self.n_free)
        self.Q_ref_free = np.zeros(self.n_free)
        self.P_ref_pcc = 0.0
        self.Q_ref_pcc = 0.0

        for br in branches:
            bsh = br['b_sh_pu']
            for bus in (br['u'], br['v']):
                if bus == self.pcc_id:
                    self.Q_ref_pcc -= bsh
                else:
                    self.Q_ref_free[self._fidx[bus]] -= bsh

        return branches

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2 — Full Jacobian (n × n blocks)
    # ─────────────────────────────────────────────────────────────────────────

    def _build_jacobian(self) -> np.ndarray:
        """
        Build the full 2n × 2n Jacobian matrix at flat start (U=1, θ=0).

        Block structure:
            J = | J_Pθ  J_PU |
                | J_Qθ  J_QU |

        Each block is n × n.  Ordering: self.node_ids.

        Jacobian elements (standard Y-bus textbook formulas at flat start):
          Off-diagonal i≠k (line between i and k):
            J_Pθ(i,k) =  b_ik   (series susceptance, < 0)
            J_PU(i,k) = -g_ik   (< 0)
            J_Qθ(i,k) =  g_ik   (> 0)
            J_QU(i,k) =  b_ik   (< 0)

          Diagonal i=i:
            J_Pθ(i,i) = -∑_k b_ik                       > 0
            J_PU(i,i) =  ∑_k g_ik                       > 0
            J_Qθ(i,i) = -∑_k g_ik                       < 0
            J_QU(i,i) = -∑_k (b_ik + 2*b_sh_ik)         > 0 (since b_ik<0)
        """
        n = self.n
        J_Pth = np.zeros((n, n))
        J_PU  = np.zeros((n, n))
        J_Qth = np.zeros((n, n))
        J_QU  = np.zeros((n, n))

        for br in self.branches:
            i = self._idx[br['u']]
            k = self._idx[br['v']]
            g   = br['g_pu']
            b   = br['b_pu']    # series susceptance (< 0)
            bsh = br['b_sh_pu'] # shunt per terminal (≥ 0)

            # ── Off-diagonal (symmetric for passive network) ─────────────────
            J_Pth[i, k] = b;   J_Pth[k, i] = b
            J_PU [i, k] = -g;  J_PU [k, i] = -g
            J_Qth[i, k] = g;   J_Qth[k, i] = g
            J_QU [i, k] = b;   J_QU [k, i] = b

            # ── Diagonal accumulation ────────────────────────────────────────
            J_Pth[i, i] -= b;   J_Pth[k, k] -= b   # -sum(b_ik) > 0
            J_PU [i, i] += g;   J_PU [k, k] += g   # +sum(g_ik) > 0
            J_Qth[i, i] -= g;   J_Qth[k, k] -= g   # -sum(g_ik) < 0
            # J_QU diagonal: -sum(b_ik + 2*b_sh_ik)
            J_QU [i, i] -= (b + 2*bsh)
            J_QU [k, k] -= (b + 2*bsh)

        # Assemble 2n × 2n
        J_top    = np.hstack([J_Pth, J_PU])
        J_bottom = np.hstack([J_Qth, J_QU])
        return np.vstack([J_top, J_bottom])

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3 — Reduce for slack bus and invert
    # ─────────────────────────────────────────────────────────────────────────

    def _invert_and_build_sensitivities(self, J: np.ndarray):
        """
        Remove the PCC row/column from J, invert the reduced system, and
        compute the sensitivity matrices M_U, M_TH, K_P, K_Q.

        The reduced system (2*n_free × 2*n_free) maps:
            [Δθ_free; ΔU_free] = J_red⁻¹ · [ΔP_free; ΔQ_free]

        The PCC row of J gives:
            ΔP_PCC = J_PCC_row · [Δθ_free; ΔU_free]
                   = (J_PCC_row · J_red⁻¹) · [ΔP_free; ΔQ_free]
                   = K_P · [ΔP_free; ΔQ_free]

        Then:
            M_TH = J_red⁻¹[0:n_free, :]    → Δθ sensitivity
            M_U  = J_red⁻¹[n_free:,  :]    → ΔU  sensitivity
        """
        n      = self.n
        n_free = self.n_free
        pcc_i  = self._idx[self.pcc_id]

        # Indices to keep in the reduced system
        # θ block: [0 .. n-1], U block: [n .. 2n-1]
        # PCC corresponds to row/col pcc_i (θ) and pcc_i + n (U)
        th_free  = [i       for i in range(n)   if i != pcc_i]
        U_free   = [i + n   for i in range(n)   if i != pcc_i]
        free_idx = th_free + U_free             # 2*n_free indices

        # PCC rows (θ and U parts)
        pcc_th_idx = pcc_i
        pcc_U_idx  = pcc_i + n

        # ── Reduced Jacobian (2*n_free × 2*n_free) ──────────────────────────
        J_red = J[np.ix_(free_idx, free_idx)]

        # ── PCC rows × free columns ──────────────────────────────────────────
        # J rows are 1-D slices → index with a plain list, not [:, idx]
        J_pcc_th_row = J[pcc_th_idx][free_idx]   # shape (2*n_free,)
        J_pcc_U_row  = J[pcc_U_idx ][free_idx]

        # ── Invert reduced Jacobian ──────────────────────────────────────────
        cond = np.linalg.cond(J_red)
        if cond > 1e12:
            raise RuntimeError(
                f"Reduced Jacobian is near-singular (cond={cond:.2e}). "
                "Check network topology for islands or zero-impedance branches."
            )
        J_red_inv = np.linalg.inv(J_red)

        # ── Sensitivity matrices ─────────────────────────────────────────────
        # [Δθ; ΔU] = J_red⁻¹ · ([ΔP; ΔQ] - [P_ref; Q_ref])
        # Since P_ref = 0, the effective RHS offset is just Q_ref.
        # We absorb the Q_ref offset into the inversion:
        #   [Δθ; ΔU] = J_red⁻¹·[ΔP; ΔQ] - J_red⁻¹·[0; Q_ref]
        # Define:  state_offset = J_red⁻¹ · concat([P_ref_free, Q_ref_free])
        pq_ref = np.concatenate([self.P_ref_free, self.Q_ref_free])
        self._state_offset = J_red_inv @ pq_ref   # (2*n_free,) — subtract this

        # ── Sensitivity matrices ─────────────────────────────────────────────
        self.M_TH = J_red_inv[:n_free,  :]    # (n_free, 2*n_free)
        self.M_U  = J_red_inv[n_free:,  :]    # (n_free, 2*n_free)

        # ── PCC power sensitivities (for FFOR objective) ─────────────────────
        self.K_P = J_pcc_th_row @ J_red_inv   # (2*n_free,)
        self.K_Q = J_pcc_U_row  @ J_red_inv

        # Store condition number for diagnostics
        self._cond_J_red = cond
        self.J_red_inv   = J_red_inv

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 4 — Branch flow sensitivity matrices
    # ─────────────────────────────────────────────────────────────────────────

    def _build_branch_flow_matrices(self):
        """
        Build F_P and F_Q such that:
            P_ij = F_P[l, :] · [ΔP; ΔQ]_free
            Q_ij = F_Q[l, :] · [ΔP; ΔQ]_free

        Linearised branch flows at flat start (π model, b_sh ≈ 0):
            P_ij ≈  g_ij*(ΔU_i - ΔU_j) - b_ij*(θ_i - θ_j)
            Q_ij ≈ -b_sh_ij - (b_ij + 2*b_sh_ij)*ΔU_i
                    + b_ij*ΔU_j - g_ij*(θ_i - θ_j)

        The reference term -b_sh_ij is the Q_ref contribution from shunt;
        for our network it is negligible but is kept for completeness.
        """
        n_free = self.n_free
        n_br   = len(self.branches)
        pcc_i  = self._idx[self.pcc_id]

        F_P      = np.zeros((n_br, 2 * n_free))
        F_Q      = np.zeros((n_br, 2 * n_free))
        Q_ref_br = np.zeros(n_br)

        def _free_th_idx(osmid):
            """Return column index for θ_osmid in [ΔP; ΔQ] vector, or None."""
            if osmid == self.pcc_id:
                return None
            return self._fidx[osmid]

        def _free_U_idx(osmid):
            """Return column index for ΔU_osmid in [ΔP; ΔQ] vector, or None."""
            if osmid == self.pcc_id:
                return None
            return self._fidx[osmid] + n_free

        for l, br in enumerate(self.branches):
            g   = br['g_pu']
            b   = br['b_pu']     # series susceptance (< 0)
            bsh = br['b_sh_pu']  # shunt per terminal

            # Indices of θ and ΔU for each endpoint in the free-variable vector
            i_th = _free_th_idx(br['u'])
            k_th = _free_th_idx(br['v'])
            i_U  = _free_U_idx(br['u'])
            k_U  = _free_U_idx(br['v'])

            # ── P_ij = g*(ΔU_i - ΔU_k) - b*(θ_i - θ_k) ────────────────────
            # expressed in terms of [ΔP; ΔQ] via M_TH and M_U
            #
            # Let c_th = (−b)*e_i^th + b*e_k^th   (coefficient on θ differences)
            # Let c_U  =   g *e_i^U  - g*e_k^U    (coefficient on ΔU differences)
            #
            # Then: P_ij = c_th @ Δθ + c_U @ ΔU
            #            = c_th @ M_TH @ [ΔP; ΔQ] + c_U @ M_U @ [ΔP; ΔQ]
            #            = (c_th @ M_TH + c_U @ M_U) @ [ΔP; ΔQ]

            c_th_P = np.zeros(n_free)
            c_U_P  = np.zeros(n_free)
            if i_th is not None: c_th_P[i_th] = -b   # -b*(+θ_i)
            if k_th is not None: c_th_P[k_th] =  b   # -b*(-θ_k) = +b*θ_k
            if i_U  is not None: c_U_P[i_U - n_free] =  g
            if k_U  is not None: c_U_P[k_U - n_free] = -g

            F_P[l, :] = c_th_P @ self.M_TH + c_U_P @ self.M_U

            # ── Q_ij = -b_sh - (b+2*b_sh)*ΔU_i + b*ΔU_k - g*(θ_i - θ_k) ──
            c_th_Q = np.zeros(n_free)
            c_U_Q  = np.zeros(n_free)
            if i_th is not None: c_th_Q[i_th] = -g
            if k_th is not None: c_th_Q[k_th] =  g
            if i_U  is not None: c_U_Q[i_U - n_free] = -(b + 2*bsh)
            if k_U  is not None: c_U_Q[k_U - n_free] =  b

            Q_ref_br[l]  = -bsh   # reference shunt term (≈ 0)
            F_Q[l, :]    = c_th_Q @ self.M_TH + c_U_Q @ self.M_U

        self.F_P      = F_P
        self.F_Q      = F_Q
        self.Q_ref_br = Q_ref_br   # small reference offsets for Q (≈ 0)

        # ── Add reverse direction (v → u) — point 4 ─────────────────────────
        # P_ji ≈ g*(ΔU_j - ΔU_i) - b*(θ_j - θ_i) = -P_ij + correction
        # With b_sh ≈ 0 this is exactly -P_ij; with shunt it differs slightly.
        F_P_to = np.zeros((n_br, 2 * n_free))
        F_Q_to = np.zeros((n_br, 2 * n_free))
        Q_ref_to = np.zeros(n_br)

        for l, br in enumerate(self.branches):
            g   = br['g_pu']
            b   = br['b_pu']
            bsh = br['b_sh_pu']

            i_th = _free_th_idx(br['v'])   # swap u↔v for reverse direction
            k_th = _free_th_idx(br['u'])
            i_U  = _free_U_idx(br['v'])
            k_U  = _free_U_idx(br['u'])

            # P_ji = g*(ΔU_j - ΔU_i) - b*(θ_j - θ_i)
            c_th_P = np.zeros(n_free);  c_U_P = np.zeros(n_free)
            if i_th is not None: c_th_P[i_th] = -b
            if k_th is not None: c_th_P[k_th] =  b
            if i_U  is not None: c_U_P[i_U - n_free] =  g
            if k_U  is not None: c_U_P[k_U - n_free] = -g
            F_P_to[l, :] = c_th_P @ self.M_TH + c_U_P @ self.M_U

            # Q_ji = -b_sh - (b+2*b_sh)*ΔU_j + b*ΔU_i - g*(θ_j - θ_i)
            c_th_Q = np.zeros(n_free);  c_U_Q = np.zeros(n_free)
            if i_th is not None: c_th_Q[i_th] = -g
            if k_th is not None: c_th_Q[k_th] =  g
            if i_U  is not None: c_U_Q[i_U - n_free] = -(b + 2*bsh)
            if k_U  is not None: c_U_Q[k_U - n_free] =  b
            Q_ref_to[l]    = -bsh
            F_Q_to[l, :]   = c_th_Q @ self.M_TH + c_U_Q @ self.M_U

        self.F_P_from  = F_P        # u → v direction (same as F_P)
        self.F_Q_from  = F_Q
        self.F_P_to    = F_P_to     # v → u direction
        self.F_Q_to    = F_Q_to
        self.Q_ref_from = Q_ref_br
        self.Q_ref_to   = Q_ref_to

    # ─────────────────────────────────────────────────────────────────────────
    # FFOR INTERFACE — baseline state and flexibility response
    # ─────────────────────────────────────────────────────────────────────────

    def compute_baseline_state(self,
                               P_inj_pu: np.ndarray,
                               Q_inj_pu: np.ndarray | None = None) -> dict:
        """
        Compute the linearised network state for the baseline injection.

        IMPORTANT: P_inj uses injection convention (P_inj = -P_net_consumption).

        Parameters
        ----------
        P_inj_pu : (n_free,) — active injection [p.u., injection-positive]
        Q_inj_pu : (n_free,) or None

        Returns
        -------
        dict with: theta, dU, U, P_from, Q_from, P_to, Q_to,
                   P_PCC, Q_PCC (all in p.u.)
        """
        if Q_inj_pu is None:
            Q_inj_pu = np.zeros(self.n_free)

        # RHS = [ΔP - P_ref; ΔQ - Q_ref]
        rhs = np.concatenate([
            P_inj_pu - self.P_ref_free,
            Q_inj_pu - self.Q_ref_free,
        ])

        theta  = self.M_TH @ rhs
        dU     = self.M_U  @ rhs
        P_from = self.F_P_from @ rhs
        Q_from = self.Q_ref_from + self.F_Q_from @ rhs
        P_to   = self.F_P_to   @ rhs
        Q_to   = self.Q_ref_to  + self.F_Q_to   @ rhs
        P_PCC  = self.P_ref_pcc + self.K_P @ rhs
        Q_PCC  = self.Q_ref_pcc + self.K_Q @ rhs

        return {
            'theta'  : theta,
            'dU'     : dU,
            'U'      : 1.0 + dU,
            'P_from' : P_from, 'Q_from': Q_from,
            'P_to'   : P_to,   'Q_to'  : Q_to,
            'P_PCC'  : P_PCC,  'Q_PCC' : Q_PCC,
            'rhs'    : rhs,    # stored for fast delta computation
        }

    def compute_flex_response(self,
                              delta_P_inj: np.ndarray,
                              delta_Q_inj: np.ndarray | None = None) -> dict:
        """
        Compute the linearised network DEVIATION due to flexibility.

        Add to baseline_state values to get the total state:
            dU_total   = baseline['dU']     + flex['dU']
            P_from_tot = baseline['P_from'] + flex['P_from']
            DP_PCC     = flex['P_PCC']          ← FFOR objective

        IMPORTANT: delta_P_inj uses injection convention.
        To increase HP consumption by ΔP_HP: delta_P_inj_k = -ΔP_HP
        To discharge BESS by ΔP_dis:          delta_P_inj_k = +ΔP_dis

        Parameters
        ----------
        delta_P_inj : (n_free,) — active injection deviation [p.u.]
        delta_Q_inj : (n_free,) or None

        Returns
        -------
        dict with: theta, dU, P_from, Q_from, P_to, Q_to, P_PCC, Q_PCC
        """
        if delta_Q_inj is None:
            delta_Q_inj = np.zeros(self.n_free)

        dx = np.concatenate([delta_P_inj, delta_Q_inj])
        dP_PCC = self.K_P @ dx
        dQ_PCC = self.K_Q @ dx
        return {
            'theta'  : self.M_TH     @ dx,
            'dU'     : self.M_U      @ dx,
            'P_from' : self.F_P_from @ dx,
            'Q_from' : self.F_Q_from @ dx,   # Q_ref cancels in deviations
            'P_to'   : self.F_P_to   @ dx,
            'Q_to'   : self.F_Q_to   @ dx,
            'P_PCC'  : dP_PCC,
            'Q_PCC'  : dQ_PCC,
            'dP_PCC' : dP_PCC,
            'dQ_PCC' : dQ_PCC,
        }

    @staticmethod
    def consumption_to_injection(P_net_mw: dict | np.ndarray,
                                 Q_net_mvar: dict | np.ndarray | None = None):
        """
        Convert consumption-positive (grid_loader_mveq) to injection-positive
        (linear model) convention.

            P_inj = −P_net_consumption

        Works on dicts or arrays.
        """
        if isinstance(P_net_mw, dict):
            P_inj = {k: -v for k, v in P_net_mw.items()}
        else:
            P_inj = -np.asarray(P_net_mw)

        if Q_net_mvar is None:
            Q_inj = None
        elif isinstance(Q_net_mvar, dict):
            Q_inj = {k: -v for k, v in Q_net_mvar.items()}
        else:
            Q_inj = -np.asarray(Q_net_mvar)

        return P_inj, Q_inj

    # ─────────────────────────────────────────────────────────────────────────
    # VALIDATION — compare linearised predictions with GeoJSON power flow
    # ─────────────────────────────────────────────────────────────────────────

    def validate_on_baseline(self, P_net_pu: np.ndarray,
                             Q_net_pu: np.ndarray | None = None,
                             verbose: bool = True) -> dict:
        """
        Validate the linear model against known voltage magnitudes.

        Given net nodal injections at flat-start deviation (P_net, Q_net for
        non-PCC nodes), compute predicted ΔU and θ, then compare |U_pred| with
        the GeoJSON `voltage` field (which stores the result of a peak
        simultaneous AC power flow).

        Parameters
        ----------
        P_net_pu : array (n_free,)  — net active injection at non-PCC buses [p.u.]
                    Positive = injection into network (generation > load).
                    For a load-dominated bus: P_net < 0.
        Q_net_pu : array (n_free,) or None  — reactive injections [p.u.].
                    If None, assumed zero.
        verbose  : print comparison table

        Returns
        -------
        dict with keys:
            theta_rad  : array (n_free,) — predicted voltage angles [rad]
            dU_pu      : array (n_free,) — predicted voltage deviations [p.u.]
            U_pred     : array (n_free,) — predicted |U| = 1 + ΔU [p.u.]
            U_true     : array (n_free,) — GeoJSON voltage magnitudes [p.u.]
            max_err_pu : float           — max |U_pred - U_true| [p.u.]
        """
        n_free = self.n_free
        if Q_net_pu is None:
            Q_net_pu = np.zeros(n_free)

        # Absolute state must include the P_ref/Q_ref offset.
        # This delegates to the same code path used by the OPF baseline.
        state = self.compute_baseline_state(P_net_pu, Q_net_pu)
        theta = state['theta']
        dU = state['dU']
        U_pred = state['U']

        # GeoJSON true voltages (from peak simultaneous AC PF)
        node_v = self.nodes_df.set_index('osmid')['voltage'].astype(float)
        U_true = np.array([float(node_v[n]) for n in self.non_pcc])

        max_err = float(np.max(np.abs(U_pred - U_true)))

        if verbose:
            print(f"\n=== LINEAR MODEL VALIDATION (Z_base={Z_BASE} Ω) ===")
            print(f"{'Node':>6} {'U_pred':>10} {'U_true':>10} {'error':>10}")
            print("─" * 42)
            for i, n in enumerate(self.non_pcc):
                err = U_pred[i] - U_true[i]
                flag = " ←" if abs(err) > 0.005 else ""
                print(f"{n:>6} {U_pred[i]:>10.6f} {U_true[i]:>10.6f}"
                      f" {err:>+10.6f}{flag}")
            print(f"\nMax |error|: {max_err:.6f} p.u.")
            if max_err < 0.005:
                print("✅ Model valid (error < 0.5%)")
            elif max_err < 0.02:
                print("⚠️  Small discrepancy (<2%) — acceptable for linearisation")
            else:
                print("❌ Large error — check operating point or base values")

        return {
            'theta_rad': theta,
            'dU_pu'    : dU,
            'U_pred'   : U_pred,
            'U_true'   : U_true,
            'max_err_pu': max_err,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # CONVENIENCE — get injection vector for a timestep baseline
    # ─────────────────────────────────────────────────────────────────────────

    def injection_to_pu(self, P_inj_mw: dict,
                        Q_inj_mvar: dict | None = None) -> tuple:
        """
        Convert already injection-positive nodal values to p.u. ordered by
        ``self.non_pcc``.

        Parameters
        ----------
        P_inj_mw : dict {osmid: P in MW}
            Positive means generation/injection into the network.
        Q_inj_mvar : dict {osmid: Q in MVAr} or None
            Positive means reactive injection.

        Returns
        -------
        P_pu, Q_pu : ndarray, ndarray
            Injection-positive p.u. vectors ordered as ``self.non_pcc``.
        """
        P_pu = np.array([
            P_inj_mw.get(n, 0.0) / S_BASE_MVA for n in self.non_pcc
        ], dtype=float)

        if Q_inj_mvar is None:
            Q_pu = np.zeros(self.n_free)
        else:
            Q_pu = np.array([
                Q_inj_mvar.get(n, 0.0) / S_BASE_MVA for n in self.non_pcc
            ], dtype=float)
        return P_pu, Q_pu

    def consumption_baseline_to_pu(self, P_net_mw: dict,
                                   Q_net_mvar: dict | None = None) -> tuple:
        """
        Convert grid_loader_mveq consumption-positive baseline to the
        injection-positive convention used by this linear network model.

        Parameters
        ----------
        P_net_mw : dict {osmid: P_net in MW}
            Positive means load/consumption, e.g. ``demand + HP - PV``.
        Q_net_mvar : dict {osmid: Q_net in MVAr} or None
            Positive means reactive consumption.

        Returns
        -------
        P_inj_pu, Q_inj_pu : ndarray, ndarray
            Injection-positive p.u. vectors ordered as ``self.non_pcc``.

        Notes
        -----
        This function flips signs:
            P_inj = -P_net_consumption
            Q_inj = -Q_net_consumption
        """
        P_pu = np.array([
            -P_net_mw.get(n, 0.0) / S_BASE_MVA for n in self.non_pcc
        ], dtype=float)

        if Q_net_mvar is None:
            Q_pu = np.zeros(self.n_free)
        else:
            Q_pu = np.array([
                -Q_net_mvar.get(n, 0.0) / S_BASE_MVA for n in self.non_pcc
            ], dtype=float)
        return P_pu, Q_pu

    def baseline_to_pu(self, P_inj_mw: dict,
                       Q_inj_mvar: dict | None = None) -> tuple:
        """
        Backward-compatible alias for ``injection_to_pu``.

        Prefer using one of the explicit functions:
          - ``injection_to_pu`` if your input is already injection-positive.
          - ``consumption_baseline_to_pu`` if your input comes from
            ``grid_loader_mveq`` and is consumption-positive.
        """
        return self.injection_to_pu(P_inj_mw, Q_inj_mvar)

    # ─────────────────────────────────────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────────────────────────────────────

    def summary(self):
        """Print a concise summary of the model."""
        print("=" * 55)
        print("LinearGridModel — network 20_0")
        print("=" * 55)
        print(f"  Buses total       : {self.n}")
        print(f"  PCC (slack) osmid : {self.pcc_id}")
        print(f"  Free buses        : {self.n_free}")
        print(f"  Branches          : {len(self.branches)}")
        print(f"  V_base            : {V_BASE_KV} kV")
        print(f"  S_base            : {S_BASE_MVA} MVA")
        print(f"  Z_base            : {Z_BASE} Ω")
        print()
        print(f"  Jacobian cond     : {self._cond_J_red:.2e}")
        print(f"  M_U  shape        : {self.M_U.shape}")
        print(f"  M_TH shape        : {self.M_TH.shape}")
        print(f"  F_P_from shape    : {self.F_P_from.shape}")
        print(f"  F_P_to   shape    : {self.F_P_to.shape}")
        print(f"  K_P      shape    : {self.K_P.shape}")
        print()
        print(f"  s_nom range       : {self.s_nom_pu.min():.2f} – "
              f"{self.s_nom_pu.max():.2f} p.u.")
        brs = self.branches
        g_vals = [br['g_pu'] for br in brs]
        b_vals = [br['b_pu'] for br in brs]
        print(f"  g_ik range (p.u.) : {min(g_vals):.1f} – {max(g_vals):.1f}")
        print(f"  b_ik range (p.u.) : {min(b_vals):.1f} – {max(b_vals):.1f}")
        print("=" * 55)


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from config import GEO_MV, GRID_ID

    print("Building LinearGridModel from GeoJSON …")
    model = LinearGridModel.from_geojson(GEO_MV, GRID_ID)
    model.summary()

    # ── Quick sanity check: load the GeoJSON el_dmd as proxy for P_net ───────
    # Each consumer node absorbs el_dmd [MW]. This is consumption-positive,
    # so use consumption_baseline_to_pu() to flip to injection-positive.
    nodes_df = model.nodes_df.set_index('osmid')
    P_load_mw = {
        int(n): float(nodes_df.loc[n, 'el_dmd'])
        for n in model.non_pcc
        if float(nodes_df.loc[n, 'el_dmd']) > 0
    }

    P_pu, Q_pu = model.consumption_baseline_to_pu(P_load_mw)

    print("\nValidating against GeoJSON peak-load voltages …")
    results = model.validate_on_baseline(P_pu, Q_pu)

    print("\n=== Branch flow sample (top 5 most loaded, from-end) ===")
    state = model.compute_baseline_state(P_pu, Q_pu)
    P_br  = state['P_from']
    Q_br  = state['Q_from']
    S_br  = np.sqrt(P_br**2 + Q_br**2)
    loading = S_br / model.s_nom_pu
    top5    = np.argsort(loading)[::-1][:5]
    print(f"{'Branch':>15} {'P_pu':>8} {'Q_pu':>8} {'|S|_pu':>8} {'loading%':>9}")
    print("─" * 55)
    for l in top5:
        br = model.branches[l]
        print(f"{br['u']:>5}→{br['v']:<5} {' ':>4}"
              f"{P_br[l]:>8.4f} {Q_br[l]:>8.4f} "
              f"{S_br[l]:>8.4f} {loading[l]*100:>8.1f}%")
