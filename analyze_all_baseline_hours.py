import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


GRID_NAME = "20_0"
YEARS = (2030, 2040, 2050)

SBASE_MVA = 100.0
SBASE_KW = SBASE_MVA * 1000.0
SBASE_KVAR = SBASE_MVA * 1000.0

PF_DER = 0.95
TAN_PHI = math.tan(math.acos(PF_DER))
T_SET_C = 20.0

VMIN = 0.95
VMAX = 1.05
VIOLATION_TOL = 1e-9


@dataclass
class GridModel:
    node_ids: list[str]
    bus_ids: list[int]
    bus_to_pos: dict[int, int]
    osmid_to_bus: dict[str, int]
    slack_idx: int
    q_over_p: dict[int, float]
    p_ref: np.ndarray
    q_ref: np.ndarray
    A_pf: np.ndarray
    line_params: list[dict]


@dataclass
class YearInputs:
    profile_dir: Path
    ffor_input_dir: Path
    nodes: pd.DataFrame
    mv_to_lv: dict[str, str]
    lv_to_mv: dict[str, str]
    time_columns: list[str]
    temperature_profiles: pd.DataFrame
    demand_by_mv: pd.DataFrame
    ev_p_by_mv: pd.DataFrame
    ev_q_by_mv: pd.DataFrame
    pv_mv: pd.DataFrame
    pv_lv: pd.DataFrame
    hp_mv: pd.DataFrame
    hp_lv: pd.DataFrame


def normalize_line(from_bus: int, to_bus: int) -> str:
    a, b = sorted((int(from_bus), int(to_bus)))
    return f"{a}-{b}"


def read_optional_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def load_year_inputs(root: Path, year: int) -> YearInputs:
    profile_dir = root / "outputs" / f"{GRID_NAME}_profiles_mveq_evbase_{year}"
    ffor_input_dir = profile_dir / "ffor_inputs"

    nodes = pd.read_csv(ffor_input_dir / "nodes.csv", dtype={"osmid": str, "lv_grid": str})
    mv_nodes_with_lv = nodes[nodes["lv_grid"].astype(str).ne("-1")].copy()
    mv_to_lv = dict(zip(mv_nodes_with_lv["osmid"].astype(str), mv_nodes_with_lv["lv_grid"].astype(str)))
    lv_to_mv = {lv_grid: mv_osmid for mv_osmid, lv_grid in mv_to_lv.items()}

    return YearInputs(
        profile_dir=profile_dir,
        ffor_input_dir=ffor_input_dir,
        nodes=nodes,
        mv_to_lv=mv_to_lv,
        lv_to_mv=lv_to_mv,
        time_columns=pd.read_csv(ffor_input_dir / "time_columns.csv")["time_column"].astype(str).tolist(),
        temperature_profiles=pd.read_csv(ffor_input_dir / "temperature_profiles.csv").set_index("Temperature_profile_name"),
        demand_by_mv=pd.read_csv(profile_dir / "baseline_demand_by_mv_node.csv"),
        ev_p_by_mv=pd.read_csv(profile_dir / "ev_baseline_by_mv_node.csv"),
        ev_q_by_mv=pd.read_csv(profile_dir / "ev_baseline_q_by_mv_node.csv"),
        pv_mv=read_optional_csv(ffor_input_dir / "pv_mv_generation.csv"),
        pv_lv=read_optional_csv(ffor_input_dir / "pv_lv_generation.csv"),
        hp_mv=read_optional_csv(ffor_input_dir / "hp_mv_allocation.csv"),
        hp_lv=read_optional_csv(ffor_input_dir / "hp_lv_allocation.csv"),
    )


def build_grid_model(root: Path, nodes: pd.DataFrame) -> GridModel:
    node_ids = [str(x) for x in nodes["osmid"].astype(str)]
    bus_ids = [int(x) + 1 for x in node_ids]
    bus_to_pos = {bus: i for i, bus in enumerate(bus_ids)}
    osmid_to_bus = {osmid: int(osmid) + 1 for osmid in node_ids}
    n_bus = len(bus_ids)

    slack_bus = int(pd.read_excel(root / "06_Grids" / "20_0_matpower.xlsx", sheet_name="generator_data").iloc[0]["GEN_BUS"])
    slack_idx = bus_to_pos[slack_bus]

    bus_data = pd.read_csv(root / "06_Grids" / "20_0_bus_data.csv")
    bus_data["BUS_I"] = bus_data["BUS_I"].astype(int)
    q_over_p = {
        int(row.BUS_I): (float(row.Qd) / float(row.Pd) if abs(float(row.Pd)) > 1e-12 else 0.0)
        for _, row in bus_data.iterrows()
    }

    branch = pd.read_csv(root / "06_Grids" / "20_0_branch_data.csv")
    branch = branch[branch["BR_STATUS"].astype(int).eq(1)].copy()

    with open(root / "06_Grids" / "20_0_edges.txt", "r", encoding="utf-8") as handle:
        edges_geojson = json.load(handle)

    line_s_nom_mva = {}
    for feature in edges_geojson["features"]:
        props = feature["properties"]
        f_bus = int(props["u"]) + 1
        t_bus = int(props["v"]) + 1
        s_nom = float(props.get("s_nom", 0.0) or 0.0)
        line_s_nom_mva[(f_bus, t_bus)] = s_nom
        line_s_nom_mva[(t_bus, f_bus)] = s_nom

    Jp_theta = np.zeros((n_bus, n_bus))
    Jp_u = np.zeros((n_bus, n_bus))
    Jq_u = np.zeros((n_bus, n_bus))
    p_ref = np.zeros(n_bus)
    q_ref = np.zeros(n_bus)
    line_params = []

    for _, row in branch.iterrows():
        f_bus = int(row["F_BUS"])
        t_bus = int(row["T_BUS"])
        if f_bus not in bus_to_pos or t_bus not in bus_to_pos:
            continue

        i = bus_to_pos[f_bus]
        k = bus_to_pos[t_bus]
        y = 1 / complex(float(row["BR_R"]), float(row["BR_X"]))
        g = y.real
        b = y.imag
        b_sh = float(row["BR_B"]) / 2.0

        Jp_theta[i, i] += -b
        Jp_theta[k, k] += -b
        Jp_theta[i, k] += b
        Jp_theta[k, i] += b

        Jp_u[i, i] += g
        Jp_u[k, k] += g
        Jp_u[i, k] += -g
        Jp_u[k, i] += -g

        Jq_u[i, i] += -2.0 * b_sh - b
        Jq_u[k, k] += -2.0 * b_sh - b
        Jq_u[i, k] += b
        Jq_u[k, i] += b

        q_ref[i] += -b_sh
        q_ref[k] += -b_sh

        line_params.append(
            {
                "from_bus": f_bus,
                "to_bus": t_bus,
                "i": i,
                "k": k,
                "g": g,
                "b": b,
                "b_sh": b_sh,
                "s_nom_mva": line_s_nom_mva[(f_bus, t_bus)],
                "physical_line": normalize_line(f_bus, t_bus),
            }
        )

    Jq_theta = -Jp_u

    n_var = 2 * n_bus + 2
    theta_offset = 0
    du_offset = n_bus
    pcc_p_idx = 2 * n_bus
    pcc_q_idx = 2 * n_bus + 1

    rows = []

    def add_row(coefs: dict[int, float]):
        row = np.zeros(n_var)
        for idx, value in coefs.items():
            row[idx] += value
        rows.append(row)

    add_row({theta_offset + slack_idx: 1.0})
    add_row({du_offset + slack_idx: 1.0})

    for i in range(n_bus):
        coefs = {}
        if i == slack_idx:
            coefs[pcc_p_idx] = 1.0
        for k in range(n_bus):
            if abs(Jp_theta[i, k]) > 0.0:
                coefs[theta_offset + k] = coefs.get(theta_offset + k, 0.0) - Jp_theta[i, k]
            if abs(Jp_u[i, k]) > 0.0:
                coefs[du_offset + k] = coefs.get(du_offset + k, 0.0) - Jp_u[i, k]
        add_row(coefs)

        coefs = {}
        if i == slack_idx:
            coefs[pcc_q_idx] = 1.0
        for k in range(n_bus):
            if abs(Jq_theta[i, k]) > 0.0:
                coefs[theta_offset + k] = coefs.get(theta_offset + k, 0.0) - Jq_theta[i, k]
            if abs(Jq_u[i, k]) > 0.0:
                coefs[du_offset + k] = coefs.get(du_offset + k, 0.0) - Jq_u[i, k]
        add_row(coefs)

    return GridModel(
        node_ids=node_ids,
        bus_ids=bus_ids,
        bus_to_pos=bus_to_pos,
        osmid_to_bus=osmid_to_bus,
        slack_idx=slack_idx,
        q_over_p=q_over_p,
        p_ref=p_ref,
        q_ref=q_ref,
        A_pf=np.vstack(rows),
        line_params=line_params,
    )


def profiles_by_bus(df: pd.DataFrame, inputs: YearInputs, grid: GridModel, source: str) -> np.ndarray:
    out = np.zeros((len(grid.bus_ids), len(inputs.time_columns)))
    if df.empty:
        return out

    for _, row in df.iterrows():
        mv_osmid = str(row["MV_osmid"]) if source == "MV" else inputs.lv_to_mv.get(str(row["LV_grid"]))
        if mv_osmid not in grid.osmid_to_bus:
            continue
        bus_pos = grid.bus_to_pos[grid.osmid_to_bus[mv_osmid]]
        out[bus_pos, :] += row[inputs.time_columns].astype(float).to_numpy()
    return out


def demand_profiles_by_bus(df: pd.DataFrame, inputs: YearInputs, grid: GridModel) -> np.ndarray:
    out = np.zeros((len(grid.bus_ids), len(inputs.time_columns)))
    indexed = df.copy()
    indexed["MV_osmid"] = indexed["MV_osmid"].astype(str)
    indexed = indexed.set_index("MV_osmid")
    for pos, osmid in enumerate(grid.node_ids):
        if osmid in indexed.index:
            out[pos, :] = indexed.loc[osmid, inputs.time_columns].astype(float).to_numpy()
    return out


def hp_profiles_by_bus(inputs: YearInputs, grid: GridModel) -> np.ndarray:
    out = np.zeros((len(grid.bus_ids), len(inputs.time_columns)))

    for df, source in [(inputs.hp_mv, "MV"), (inputs.hp_lv, "LV")]:
        if df.empty:
            continue
        for _, row in df.iterrows():
            mv_osmid = str(row["MV_osmid"]) if source == "MV" else inputs.lv_to_mv.get(str(row["LV_grid"]))
            if mv_osmid not in grid.osmid_to_bus:
                continue
            profile_name = row["Temperature_profile_name"]
            if profile_name not in inputs.temperature_profiles.index:
                continue

            temp = inputs.temperature_profiles.loc[profile_name, inputs.time_columns].astype(float)
            delta = (T_SET_C - temp).clip(lower=0.0)
            thermal_kw = float(row["Thermal_conductivity_kW/K"]) * delta

            if {"COP_0", "COP_1", "COP_2"}.issubset(row.index) and not pd.isna(row.get("COP_0")):
                cop = (
                    float(row["COP_0"])
                    + float(row["COP_1"]) * delta
                    + float(row["COP_2"]) * delta * delta
                ).clip(lower=1.0)
            else:
                cop = pd.Series(float(row["COP"]), index=inputs.time_columns)

            electric_kw = (thermal_kw / cop).clip(lower=0.0, upper=float(row["Nominal_power_kW"]))
            bus_pos = grid.bus_to_pos[grid.osmid_to_bus[mv_osmid]]
            out[bus_pos, :] += electric_kw.to_numpy()

    return out


def solve_year_baselines(inputs: YearInputs, grid: GridModel):
    n_bus = len(grid.bus_ids)
    n_times = len(inputs.time_columns)

    conventional_p_kw = demand_profiles_by_bus(inputs.demand_by_mv, inputs, grid)
    ev_p_kw = demand_profiles_by_bus(inputs.ev_p_by_mv, inputs, grid)
    ev_q_kvar = demand_profiles_by_bus(inputs.ev_q_by_mv, inputs, grid)
    pv_p_kw = profiles_by_bus(inputs.pv_mv, inputs, grid, "MV") + profiles_by_bus(inputs.pv_lv, inputs, grid, "LV")
    hp_p_kw = hp_profiles_by_bus(inputs, grid)

    conventional_q_kvar = np.zeros_like(conventional_p_kw)
    for i, bus in enumerate(grid.bus_ids):
        conventional_q_kvar[i, :] = conventional_p_kw[i, :] * grid.q_over_p.get(bus, 0.0)

    fixed_p_kw = conventional_p_kw + ev_p_kw
    fixed_q_kvar = conventional_q_kvar + ev_q_kvar

    rhs = np.zeros((2 * n_bus + 2, n_times))
    rhs[0, :] = 0.0
    rhs[1, :] = 0.0

    p_ref_col = grid.p_ref.reshape(-1, 1)
    q_ref_col = grid.q_ref.reshape(-1, 1)

    # Same baseline convention as FFOR_definitivo.ipynb:
    # network + PCC = fixed demand - PV + HP.
    rhs[2 : 2 + 2 * n_bus : 2, :] = (
        p_ref_col + fixed_p_kw / SBASE_KW - pv_p_kw / SBASE_KW + hp_p_kw / SBASE_KW
    )
    rhs[3 : 2 + 2 * n_bus : 2, :] = (
        q_ref_col + fixed_q_kvar / SBASE_KVAR + hp_p_kw * TAN_PHI / SBASE_KVAR
    )

    solution = np.linalg.solve(grid.A_pf, rhs)
    theta = solution[:n_bus, :]
    du = solution[n_bus : 2 * n_bus, :]
    pcc_p = solution[2 * n_bus, :]
    pcc_q = solution[2 * n_bus + 1, :]

    max_loading = np.full(n_times, -np.inf)
    most_loaded_line = np.full(n_times, "", dtype=object)
    violated_lines = [set() for _ in range(n_times)]

    for params in grid.line_params:
        for i, k, from_bus, to_bus in [
            (params["i"], params["k"], params["from_bus"], params["to_bus"]),
            (params["k"], params["i"], params["to_bus"], params["from_bus"]),
        ]:
            theta_ik = theta[i, :] - theta[k, :]
            du_ik = du[i, :] - du[k, :]
            p_pu = -params["b"] * theta_ik + params["g"] * du_ik
            q_pu = -params["b_sh"] * (1.0 + 2.0 * du[i, :]) - params["b"] * du_ik - params["g"] * theta_ik
            s_mva = np.hypot(p_pu * SBASE_MVA, q_pu * SBASE_MVA)
            loading = s_mva / params["s_nom_mva"]

            better = loading > max_loading
            max_loading[better] = loading[better]
            most_loaded_line[better] = params["physical_line"]

            bad_idx = np.flatnonzero(loading > 1.0 + VIOLATION_TOL)
            for idx in bad_idx:
                violated_lines[idx].add(params["physical_line"])

    voltage = 1.0 + du
    v_min = voltage.min(axis=0)
    v_max = voltage.max(axis=0)
    voltage_bad = (voltage < VMIN - VIOLATION_TOL) | (voltage > VMAX + VIOLATION_TOL)
    violated_voltage_buses = []
    for t in range(n_times):
        buses = [str(grid.bus_ids[i]) for i in np.flatnonzero(voltage_bad[:, t])]
        violated_voltage_buses.append(",".join(buses))

    return pd.DataFrame(
        {
            "timestamp": inputs.time_columns,
            "hour": [ts.split(" ")[1][:2] + ":00" for ts in inputs.time_columns],
            "P_pcc_import_MW": pcc_p * SBASE_MVA,
            "Q_pcc_import_MVAr": pcc_q * SBASE_MVA,
            "V_min_pu": v_min,
            "V_max_pu": v_max,
            "line_loading_max": max_loading,
            "most_loaded_line": most_loaded_line,
            "violated_lines": [",".join(sorted(lines)) for lines in violated_lines],
            "violated_voltage_buses": violated_voltage_buses,
        }
    )


def main():
    root = Path.cwd()
    output_dir = root / "outputs" / "baseline_all_hours_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_details = []
    hour_rows = []
    year_rows = []

    for year in YEARS:
        print(f"Analyzing {year}...")
        inputs = load_year_inputs(root, year)
        grid = build_grid_model(root, inputs.nodes)
        df = solve_year_baselines(inputs, grid)
        df.insert(0, "year", year)

        df["line_overload"] = df["violated_lines"].astype(str).ne("")
        df["voltage_violation"] = df["violated_voltage_buses"].astype(str).ne("")
        df["baseline_unfeasible"] = df["line_overload"] | df["voltage_violation"]
        df["cause"] = np.select(
            [
                df["line_overload"] & df["voltage_violation"],
                df["line_overload"],
                df["voltage_violation"],
            ],
            ["line_overload;voltage_violation", "line_overload", "voltage_violation"],
            default="none",
        )

        all_details.append(df)

        hour_counts = (
            df[df["baseline_unfeasible"]]
            .groupby("hour")
            .size()
            .reindex([f"{h:02d}:00" for h in range(24)], fill_value=0)
            .reset_index(name="unfeasible_baselines")
            .rename(columns={"index": "hour"})
        )
        hour_counts.insert(0, "year", year)
        hour_rows.append(hour_counts)

        year_rows.append(
            {
                "year": year,
                "total_baselines": len(df),
                "unfeasible_baselines": int(df["baseline_unfeasible"].sum()),
                "line_overload_baselines": int(df["line_overload"].sum()),
                "voltage_violation_baselines": int(df["voltage_violation"].sum()),
            }
        )

    detail = pd.concat(all_details, ignore_index=True)
    by_hour = pd.concat(hour_rows, ignore_index=True)
    by_year = pd.DataFrame(year_rows)
    infeasible_detail = detail[detail["baseline_unfeasible"]].copy()

    detail.to_csv(output_dir / "baseline_all_hours_detail.csv", index=False)
    infeasible_detail.to_csv(output_dir / "baseline_all_hours_unfeasible_detail.csv", index=False)
    by_hour.to_csv(output_dir / "baseline_unfeasible_counts_by_year_hour.csv", index=False)
    by_year.to_csv(output_dir / "baseline_unfeasible_counts_by_year.csv", index=False)

    print("\nUnfeasible baselines by year:")
    print(by_year.to_string(index=False))

    print("\nUnfeasible baselines by hour of day:")
    print(by_hour.pivot(index="hour", columns="year", values="unfeasible_baselines").to_string())

    print(f"\nCSV outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
