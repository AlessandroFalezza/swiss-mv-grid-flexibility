from pathlib import Path
import argparse
import json
import math

import pandas as pd

from loader_pierpollo import PierpolloGrid20Loader, _safe_to_csv


GRID_NAME = "20_0"
DEFAULT_YEARS = (2030, 2040, 2050)
T_SET_C = 20.0
EV_POWER_FACTOR = 0.95
OUTPUT_TAG = "profiles_mveq_evbase"


class PierpolloGrid20EVBaseLoader(PierpolloGrid20Loader):
    """Pierpollo loader variant that adds fixed EV baseline demand only."""

    ev_baseline_by_mv_node = None
    ev_baseline_total = None
    ev_baseline_q_by_mv_node = None
    ev_baseline_q_total = None

    def _read_ev_base_profiles(self, bfs_codes):
        path = self.project_path / "04_EV" / str(self.data_year) / "EV_power_profiles_LV.csv"
        bfs_codes = set(int(code) for code in bfs_codes)
        frames = []
        columns = None

        for chunk in pd.read_csv(path, chunksize=250):
            columns = chunk.columns
            mask = (
                chunk["BFS_municipality_code"].isin(bfs_codes)
                & chunk["Profile_type"].astype(str).eq("Base")
            )
            chunk = chunk.loc[mask].copy()
            if not chunk.empty:
                frames.append(chunk)

        if frames:
            return pd.concat(frames, ignore_index=True, sort=False)
        return pd.DataFrame(columns=columns)

    def build_ev_baseline_lv_aggregated_to_mv(self):
        if self.nodes is None:
            self.load_grid()
        if self.time_columns is None:
            self._ensure_time_columns()

        connected_lv = set(str(value) for value in self.mv_to_lv.values())
        allocation_path = self.project_path / "04_EV" / str(self.data_year) / "EV_allocation_LV.csv"
        rows = {}

        if allocation_path.exists() and connected_lv:
            allocation = self.read_csv_filtered_by_values(
                allocation_path, "LV_grid", connected_lv
            )
            if not allocation.empty:
                allocation["LV_grid"] = allocation["LV_grid"].astype(str)
                allocation["BFS_municipality_code"] = (
                    allocation["LV_grid"].str.split("-").str[0].astype(int)
                )
                lv_shares = allocation.groupby(
                    ["LV_grid", "BFS_municipality_code"], as_index=False
                )["EV_share"].sum()

                base_profiles = self._read_ev_base_profiles(
                    lv_shares["BFS_municipality_code"].unique()
                )
                if not base_profiles.empty:
                    base_profiles = base_profiles.set_index("BFS_municipality_code")

                    for _, row in lv_shares.iterrows():
                        lv_grid = str(row["LV_grid"])
                        mv_osmid = self.lv_to_mv.get(lv_grid)
                        bfs_code = int(row["BFS_municipality_code"])
                        if mv_osmid is None or bfs_code not in base_profiles.index:
                            continue

                        share = float(row["EV_share"])
                        values = base_profiles.loc[bfs_code, self.time_columns].astype(float) * share
                        rows[str(mv_osmid)] = rows.get(str(mv_osmid), self._zero_series()) + values

        self.ev_baseline_by_mv_node = self._profile_dict_to_dataframe(rows)
        self.ev_baseline_total = self._sum_profile(self.ev_baseline_by_mv_node)

        q_over_p = math.sqrt(1.0 - EV_POWER_FACTOR**2) / EV_POWER_FACTOR
        q_rows = {}
        if self.ev_baseline_by_mv_node is not None and not self.ev_baseline_by_mv_node.empty:
            for _, row in self.ev_baseline_by_mv_node.iterrows():
                mv_osmid = str(row["MV_osmid"])
                q_rows[mv_osmid] = row[self.time_columns].astype(float) * q_over_p

        self.ev_baseline_q_by_mv_node = self._profile_dict_to_dataframe(q_rows)
        self.ev_baseline_q_total = self._sum_profile(self.ev_baseline_q_by_mv_node)
        return self.ev_baseline_by_mv_node


def _write_cache(loader):
    output_dir = Path("outputs") / f"{loader.grid_name}_{OUTPUT_TAG}_{loader.data_year}"
    input_dir = output_dir / "ffor_inputs"
    input_dir.mkdir(parents=True, exist_ok=True)

    profiles = {
        "baseline_demand_total.csv": loader.baseline_demand_total,
        "pv_mv_direct_total.csv": loader.pv_mv_direct_total,
        "pv_lv_aggregated_total.csv": loader.pv_lv_aggregated_total,
        "pv_generation_total.csv": loader.pv_generation_total,
        "hp_mv_direct_total.csv": loader.hp_mv_direct_total,
        "hp_lv_aggregated_total.csv": loader.hp_lv_aggregated_total,
        "hp_baseline_total.csv": loader.hp_baseline_total,
        "net_baseline_total.csv": loader.net_baseline_total,
        "ev_baseline_total.csv": loader.ev_baseline_total,
    }
    for filename, series in profiles.items():
        series.to_csv(output_dir / filename, header=["value_kW"])

    loader.ev_baseline_q_total.to_csv(output_dir / "ev_baseline_q_total.csv", header=["value_kvar"])

    by_node_outputs = {
        "baseline_demand_by_mv_node.csv": loader.baseline_demand_by_mv_node,
        "pv_mv_direct_by_mv_node.csv": loader.pv_mv_direct_by_mv_node,
        "pv_lv_aggregated_to_mv_node.csv": loader.pv_lv_aggregated_to_mv_node,
        "pv_generation_by_mv_node.csv": loader.pv_generation_by_mv_node,
        "hp_mv_direct_by_mv_node.csv": loader.hp_mv_direct_by_mv_node,
        "hp_lv_aggregated_to_mv_node.csv": loader.hp_lv_aggregated_to_mv_node,
        "hp_baseline_by_mv_node.csv": loader.hp_baseline_by_mv_node,
        "net_baseline_by_mv_node.csv": loader.net_baseline_by_mv_node,
        "ev_baseline_by_mv_node.csv": loader.ev_baseline_by_mv_node,
        "ev_baseline_q_by_mv_node.csv": loader.ev_baseline_q_by_mv_node,
    }
    for filename, df in by_node_outputs.items():
        df.to_csv(output_dir / filename, index=False)

    _safe_to_csv(loader.bess_mv_direct_allocation, output_dir / "bess_allocation_mv_direct.csv")
    _safe_to_csv(loader.bess_lv_aggregated_to_mv_allocation, output_dir / "bess_allocation_lv_aggregated_to_mv.csv")
    _safe_to_csv(loader.bess_allocation, output_dir / "bess_allocation_mveq.csv")
    loader.node_capacity_summary.to_csv(output_dir / "node_capacity_summary_mveq.csv", index=False)

    nodes = loader.nodes.copy()
    node_columns = [
        column for column in ["osmid", "lv_grid", "el_dmd", "source", "x", "y", "voltage"]
        if column in nodes.columns
    ]
    nodes[node_columns].to_csv(input_dir / "nodes.csv", index=False)

    connected_lv = set(str(value) for value in loader.mv_to_lv.values())
    pd.DataFrame({"LV_grid": sorted(connected_lv)}).to_csv(input_dir / "connected_lv.csv", index=False)
    pd.DataFrame({"time_column": loader.time_columns}).to_csv(input_dir / "time_columns.csv", index=False)

    year = str(loader.data_year)
    project_path = Path.cwd()

    mv_pv = pd.read_csv(project_path / "01_PV" / year / "MV_generation.csv")
    mv_pv = mv_pv[mv_pv["MV_grid"].astype(str).eq(loader.grid_name)].copy()
    mv_pv = loader._expand_monthly_representative_days(mv_pv, ["MV_grid", "MV_osmid"])
    mv_pv.to_csv(input_dir / "pv_mv_generation.csv", index=False)

    lv_pv = loader.read_csv_filtered_by_values(project_path / "01_PV" / year / "LV_generation.csv", "LV_grid", connected_lv)
    lv_pv = loader._expand_monthly_representative_days(lv_pv, ["LV_grid", "LV_osmid"])
    lv_pv.to_csv(input_dir / "pv_lv_generation.csv", index=False)

    mv_hp = pd.read_csv(project_path / "03_HP" / year / "MV_heat_pump_allocation.csv")
    mv_hp = mv_hp[mv_hp["MV_grid"].astype(str).eq(loader.grid_name)].copy()
    mv_hp.to_csv(input_dir / "hp_mv_allocation.csv", index=False)

    lv_hp = loader.read_csv_filtered_by_values(project_path / "03_HP" / year / "LV_heat_pump_allocation.csv", "LV_grid", connected_lv)
    lv_hp.to_csv(input_dir / "hp_lv_allocation.csv", index=False)

    temperature_profiles = pd.read_csv(project_path / "03_HP" / year / "Temperature_profiles.csv")
    used_profiles = set()
    for hp in (mv_hp, lv_hp):
        if "Temperature_profile_name" in hp.columns:
            used_profiles.update(hp["Temperature_profile_name"].dropna().astype(str))
    if used_profiles:
        temperature_profiles = temperature_profiles[
            temperature_profiles["Temperature_profile_name"].astype(str).isin(used_profiles)
        ].copy()
    temperature_profiles.to_csv(input_dir / "temperature_profiles.csv", index=False)

    manifest = {
        "grid_name": loader.grid_name,
        "data_year": loader.data_year,
        "setpoint_c": loader.setpoint_c,
        "ev_power_factor": EV_POWER_FACTOR,
        "cache_dir": str(output_dir),
        "ffor_inputs_dir": str(input_dir),
    }
    (input_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output_dir


def build_year(year):
    loader = PierpolloGrid20EVBaseLoader(
        grid_name=GRID_NAME,
        data_year=int(year),
        start_date="01-01 00:00:00",
        end_date="12-31 23:00:00",
        setpoint_c=T_SET_C,
    )
    loader.build_all_profiles()
    loader.build_ev_baseline_lv_aggregated_to_mv()
    return _write_cache(loader)


def main():
    parser = argparse.ArgumentParser(
        description="Build cached inputs for FFOR_pierpollo_evbase.ipynb."
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=DEFAULT_YEARS,
        help="Years to cache. Default: 2030 2040 2050.",
    )
    args = parser.parse_args()

    for year in args.years:
        print(f"Building EV-baseline cache for {GRID_NAME}, {year}...")
        output_dir = build_year(year)
        print(f"Saved: {output_dir}")


if __name__ == "__main__":
    main()
