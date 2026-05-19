from pathlib import Path
import argparse
import json

import pandas as pd

from grid_loader_mveq import Grid20Loader


GRID_NAME = "20_0"
DEFAULT_YEARS = (2030, 2040, 2050)
T_SET_C = 20.0


class PierpolloGrid20Loader(Grid20Loader):
    """Grid20Loader variant that filters large LV files while reading them."""

    @staticmethod
    def read_csv_filtered_by_values(path, column, values, chunksize=100_000):
        values = set(str(value) for value in values)
        frames = []
        columns = None

        for chunk in pd.read_csv(str(path), chunksize=chunksize):
            columns = chunk.columns
            chunk[column] = chunk[column].astype(str)
            chunk = chunk[chunk[column].isin(values)].copy()
            if not chunk.empty:
                frames.append(chunk)

        if frames:
            return pd.concat(frames, ignore_index=True, sort=False)
        return pd.DataFrame(columns=columns)

    def build_pv_lv_aggregated_to_mv(self):
        if self.nodes is None:
            self.load_grid()
        if self.time_columns is None:
            self._ensure_time_columns()

        path = self.project_path / "01_PV" / str(self.data_year) / "LV_generation.csv"
        rows = {}
        connected_lv = set(str(value) for value in self.mv_to_lv.values())

        if path.exists() and connected_lv:
            lv_generation = self.read_csv_filtered_by_values(path, "LV_grid", connected_lv)
            lv_generation = self._expand_monthly_representative_days(
                lv_generation, ["LV_grid", "LV_osmid"]
            )

            for lv_grid, group in lv_generation.groupby("LV_grid"):
                mv_osmid = self.lv_to_mv.get(str(lv_grid))
                if mv_osmid is None:
                    continue
                total = group[self.time_columns].astype(float).sum(axis=0)
                rows[str(mv_osmid)] = rows.get(str(mv_osmid), self._zero_series()) + total

        self.pv_lv_aggregated_to_mv_node = self._profile_dict_to_dataframe(rows)
        self.pv_lv_aggregated_total = self._sum_profile(self.pv_lv_aggregated_to_mv_node)
        return self.pv_lv_aggregated_to_mv_node

    def build_hp_lv_aggregated_to_mv(self):
        if self.nodes is None:
            self.load_grid()
        if self.time_columns is None:
            self._ensure_time_columns()

        base_path = self.project_path / "03_HP" / str(self.data_year)
        temperature_profiles = self._load_temperature_profiles(base_path)
        connected_lv = set(str(value) for value in self.mv_to_lv.values())
        rows = {}

        path = base_path / "LV_heat_pump_allocation.csv"
        if path.exists() and connected_lv:
            lv_hp = self.read_csv_filtered_by_values(path, "LV_grid", connected_lv)
            for _, hp in lv_hp.iterrows():
                lv_grid = str(hp["LV_grid"])
                mv_osmid = self.lv_to_mv.get(lv_grid)
                if mv_osmid is None:
                    continue
                values = self._hp_consumption_for_row(hp, temperature_profiles)
                rows[str(mv_osmid)] = rows.get(str(mv_osmid), self._zero_series()) + values

        self.hp_lv_aggregated_to_mv_node = self._profile_dict_to_dataframe(rows)
        self.hp_lv_aggregated_total = self._sum_profile(self.hp_lv_aggregated_to_mv_node)
        return self.hp_lv_aggregated_to_mv_node

    def load_bess_lv_aggregated_to_mv(self):
        if self.nodes is None:
            self.load_grid()

        path = self.project_path / "02_BESS" / str(self.data_year) / "BESS_allocation_LV.csv"
        connected_lv = set(str(value) for value in self.mv_to_lv.values())

        if not path.exists() or not connected_lv:
            self.bess_lv_aggregated_to_mv_allocation = pd.DataFrame(columns=self._bess_output_columns())
            return self.bess_lv_aggregated_to_mv_allocation

        lv_bess = self.read_csv_filtered_by_values(path, "LV_grid", connected_lv)

        rows = []
        for lv_grid, group in lv_bess.groupby("LV_grid"):
            mv_osmid = self.lv_to_mv.get(str(lv_grid))
            if mv_osmid is None:
                continue

            rows.append({
                "MV_grid": self.grid_name,
                "MV_osmid": str(mv_osmid),
                "Battery_capacity_kWh": float(group["Battery_capacity_kWh"].sum()),
                "Nominal_power_kW": float(group["Nominal_power_kW"].sum()),
                "Charging_efficiency": self._weighted_average(group, "Charging_efficiency", "Battery_capacity_kWh"),
                "Discharging_efficiency": self._weighted_average(group, "Discharging_efficiency", "Battery_capacity_kWh"),
                "source_component": "LV_aggregated",
                "source_lv_grid": str(lv_grid),
                "bess_count": int(len(group)),
            })

        self.bess_lv_aggregated_to_mv_allocation = pd.DataFrame(rows, columns=self._bess_output_columns())
        return self.bess_lv_aggregated_to_mv_allocation

    def _pv_capacity_lv_aggregated_summary(self):
        path = self.project_path / "01_PV" / str(self.data_year) / "LV_P_installed.csv"
        connected_lv = set(str(value) for value in self.mv_to_lv.values())
        if not path.exists() or not connected_lv:
            return pd.DataFrame(columns=["MV_osmid"])

        pv = self.read_csv_filtered_by_values(path, "LV_grid", connected_lv)
        if pv.empty:
            return pd.DataFrame(columns=["MV_osmid"])
        pv["MV_osmid"] = pv["LV_grid"].map(self.lv_to_mv).astype(str)
        return pv.groupby("MV_osmid", as_index=False).agg(
            pv_count_lv_aggregated=("P_installed_kW", "size"),
            pv_p_installed_lv_aggregated_kW=("P_installed_kW", "sum"),
        )

    def _hp_capacity_lv_aggregated_summary(self):
        path = self.project_path / "03_HP" / str(self.data_year) / "LV_heat_pump_allocation.csv"
        connected_lv = set(str(value) for value in self.mv_to_lv.values())
        if not path.exists() or not connected_lv:
            return pd.DataFrame(columns=["MV_osmid"])

        hp = self.read_csv_filtered_by_values(path, "LV_grid", connected_lv)
        if hp.empty:
            return pd.DataFrame(columns=["MV_osmid"])
        hp["MV_osmid"] = hp["LV_grid"].map(self.lv_to_mv).astype(str)
        return hp.groupby("MV_osmid", as_index=False).agg(
            hp_count_lv_aggregated=("Nominal_power_kW", "size"),
            hp_nominal_power_lv_aggregated_kW=("Nominal_power_kW", "sum"),
            hp_thermal_capacitance_lv_aggregated_kWh_per_K=("Thermal_capacitance_KWh/K", "sum"),
            hp_thermal_conductivity_lv_aggregated_kW_per_K=("Thermal_conductivity_kW/K", "sum"),
        )


def _safe_to_csv(df, path):
    if df is None:
        pd.DataFrame().to_csv(path, index=False)
    else:
        df.to_csv(path, index=False)


def _write_cache(loader):
    output_dir = Path("outputs") / f"{loader.grid_name}_profiles_mveq_{loader.data_year}"
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
    }
    for filename, series in profiles.items():
        series.to_csv(output_dir / filename, header=["value_kW"])

    by_node_outputs = {
        "baseline_demand_by_mv_node.csv": loader.baseline_demand_by_mv_node,
        "pv_mv_direct_by_mv_node.csv": loader.pv_mv_direct_by_mv_node,
        "pv_lv_aggregated_to_mv_node.csv": loader.pv_lv_aggregated_to_mv_node,
        "pv_generation_by_mv_node.csv": loader.pv_generation_by_mv_node,
        "hp_mv_direct_by_mv_node.csv": loader.hp_mv_direct_by_mv_node,
        "hp_lv_aggregated_to_mv_node.csv": loader.hp_lv_aggregated_to_mv_node,
        "hp_baseline_by_mv_node.csv": loader.hp_baseline_by_mv_node,
        "net_baseline_by_mv_node.csv": loader.net_baseline_by_mv_node,
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
        "cache_dir": str(output_dir),
        "ffor_inputs_dir": str(input_dir),
    }
    (input_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output_dir


def build_year(year):
    loader = PierpolloGrid20Loader(
        grid_name=GRID_NAME,
        data_year=int(year),
        start_date="01-01 00:00:00",
        end_date="12-31 23:00:00",
        setpoint_c=T_SET_C,
    )
    loader.build_all_profiles()
    return _write_cache(loader)


def main():
    parser = argparse.ArgumentParser(
        description="Build cached inputs for FFOR_pierpollo.ipynb."
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
        print(f"Building cache for {GRID_NAME}, {year}...")
        output_dir = build_year(year)
        print(f"Saved: {output_dir}")


if __name__ == "__main__":
    main()
