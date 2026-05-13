"""
grid_loader_mveq.py

Grid20Loader per costruire una baseline MV-equivalent della rete MV 20_0.

Idea metodologica:
  - La rete modellata esplicitamente è la rete MV.
  - Le reti LV a valle delle sottostazioni MV/LV sono aggregate al nodo MV
    corrispondente.
  - La baseline nodale MV-equivalent è:

        P_base_MVeq(t) = demand_MVeq(t) + HP_MVeq_base(t) - PV_MVeq(t)

    con convenzione:
        P > 0 = consumo / assorbimento dalla rete
        P < 0 = generazione netta / iniezione verso rete

Differenze principali rispetto alla versione precedente:
  - PV: include sia MV_generation.csv sia LV_generation.csv aggregato ai nodi MV/LV.
  - HP: include sia MV_heat_pump_allocation.csv sia LV_heat_pump_allocation.csv
    aggregato ai nodi MV/LV.
  - BESS: include sia BESS_allocation_MV.csv sia BESS_allocation_LV.csv aggregato
    ai nodi MV/LV. BESS baseline = 0; serve per la flessibilità successiva.
  - Temperatura HP: nessun filtro stagionale a 14 °C. Il riscaldamento è attivo
    quando T_set > T_out, coerentemente con il modello stazionario a temperatura
    interna costante.

Uso:
    from grid_loader_mveq import Grid20Loader

    loader = Grid20Loader(data_year=2030)
    loader.build_all_profiles()
    loader.save_outputs()
"""

import calendar
import json
import os
import zipfile
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from config import SWISS_PDGS, ZAPPAROLI, GRID_ID, DATA_YEAR


class Grid20Loader:
    """Loader per rete MV con aggregazione downstream LV a nodi MV/LV."""

    def __init__(
        self,
        grid_name=GRID_ID,
        data_year=DATA_YEAR,
        start_date="01-01 00:00:00",
        end_date="12-31 23:00:00",
        setpoint_c=20.0,
    ):
        self.grid_name = grid_name
        self.data_year = data_year
        self.start_date = self._parse_month_day(start_date)
        self.end_date = self._parse_month_day(end_date)
        self.setpoint_c = float(setpoint_c)

        self.swiss_pdgs_path = Path(SWISS_PDGS)
        self.zapparoli_path = Path(ZAPPAROLI)

        self.geo_mv_path = self.swiss_pdgs_path / "grids" / "geojson_data" / "MV"
        self.geo_lv_path = self.swiss_pdgs_path / "grids" / "geojson_data" / "LV"

        self.nodes = None
        self.edges = None
        self.mv_to_lv = {}
        self.lv_to_mv = {}
        self.time_columns = None

        # Domanda convenzionale già MV-equivalent:
        # MV nodes puri + domanda LV aggregata alle sottostazioni MV/LV.
        self.baseline_demand_by_mv_node = None
        self.baseline_demand_total = None

        # PV: componenti e totale MV-equivalent.
        self.pv_mv_direct_by_mv_node = None
        self.pv_mv_direct_total = None
        self.pv_lv_aggregated_to_mv_node = None
        self.pv_lv_aggregated_total = None
        self.pv_generation_by_mv_node = None          # totale MV-equivalent
        self.pv_generation_total = None               # totale MV-equivalent

        # HP: componenti e totale MV-equivalent.
        self.hp_mv_direct_by_mv_node = None
        self.hp_mv_direct_total = None
        self.hp_lv_aggregated_to_mv_node = None
        self.hp_lv_aggregated_total = None
        self.hp_baseline_by_mv_node = None             # totale MV-equivalent
        self.hp_baseline_total = None                  # totale MV-equivalent

        # BESS: baseline nulla, ma allocation utile per FFOR.
        self.bess_mv_direct_allocation = None
        self.bess_lv_aggregated_to_mv_allocation = None
        self.bess_allocation = None                    # MV direct + LV aggregated

        # Baseline netta MV-equivalent.
        self.net_baseline_by_mv_node = None
        self.net_baseline_total = None

        self.node_capacity_summary = None

        if self.data_year not in [2030, 2040, 2050]:
            raise ValueError("data_year deve essere 2030, 2040 o 2050.")

    # ══════════════════════════════════════════════════════════════
    # METODI PUBBLICI
    # ══════════════════════════════════════════════════════════════

    def load_grid(self):
        """Carica nodi e linee MV e costruisce la mappa MV_osmid ↔ LV_grid."""
        nodes_path = self.geo_mv_path / f"{self.grid_name}_nodes"
        edges_path = self.geo_mv_path / f"{self.grid_name}_edges"

        self.nodes = self._geojson_to_dataframe(str(nodes_path))
        self.edges = self._geojson_to_dataframe(str(edges_path))

        connected_lv = self.nodes[self.nodes["lv_grid"].astype(str).ne("-1")].copy()
        self.mv_to_lv = dict(
            zip(connected_lv["osmid"].astype(str), connected_lv["lv_grid"].astype(str))
        )
        self.lv_to_mv = {lv_grid: mv_osmid for mv_osmid, lv_grid in self.mv_to_lv.items()}
        return self.nodes, self.edges

    def build_baseline_demand(self):
        """
        Costruisce domanda convenzionale MV-equivalent in kW.

        - Nodi MV puri (lv_grid == -1): MV_load_profile × el_dmd del nodo MV.
        - Nodi MV/LV (lv_grid != -1): aggregazione dei nodi LV a valle.

        Questa domanda è non flessibile ed esclude HP ed EV.
        """
        if self.nodes is None:
            self.load_grid()

        base_path = self.zapparoli_path / "05_Demand" / str(self.data_year)
        mv_profile = self._read_single_row_profile(str(base_path / "MV_load_profile.csv"))
        self.time_columns = list(mv_profile.index)

        rows = {}

        for _, node in self.nodes.iterrows():
            mv_osmid = str(node["osmid"])
            if str(node["lv_grid"]) == "-1":
                peak_kw = float(node.get("el_dmd", 0.0) or 0.0) * 1000.0
                rows[mv_osmid] = mv_profile.astype(float) * peak_kw

        lv_profiles = self._build_all_connected_lv_demand(str(base_path))
        for mv_osmid, lv_grid in self.mv_to_lv.items():
            values = lv_profiles.get(str(lv_grid))
            if values is None:
                values = self._zero_series()
                print(f"Nessun profilo LV per {lv_grid}; zero al nodo MV {mv_osmid}.")
            rows[str(mv_osmid)] = values

        self.baseline_demand_by_mv_node = self._profile_dict_to_dataframe(rows)
        self.baseline_demand_total = self._sum_profile(self.baseline_demand_by_mv_node)
        return self.baseline_demand_by_mv_node

    def build_pv_generation(self):
        """
        Costruisce PV MV-equivalent in kW.

        Componenti:
          - pv_mv_direct_by_mv_node: PV direttamente connesso a nodi MV.
          - pv_lv_aggregated_to_mv_node: PV nei LV_grid downstream aggregato al nodo MV/LV.
          - pv_generation_by_mv_node: somma delle due componenti.
        """
        if self.nodes is None:
            self.load_grid()
        if self.time_columns is None:
            self._ensure_time_columns()

        self.build_pv_mv_direct()
        self.build_pv_lv_aggregated_to_mv()

        self.pv_generation_by_mv_node = self._combine_profile_dataframes(
            self.pv_mv_direct_by_mv_node,
            self.pv_lv_aggregated_to_mv_node,
        )
        self.pv_generation_total = self._sum_profile(self.pv_generation_by_mv_node)
        return self.pv_generation_by_mv_node

    def build_pv_mv_direct(self):
        """PV direttamente connesso ai nodi MV, da MV_generation.csv."""
        if self.nodes is None:
            self.load_grid()
        if self.time_columns is None:
            self._ensure_time_columns()

        base_path = self.zapparoli_path / "01_PV" / str(self.data_year)
        path = base_path / "MV_generation.csv"
        rows = {}

        if path.exists():
            mv_generation = pd.read_csv(str(path))
            mv_generation = mv_generation[mv_generation["MV_grid"].astype(str).eq(str(self.grid_name))]
            mv_generation = self._expand_monthly_representative_days(
                mv_generation, ["MV_grid", "MV_osmid"]
            )

            for _, row in mv_generation.iterrows():
                mv_osmid = str(row["MV_osmid"])
                rows[mv_osmid] = rows.get(mv_osmid, self._zero_series()) + row[self.time_columns].astype(float)

        self.pv_mv_direct_by_mv_node = self._profile_dict_to_dataframe(rows)
        self.pv_mv_direct_total = self._sum_profile(self.pv_mv_direct_by_mv_node)
        return self.pv_mv_direct_by_mv_node

    def build_pv_lv_aggregated_to_mv(self):
        """PV dei LV_grid downstream, aggregato alla sottostazione MV/LV."""
        if self.nodes is None:
            self.load_grid()
        if self.time_columns is None:
            self._ensure_time_columns()

        base_path = self.zapparoli_path / "01_PV" / str(self.data_year)
        path = base_path / "LV_generation.csv"
        rows = {}
        connected_lv = set(str(v) for v in self.mv_to_lv.values())

        if path.exists() and connected_lv:
            lv_generation = pd.read_csv(str(path))
            lv_generation["LV_grid"] = lv_generation["LV_grid"].astype(str)
            lv_generation = lv_generation[lv_generation["LV_grid"].isin(connected_lv)]
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

    def build_hp_baseline(self):
        """
        Costruisce HP baseline MV-equivalent in kW.

        Componenti:
          - hp_mv_direct_by_mv_node: HP direttamente connesse a nodi MV.
          - hp_lv_aggregated_to_mv_node: HP nei LV_grid downstream aggregate al nodo MV/LV.
          - hp_baseline_by_mv_node: somma delle due componenti.

        Modello termico baseline:
          T_in = T_set costante
          Q_heat(t) = H × max(T_set - T_out(t), 0)
          COP(t) = COP_0 + COP_1 Δ + COP_2 Δ², con Δ = T_set - T_out(t)
          P_HP(t) = Q_heat(t) / COP(t), clip [0, P_nom]

        Non viene applicato il filtro stagionale a 14 °C.
        """
        if self.nodes is None:
            self.load_grid()
        if self.time_columns is None:
            self._ensure_time_columns()

        self.build_hp_mv_direct()
        self.build_hp_lv_aggregated_to_mv()

        self.hp_baseline_by_mv_node = self._combine_profile_dataframes(
            self.hp_mv_direct_by_mv_node,
            self.hp_lv_aggregated_to_mv_node,
        )
        self.hp_baseline_total = self._sum_profile(self.hp_baseline_by_mv_node)
        return self.hp_baseline_by_mv_node

    def build_hp_mv_direct(self):
        """HP direttamente connesse ai nodi MV, da MV_heat_pump_allocation.csv."""
        if self.nodes is None:
            self.load_grid()
        if self.time_columns is None:
            self._ensure_time_columns()

        base_path = self.zapparoli_path / "03_HP" / str(self.data_year)
        temperature_profiles = self._load_temperature_profiles(base_path)
        rows = {}

        path = base_path / "MV_heat_pump_allocation.csv"
        if path.exists():
            mv_hp = pd.read_csv(str(path))
            mv_hp = mv_hp[mv_hp["MV_grid"].astype(str).eq(str(self.grid_name))]
            for _, hp in mv_hp.iterrows():
                mv_osmid = str(hp["MV_osmid"])
                values = self._hp_consumption_for_row(hp, temperature_profiles)
                rows[mv_osmid] = rows.get(mv_osmid, self._zero_series()) + values

        self.hp_mv_direct_by_mv_node = self._profile_dict_to_dataframe(rows)
        self.hp_mv_direct_total = self._sum_profile(self.hp_mv_direct_by_mv_node)
        return self.hp_mv_direct_by_mv_node

    def build_hp_lv_aggregated_to_mv(self):
        """HP dei LV_grid downstream, aggregate alla sottostazione MV/LV."""
        if self.nodes is None:
            self.load_grid()
        if self.time_columns is None:
            self._ensure_time_columns()

        base_path = self.zapparoli_path / "03_HP" / str(self.data_year)
        temperature_profiles = self._load_temperature_profiles(base_path)
        connected_lv = set(str(v) for v in self.mv_to_lv.values())
        rows = {}

        path = base_path / "LV_heat_pump_allocation.csv"
        if path.exists() and connected_lv:
            lv_hp = pd.read_csv(str(path))
            lv_hp["LV_grid"] = lv_hp["LV_grid"].astype(str)
            lv_hp = lv_hp[lv_hp["LV_grid"].isin(connected_lv)]
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

    def load_bess_data(self):
        """
        Carica BESS MV-equivalent.

        Output:
          - bess_mv_direct_allocation: BESS direttamente su nodi MV.
          - bess_lv_aggregated_to_mv_allocation: BESS downstream LV aggregate al nodo MV/LV.
          - bess_allocation: concatenazione delle due componenti.

        BESS baseline = 0; questi dati sono per la flessibilità.
        """
        if self.nodes is None:
            self.load_grid()

        self.load_bess_mv_direct()
        self.load_bess_lv_aggregated_to_mv()

        frames = []
        if self.bess_mv_direct_allocation is not None and not self.bess_mv_direct_allocation.empty:
            frames.append(self.bess_mv_direct_allocation)
        if self.bess_lv_aggregated_to_mv_allocation is not None and not self.bess_lv_aggregated_to_mv_allocation.empty:
            frames.append(self.bess_lv_aggregated_to_mv_allocation)

        if frames:
            self.bess_allocation = pd.concat(frames, ignore_index=True, sort=False)
        else:
            self.bess_allocation = pd.DataFrame(columns=self._bess_output_columns())
            print(f"Nessuna BESS MV-equivalent per la rete {self.grid_name}.")
        return self.bess_allocation

    def load_bess_mv_direct(self):
        """BESS direttamente connesse ai nodi MV."""
        base_path = self.zapparoli_path / "02_BESS" / str(self.data_year)
        path = base_path / "BESS_allocation_MV.csv"

        if not path.exists():
            self.bess_mv_direct_allocation = pd.DataFrame(columns=self._bess_output_columns())
            return self.bess_mv_direct_allocation

        df = pd.read_csv(str(path))
        filtered = df[df["MV_grid"].astype(str).eq(str(self.grid_name))].copy()
        if filtered.empty:
            self.bess_mv_direct_allocation = pd.DataFrame(columns=self._bess_output_columns())
        else:
            filtered["MV_grid"] = self.grid_name
            filtered["MV_osmid"] = filtered["MV_osmid"].astype(str)
            filtered["source_component"] = "MV_direct"
            filtered["source_lv_grid"] = ""
            filtered["bess_count"] = 1
            self.bess_mv_direct_allocation = filtered[self._bess_output_columns()]
        return self.bess_mv_direct_allocation

    def load_bess_lv_aggregated_to_mv(self):
        """BESS dei LV_grid downstream, aggregate alla sottostazione MV/LV."""
        if self.nodes is None:
            self.load_grid()

        base_path = self.zapparoli_path / "02_BESS" / str(self.data_year)
        path = base_path / "BESS_allocation_LV.csv"
        connected_lv = set(str(v) for v in self.mv_to_lv.values())

        if not path.exists() or not connected_lv:
            self.bess_lv_aggregated_to_mv_allocation = pd.DataFrame(columns=self._bess_output_columns())
            return self.bess_lv_aggregated_to_mv_allocation

        lv_bess = pd.read_csv(str(path))
        lv_bess["LV_grid"] = lv_bess["LV_grid"].astype(str)
        lv_bess = lv_bess[lv_bess["LV_grid"].isin(connected_lv)].copy()

        rows = []
        for lv_grid, group in lv_bess.groupby("LV_grid"):
            mv_osmid = self.lv_to_mv.get(str(lv_grid))
            if mv_osmid is None:
                continue

            capacity = float(group["Battery_capacity_kWh"].sum())
            power = float(group["Nominal_power_kW"].sum())
            count = int(len(group))
            eta_ch = self._weighted_average(group, "Charging_efficiency", "Battery_capacity_kWh")
            eta_dis = self._weighted_average(group, "Discharging_efficiency", "Battery_capacity_kWh")

            rows.append({
                "MV_grid": self.grid_name,
                "MV_osmid": str(mv_osmid),
                "Battery_capacity_kWh": capacity,
                "Nominal_power_kW": power,
                "Charging_efficiency": eta_ch,
                "Discharging_efficiency": eta_dis,
                "source_component": "LV_aggregated",
                "source_lv_grid": str(lv_grid),
                "bess_count": count,
            })

        self.bess_lv_aggregated_to_mv_allocation = pd.DataFrame(rows, columns=self._bess_output_columns())
        return self.bess_lv_aggregated_to_mv_allocation

    def build_net_baseline_by_node(self):
        """Costruisce P_base = demand_MVeq + HP_MVeq - PV_MVeq per nodo MV."""
        if self.baseline_demand_by_mv_node is None:
            self.build_baseline_demand()
        if self.pv_generation_by_mv_node is None:
            self.build_pv_generation()
        if self.hp_baseline_by_mv_node is None:
            self.build_hp_baseline()

        demand = self._profile_df_to_index(self.baseline_demand_by_mv_node)
        hp = self._profile_df_to_index(self.hp_baseline_by_mv_node)
        pv = self._profile_df_to_index(self.pv_generation_by_mv_node)

        all_nodes = pd.Index(self._all_mv_osmids(), dtype="object")
        demand = demand.reindex(all_nodes, fill_value=0.0)
        hp = hp.reindex(all_nodes, fill_value=0.0)
        pv = pv.reindex(all_nodes, fill_value=0.0)

        net = demand + hp - pv
        self.net_baseline_by_mv_node = self._profile_index_to_dataframe(net)
        self.net_baseline_total = net.sum(axis=0)
        return self.net_baseline_by_mv_node

    def build_node_capacity_summary(self):
        """Tabella capacità MV-equivalent: MV direct + LV aggregated + totale per nodo MV."""
        if self.nodes is None:
            self.load_grid()

        summary = pd.DataFrame([{
            "MV_grid": self.grid_name,
            "MV_osmid": str(node["osmid"]),
            "lv_grid": str(node["lv_grid"]),
            "has_lv_connection": str(node["lv_grid"]) != "-1",
            "baseline_el_dmd_MW": float(node.get("el_dmd", 0.0) or 0.0),
        } for _, node in self.nodes.iterrows()])

        summary = self._merge_node_summary(summary, self._pv_capacity_mv_direct_summary())
        summary = self._merge_node_summary(summary, self._pv_capacity_lv_aggregated_summary())
        summary = self._merge_node_summary(summary, self._hp_capacity_mv_direct_summary())
        summary = self._merge_node_summary(summary, self._hp_capacity_lv_aggregated_summary())

        if self.bess_allocation is None:
            self.load_bess_data()
        summary = self._merge_node_summary(summary, self._bess_capacity_summary("MV_direct"))
        summary = self._merge_node_summary(summary, self._bess_capacity_summary("LV_aggregated"))

        # Riempi componenti mancanti e crea totali MV-equivalent.
        component_cols = [
            "pv_count_mv_direct", "pv_p_installed_mv_direct_kW",
            "pv_count_lv_aggregated", "pv_p_installed_lv_aggregated_kW",
            "hp_count_mv_direct", "hp_nominal_power_mv_direct_kW",
            "hp_thermal_capacitance_mv_direct_kWh_per_K",
            "hp_thermal_conductivity_mv_direct_kW_per_K",
            "hp_count_lv_aggregated", "hp_nominal_power_lv_aggregated_kW",
            "hp_thermal_capacitance_lv_aggregated_kWh_per_K",
            "hp_thermal_conductivity_lv_aggregated_kW_per_K",
            "bess_count_mv_direct", "bess_capacity_mv_direct_kWh", "bess_nominal_power_mv_direct_kW",
            "bess_count_lv_aggregated", "bess_capacity_lv_aggregated_kWh", "bess_nominal_power_lv_aggregated_kW",
        ]
        for col in component_cols:
            if col not in summary.columns:
                summary[col] = 0.0
            summary[col] = summary[col].fillna(0.0)

        summary["pv_count"] = summary["pv_count_mv_direct"] + summary["pv_count_lv_aggregated"]
        summary["pv_p_installed_kW"] = summary["pv_p_installed_mv_direct_kW"] + summary["pv_p_installed_lv_aggregated_kW"]

        summary["hp_count"] = summary["hp_count_mv_direct"] + summary["hp_count_lv_aggregated"]
        summary["hp_nominal_power_kW"] = summary["hp_nominal_power_mv_direct_kW"] + summary["hp_nominal_power_lv_aggregated_kW"]
        summary["hp_thermal_capacitance_kWh_per_K"] = (
            summary["hp_thermal_capacitance_mv_direct_kWh_per_K"]
            + summary["hp_thermal_capacitance_lv_aggregated_kWh_per_K"]
        )
        summary["hp_thermal_conductivity_kW_per_K"] = (
            summary["hp_thermal_conductivity_mv_direct_kW_per_K"]
            + summary["hp_thermal_conductivity_lv_aggregated_kW_per_K"]
        )

        summary["bess_count"] = summary["bess_count_mv_direct"] + summary["bess_count_lv_aggregated"]
        summary["bess_capacity_kWh"] = summary["bess_capacity_mv_direct_kWh"] + summary["bess_capacity_lv_aggregated_kWh"]
        summary["bess_nominal_power_kW"] = summary["bess_nominal_power_mv_direct_kW"] + summary["bess_nominal_power_lv_aggregated_kW"]

        self.node_capacity_summary = summary.sort_values(
            "MV_osmid", key=lambda col: col.astype(str).map(self._natural_sort_key)
        )
        return self.node_capacity_summary

    def build_all_profiles(self):
        """Costruisce tutti i profili MV-equivalent in sequenza."""
        self.load_grid()
        self.build_baseline_demand()
        self.build_pv_generation()
        self.build_hp_baseline()
        self.load_bess_data()
        self.build_net_baseline_by_node()
        self.build_node_capacity_summary()
        return {
            "baseline_demand_kW": self.baseline_demand_total,
            "pv_generation_kW": self.pv_generation_total,
            "hp_baseline_kW": self.hp_baseline_total,
            "net_baseline_kW": self.net_baseline_total,
        }

    def save_outputs(self, output_dir=None, winter_day="01-15", summer_day="07-15"):
        """Salva CSV e figure MV-equivalent senza sovrascrivere la vecchia cartella."""
        if self.net_baseline_total is None:
            self.build_all_profiles()

        if output_dir is None:
            output_dir = os.path.join(os.getcwd(), "outputs", f"{self.grid_name}_profiles_mveq_{self.data_year}")
        os.makedirs(output_dir, exist_ok=True)

        profiles = {
            "baseline_demand": self.baseline_demand_total,
            "pv_mv_direct": self.pv_mv_direct_total,
            "pv_lv_aggregated": self.pv_lv_aggregated_total,
            "pv_generation": self.pv_generation_total,
            "hp_mv_direct": self.hp_mv_direct_total,
            "hp_lv_aggregated": self.hp_lv_aggregated_total,
            "hp_baseline": self.hp_baseline_total,
            "net_baseline": self.net_baseline_total,
        }

        for name, series in profiles.items():
            series.to_csv(os.path.join(output_dir, f"{name}_total.csv"), header=["value_kW"])
            self._plot_profile_set(
                series=series,
                name=name,
                ylabel="kW",
                output_path=os.path.join(output_dir, f"{name}_annual_winter_summer.png"),
                winter_day=winter_day,
                summer_day=summer_day,
            )

        by_node_outputs = {
            "baseline_demand_by_mv_node.csv": self.baseline_demand_by_mv_node,
            "pv_mv_direct_by_mv_node.csv": self.pv_mv_direct_by_mv_node,
            "pv_lv_aggregated_to_mv_node.csv": self.pv_lv_aggregated_to_mv_node,
            "pv_generation_by_mv_node.csv": self.pv_generation_by_mv_node,
            "hp_mv_direct_by_mv_node.csv": self.hp_mv_direct_by_mv_node,
            "hp_lv_aggregated_to_mv_node.csv": self.hp_lv_aggregated_to_mv_node,
            "hp_baseline_by_mv_node.csv": self.hp_baseline_by_mv_node,
            "net_baseline_by_mv_node.csv": self.net_baseline_by_mv_node,
        }
        for filename, df in by_node_outputs.items():
            df.to_csv(os.path.join(output_dir, filename), index=False)

        self._safe_to_csv(self.bess_mv_direct_allocation, os.path.join(output_dir, "bess_allocation_mv_direct.csv"))
        self._safe_to_csv(self.bess_lv_aggregated_to_mv_allocation, os.path.join(output_dir, "bess_allocation_lv_aggregated_to_mv.csv"))
        self._safe_to_csv(self.bess_allocation, os.path.join(output_dir, "bess_allocation_mveq.csv"))

        self.node_capacity_summary.to_csv(os.path.join(output_dir, "node_capacity_summary_mveq.csv"), index=False)
        self._write_manifest(output_dir)
        self._plot_network_topology(os.path.join(output_dir, "network_topology_mveq.png"))
        return output_dir

    def save_node_outputs(self, mv_osmid, output_dir=None, winter_day="01-15", summer_day="07-15"):
        """Salva profili e figure MV-equivalent per un singolo nodo MV."""
        if self.net_baseline_total is None:
            self.build_all_profiles()

        mv_osmid = str(mv_osmid)
        if output_dir is None:
            output_dir = os.path.join(
                os.getcwd(), "outputs", f"{self.grid_name}_profiles_mveq_{self.data_year}", f"node_{mv_osmid}"
            )
        os.makedirs(output_dir, exist_ok=True)

        profiles = {
            "baseline_demand": self._node_series(self.baseline_demand_by_mv_node, mv_osmid),
            "pv_mv_direct": self._node_series(self.pv_mv_direct_by_mv_node, mv_osmid),
            "pv_lv_aggregated": self._node_series(self.pv_lv_aggregated_to_mv_node, mv_osmid),
            "pv_generation": self._node_series(self.pv_generation_by_mv_node, mv_osmid),
            "hp_mv_direct": self._node_series(self.hp_mv_direct_by_mv_node, mv_osmid),
            "hp_lv_aggregated": self._node_series(self.hp_lv_aggregated_to_mv_node, mv_osmid),
            "hp_baseline": self._node_series(self.hp_baseline_by_mv_node, mv_osmid),
            "net_baseline": self._node_series(self.net_baseline_by_mv_node, mv_osmid),
        }

        for name, series in profiles.items():
            series.to_csv(os.path.join(output_dir, f"{name}_node_{mv_osmid}.csv"), header=["value_kW"])
            self._plot_profile_set(
                series=series,
                name=f"{name}_node_{mv_osmid}",
                ylabel="kW",
                output_path=os.path.join(output_dir, f"{name}_node_{mv_osmid}_annual_winter_summer.png"),
                winter_day=winter_day,
                summer_day=summer_day,
            )

        node_summary = self.node_capacity_summary[self.node_capacity_summary["MV_osmid"].astype(str).eq(mv_osmid)]
        node_summary.to_csv(os.path.join(output_dir, f"capacity_summary_mveq_node_{mv_osmid}.csv"), index=False)
        return output_dir

    # ══════════════════════════════════════════════════════════════
    # METODI PRIVATI — DOMANDA LV
    # ══════════════════════════════════════════════════════════════

    def _build_all_connected_lv_demand(self, base_path):
        """Aggrega domanda LV convenzionale pesata per el_dmd ai nodi MV/LV."""
        commercial_profiles = pd.read_csv(os.path.join(base_path, "Commercial_profiles.csv"))
        residential_profiles = pd.read_csv(os.path.join(base_path, "Residential_profiles.csv"))
        basicload_shares = pd.read_csv(os.path.join(base_path, "LV_basicload_shares.csv"))

        connected_lv = set(str(v) for v in self.mv_to_lv.values())
        basicload_shares["LV_grid"] = basicload_shares["LV_grid"].astype(str)
        basicload_shares = basicload_shares[basicload_shares["LV_grid"].isin(connected_lv)]
        lv_peak_maps = self._load_lv_peak_maps(connected_lv)

        result = {}
        for lv_grid, shares in basicload_shares.groupby("LV_grid"):
            bfs_code = int(str(lv_grid).split("-")[0])
            commercial = commercial_profiles[commercial_profiles["BFS_municipality_code"].eq(bfs_code)]
            residential = residential_profiles[residential_profiles["BFS_municipality_code"].eq(bfs_code)]
            if commercial.empty or residential.empty:
                print(f"Profilo comm/res mancante per BFS {bfs_code}.")
                continue

            commercial = commercial.iloc[0][self.time_columns].astype(float)
            residential = residential.iloc[0][self.time_columns].astype(float)
            peak_map = lv_peak_maps.get(str(lv_grid), {})
            total = self._zero_series()

            for _, share in shares.iterrows():
                lv_osmid = str(share["LV_osmid"])
                peak_kw = float(peak_map.get(lv_osmid, 0.0)) * 1000.0
                profile_pu = (
                    commercial * float(share["Commercial_demand_share"])
                    + residential * float(share["Residential_demand_share"])
                )
                total = total + profile_pu * peak_kw
            result[str(lv_grid)] = total
        return result

    def _load_lv_peak_maps(self, lv_grids):
        grids_path = str(self.zapparoli_path / "06_Grids")
        dict_path = os.path.join(grids_path, "dict_folder.json")
        zip_path = os.path.join(grids_path, "LV.zip")
        extracted_flat = self.geo_lv_path

        if not os.path.exists(dict_path):
            return self._load_lv_peak_maps_flat(lv_grids, extracted_flat)

        with open(dict_path, "r") as f:
            folder_dict = json.load(f)

        peak_maps = {}
        zip_ref = zipfile.ZipFile(zip_path, "r") if os.path.exists(zip_path) else None
        entries = {entry.filename for entry in zip_ref.infolist()} if zip_ref is not None else set()

        try:
            for lv_grid in lv_grids:
                municipality_code = str(lv_grid).split("-")[0]
                folder_name = folder_dict.get(municipality_code)
                if folder_name is None:
                    print(f"Nessuna cartella LV per municipio {municipality_code}.")
                    continue

                entry_name = f"LV/{folder_name}/{lv_grid}_nodes"
                extracted_path = os.path.join(grids_path, "geojson_data", "LV", folder_name, f"{lv_grid}_nodes")

                if zip_ref is not None and entry_name in entries:
                    with zip_ref.open(entry_name) as f:
                        geojson = json.load(f)
                elif os.path.exists(extracted_path):
                    with open(extracted_path, "r") as f:
                        geojson = json.load(f)
                else:
                    flat_path = self._find_lv_nodes_flat(lv_grid, extracted_flat)
                    if flat_path is None:
                        print(f"{lv_grid}_nodes non trovato.")
                        continue
                    with open(flat_path, "r") as f:
                        geojson = json.load(f)

                peak_maps[str(lv_grid)] = {
                    str(feat["properties"]["osmid"]): float(feat["properties"].get("el_dmd", 0.0) or 0.0)
                    for feat in geojson["features"]
                }
        finally:
            if zip_ref is not None:
                zip_ref.close()
        return peak_maps

    def _load_lv_peak_maps_flat(self, lv_grids, lv_root):
        index = {}
        for root, _, files in os.walk(str(lv_root)):
            for fname in files:
                if fname.endswith("_nodes") or "_nodes" in fname:
                    key = fname.replace("_nodes", "").replace(".geojson", "")
                    index[key] = os.path.join(root, fname)

        peak_maps = {}
        for lv_grid in lv_grids:
            path = index.get(str(lv_grid))
            if path is None:
                print(f"{lv_grid}_nodes non trovato in {lv_root}.")
                continue
            with open(path, "r") as f:
                geojson = json.load(f)
            peak_maps[str(lv_grid)] = {
                str(feat["properties"]["osmid"]): float(feat["properties"].get("el_dmd", 0.0) or 0.0)
                for feat in geojson["features"]
            }
        return peak_maps

    def _find_lv_nodes_flat(self, lv_grid, lv_root):
        for root, _, files in os.walk(str(lv_root)):
            for fname in files:
                base = fname.replace(".geojson", "").replace("_nodes", "")
                if base == str(lv_grid):
                    return os.path.join(root, fname)
        return None

    # ══════════════════════════════════════════════════════════════
    # METODI PRIVATI — HP
    # ══════════════════════════════════════════════════════════════

    def _load_temperature_profiles(self, hp_base_path):
        temperature_profiles = pd.read_csv(str(hp_base_path / "Temperature_profiles.csv"))
        return temperature_profiles.set_index("Temperature_profile_name")

    def _hp_consumption_for_row(self, hp, temperature_profiles):
        """
        Consumo elettrico baseline HP per una riga di allocation.

        Nessun filtro a 14 °C:
          Q(t) = H × max(T_set - T_out(t), 0)

        La capacità termica C non compare nella baseline stazionaria perché
        assumiamo T_in(t) = T_in(t-1) = T_set.
        """
        profile_name = hp["Temperature_profile_name"]
        if profile_name not in temperature_profiles.index:
            return self._zero_series()

        temp = temperature_profiles.loc[profile_name, self.time_columns].astype(float)
        delta = self.setpoint_c - temp

        thermal_kw = float(hp["Thermal_conductivity_kW/K"]) * delta.clip(lower=0.0)

        if {"COP_0", "COP_1", "COP_2"}.issubset(hp.index) and not pd.isna(hp.get("COP_0")):
            cop = (
                float(hp["COP_0"])
                + float(hp["COP_1"]) * delta
                + float(hp["COP_2"]) * delta * delta
            ).clip(lower=1.0)
        else:
            cop = pd.Series(float(hp["COP"]), index=self.time_columns)

        electric_kw = thermal_kw / cop
        return electric_kw.clip(lower=0.0, upper=float(hp["Nominal_power_kW"]))

    # ══════════════════════════════════════════════════════════════
    # METODI PRIVATI — CAPACITY SUMMARY
    # ══════════════════════════════════════════════════════════════

    def _pv_capacity_mv_direct_summary(self):
        path = self.zapparoli_path / "01_PV" / str(self.data_year) / "MV_P_installed.csv"
        if not path.exists():
            return pd.DataFrame(columns=["MV_osmid"])
        pv = pd.read_csv(str(path))
        pv = pv[pv["MV_grid"].astype(str).eq(str(self.grid_name))].copy()
        if pv.empty:
            return pd.DataFrame(columns=["MV_osmid"])
        pv["MV_osmid"] = pv["MV_osmid"].astype(str)
        return pv.groupby("MV_osmid", as_index=False).agg(
            pv_count_mv_direct=("P_installed_kW", "size"),
            pv_p_installed_mv_direct_kW=("P_installed_kW", "sum"),
        )

    def _pv_capacity_lv_aggregated_summary(self):
        path = self.zapparoli_path / "01_PV" / str(self.data_year) / "LV_P_installed.csv"
        connected_lv = set(str(v) for v in self.mv_to_lv.values())
        if not path.exists() or not connected_lv:
            return pd.DataFrame(columns=["MV_osmid"])
        pv = pd.read_csv(str(path))
        pv["LV_grid"] = pv["LV_grid"].astype(str)
        pv = pv[pv["LV_grid"].isin(connected_lv)].copy()
        if pv.empty:
            return pd.DataFrame(columns=["MV_osmid"])
        pv["MV_osmid"] = pv["LV_grid"].map(self.lv_to_mv).astype(str)
        return pv.groupby("MV_osmid", as_index=False).agg(
            pv_count_lv_aggregated=("P_installed_kW", "size"),
            pv_p_installed_lv_aggregated_kW=("P_installed_kW", "sum"),
        )

    def _hp_capacity_mv_direct_summary(self):
        path = self.zapparoli_path / "03_HP" / str(self.data_year) / "MV_heat_pump_allocation.csv"
        if not path.exists():
            return pd.DataFrame(columns=["MV_osmid"])
        hp = pd.read_csv(str(path))
        hp = hp[hp["MV_grid"].astype(str).eq(str(self.grid_name))].copy()
        if hp.empty:
            return pd.DataFrame(columns=["MV_osmid"])
        hp["MV_osmid"] = hp["MV_osmid"].astype(str)
        return hp.groupby("MV_osmid", as_index=False).agg(
            hp_count_mv_direct=("Nominal_power_kW", "size"),
            hp_nominal_power_mv_direct_kW=("Nominal_power_kW", "sum"),
            hp_thermal_capacitance_mv_direct_kWh_per_K=("Thermal_capacitance_KWh/K", "sum"),
            hp_thermal_conductivity_mv_direct_kW_per_K=("Thermal_conductivity_kW/K", "sum"),
        )

    def _hp_capacity_lv_aggregated_summary(self):
        path = self.zapparoli_path / "03_HP" / str(self.data_year) / "LV_heat_pump_allocation.csv"
        connected_lv = set(str(v) for v in self.mv_to_lv.values())
        if not path.exists() or not connected_lv:
            return pd.DataFrame(columns=["MV_osmid"])
        hp = pd.read_csv(str(path))
        hp["LV_grid"] = hp["LV_grid"].astype(str)
        hp = hp[hp["LV_grid"].isin(connected_lv)].copy()
        if hp.empty:
            return pd.DataFrame(columns=["MV_osmid"])
        hp["MV_osmid"] = hp["LV_grid"].map(self.lv_to_mv).astype(str)
        return hp.groupby("MV_osmid", as_index=False).agg(
            hp_count_lv_aggregated=("Nominal_power_kW", "size"),
            hp_nominal_power_lv_aggregated_kW=("Nominal_power_kW", "sum"),
            hp_thermal_capacitance_lv_aggregated_kWh_per_K=("Thermal_capacitance_KWh/K", "sum"),
            hp_thermal_conductivity_lv_aggregated_kW_per_K=("Thermal_conductivity_kW/K", "sum"),
        )

    def _bess_capacity_summary(self, source_component):
        if self.bess_allocation is None or self.bess_allocation.empty:
            return pd.DataFrame(columns=["MV_osmid"])
        df = self.bess_allocation[self.bess_allocation["source_component"].eq(source_component)].copy()
        if df.empty:
            return pd.DataFrame(columns=["MV_osmid"])
        suffix = "mv_direct" if source_component == "MV_direct" else "lv_aggregated"
        return df.groupby("MV_osmid", as_index=False).agg(**{
            f"bess_count_{suffix}": ("bess_count", "sum"),
            f"bess_capacity_{suffix}_kWh": ("Battery_capacity_kWh", "sum"),
            f"bess_nominal_power_{suffix}_kW": ("Nominal_power_kW", "sum"),
        })

    # ══════════════════════════════════════════════════════════════
    # METODI PRIVATI — UTILITÀ PROFILI / CSV
    # ══════════════════════════════════════════════════════════════

    def _expand_monthly_representative_days(self, df, id_columns):
        if df.empty:
            return pd.DataFrame(columns=id_columns + self._get_time_columns())

        expanded = {}
        time_columns = [c for c in df.columns if c not in id_columns]
        for col in time_columns:
            month_day, hour = col.split()
            month = int(month_day.split("-")[0])
            # Il dataset usa sempre anni di 365 giorni; non includere 29 febbraio.
            days_in_month = calendar.monthrange(2030, month)[1]
            for day in range(1, days_in_month + 1):
                expanded[f"{month:02d}-{day:02d} {hour}"] = df[col].values

        expanded_df = pd.DataFrame(expanded)
        for idx, col in enumerate(id_columns):
            expanded_df.insert(idx, col, df[col].values)
        self.time_columns = self._filter_time_columns([c for c in expanded_df.columns if c not in id_columns])
        return expanded_df[id_columns + self.time_columns]

    def _read_single_row_profile(self, path):
        with open(path, "r") as f:
            timestamps = f.readline().strip().split(",")
            values = f.readline().strip().split(",")
        full_series = pd.Series([float(v) for v in values], index=timestamps)
        filtered = self._filter_time_columns(timestamps)
        return full_series.loc[filtered]

    def _ensure_time_columns(self):
        demand_path = str(self.zapparoli_path / "05_Demand" / str(self.data_year) / "MV_load_profile.csv")
        self.time_columns = list(self._read_single_row_profile(demand_path).index)

    def _get_time_columns(self):
        if self.time_columns is None:
            self._ensure_time_columns()
        return self.time_columns

    def _filter_time_columns(self, columns):
        filtered = [
            col for col in columns
            if self.start_date <= self._parse_month_day(col) <= self.end_date
        ]
        return sorted(filtered, key=self._parse_month_day)

    def _zero_series(self):
        return pd.Series(0.0, index=self._get_time_columns())

    def _all_mv_osmids(self):
        if self.nodes is None:
            self.load_grid()
        return list(self.nodes["osmid"].astype(str))

    def _profile_row(self, mv_osmid, values):
        row = {"MV_grid": self.grid_name, "MV_osmid": str(mv_osmid)}
        row.update(values.astype(float).to_dict())
        return row

    def _profile_dict_to_dataframe(self, rows, include_all_nodes=True):
        if include_all_nodes:
            node_ids = self._all_mv_osmids()
        else:
            node_ids = list(rows.keys())
        out_rows = []
        for mv_osmid in node_ids:
            values = rows.get(str(mv_osmid), self._zero_series())
            out_rows.append(self._profile_row(str(mv_osmid), values))
        return pd.DataFrame(out_rows)

    def _profile_df_to_index(self, df):
        if df is None or df.empty:
            return pd.DataFrame(0.0, index=pd.Index(self._all_mv_osmids(), dtype="object"), columns=self.time_columns)
        out = df.copy()
        out["MV_osmid"] = out["MV_osmid"].astype(str)
        return out.set_index("MV_osmid")[self.time_columns].astype(float)

    def _profile_index_to_dataframe(self, df):
        rows = []
        for mv_osmid, values in df.iterrows():
            rows.append(self._profile_row(str(mv_osmid), values))
        return pd.DataFrame(rows)

    def _combine_profile_dataframes(self, *dfs):
        all_nodes = pd.Index(self._all_mv_osmids(), dtype="object")
        total = pd.DataFrame(0.0, index=all_nodes, columns=self.time_columns)
        for df in dfs:
            if df is None or df.empty:
                continue
            idx = self._profile_df_to_index(df).reindex(all_nodes, fill_value=0.0)
            total = total + idx[self.time_columns].astype(float)
        return self._profile_index_to_dataframe(total)

    def _sum_profile(self, df):
        if df is None or df.empty:
            return self._zero_series()
        return df[self.time_columns].astype(float).sum(axis=0)

    def _node_series(self, df, mv_osmid):
        if df is None or df.empty:
            return self._zero_series()
        row = df[df["MV_osmid"].astype(str).eq(str(mv_osmid))]
        if row.empty:
            return self._zero_series()
        return row.iloc[0][self.time_columns].astype(float)

    def _merge_node_summary(self, summary, node_data):
        if node_data is None or node_data.empty:
            return summary
        node_data = node_data.copy()
        node_data["MV_osmid"] = node_data["MV_osmid"].astype(str)
        return summary.merge(node_data, on="MV_osmid", how="left")

    def _weighted_average(self, df, value_col, weight_col):
        weights = df[weight_col].astype(float)
        values = df[value_col].astype(float)
        if float(weights.sum()) == 0.0:
            return float(values.mean())
        return float((values * weights).sum() / weights.sum())

    def _bess_output_columns(self):
        return [
            "MV_grid", "MV_osmid", "Battery_capacity_kWh", "Nominal_power_kW",
            "Charging_efficiency", "Discharging_efficiency", "source_component",
            "source_lv_grid", "bess_count",
        ]

    def _safe_to_csv(self, df, path):
        if df is None:
            df = pd.DataFrame()
        df.to_csv(path, index=False)

    def _write_manifest(self, output_dir):
        lines = [
            f"Grid: {self.grid_name}",
            f"Data year: {self.data_year}",
            "Output type: MV-equivalent profiles",
            "Sign convention: P > 0 consumption, P < 0 generation/injection",
            "Baseline formula: P_base = demand_MVeq + HP_MVeq - PV_MVeq",
            "BESS baseline: 0; BESS allocation saved for flexibility modeling",
            "HP temperature model: constant indoor setpoint, no 14°C seasonal filter",
            f"HP setpoint: {self.setpoint_c} °C",
            "PV: MV_generation + LV_generation aggregated to MV/LV nodes",
            "HP: MV_heat_pump_allocation + LV_heat_pump_allocation aggregated to MV/LV nodes",
            "BESS: BESS_allocation_MV + BESS_allocation_LV aggregated to MV/LV nodes",
            "Demand: MV pure nodes + LV conventional demand aggregated to MV/LV nodes",
        ]
        with open(os.path.join(output_dir, "00_manifest.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    # ══════════════════════════════════════════════════════════════
    # PLOT
    # ══════════════════════════════════════════════════════════════

    def _plot_network_topology(self, output_path):
        if self.nodes is None or self.edges is None:
            self.load_grid()
        if self.node_capacity_summary is None:
            self.build_node_capacity_summary()

        summary = self.node_capacity_summary.copy()
        summary["total_asset_power_kW"] = (
            summary["pv_p_installed_kW"]
            + summary["hp_nominal_power_kW"]
            + summary["bess_nominal_power_kW"]
        )
        asset_power = dict(zip(summary["MV_osmid"].astype(str), summary["total_asset_power_kW"]))

        fig, ax = plt.subplots(figsize=(11, 9), constrained_layout=True)

        for _, edge in self.edges.iterrows():
            coords = edge["geometry"].get("coordinates", [])
            if not coords:
                continue
            xs = [p[0] for p in coords]
            ys = [p[1] for p in coords]
            ax.plot(xs, ys, color="#7a7a7a", linewidth=0.8, alpha=0.8, zorder=1)

        node_df = self.nodes.copy()
        node_df["osmid"] = node_df["osmid"].astype(str)
        node_df["asset_power_kW"] = node_df["osmid"].map(asset_power).fillna(0.0)
        max_power = max(float(node_df["asset_power_kW"].max()), 1.0)
        node_sizes = 35.0 + 180.0 * (node_df["asset_power_kW"] / max_power)

        scatter = ax.scatter(
            node_df["x"].astype(float),
            node_df["y"].astype(float),
            s=node_sizes,
            c=node_df["asset_power_kW"],
            cmap="viridis",
            edgecolors="#202020",
            linewidths=0.6,
            zorder=3,
        )

        connected_lv = node_df[node_df["lv_grid"].astype(str).ne("-1")]
        if not connected_lv.empty:
            ax.scatter(
                connected_lv["x"].astype(float),
                connected_lv["y"].astype(float),
                s=45,
                marker="s",
                facecolors="none",
                edgecolors="#d95f02",
                linewidths=1.2,
                label="MV/LV connection",
                zorder=4,
            )

        source_nodes = node_df[node_df["source"].astype(str).str.lower().eq("true")]
        if not source_nodes.empty:
            ax.scatter(
                source_nodes["x"].astype(float),
                source_nodes["y"].astype(float),
                s=160,
                marker="*",
                color="#d62728",
                edgecolors="#202020",
                linewidths=0.6,
                label="Source (PCC)",
                zorder=5,
            )

        for _, node in node_df.iterrows():
            ax.annotate(
                str(node["osmid"]),
                (float(node["x"]), float(node["y"])),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=6,
                color="#202020",
                zorder=6,
            )

        cbar = fig.colorbar(scatter, ax=ax, shrink=0.78)
        cbar.set_label("MVeq PV + HP + BESS nominal power (kW)")
        ax.set_title(f"{self.grid_name} — Topologia rete MV, asset MV-equivalent")
        ax.set_xlabel("x [m, EPSG:2056]")
        ax.set_ylabel("y [m, EPSG:2056]")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.2)
        if not connected_lv.empty or not source_nodes.empty:
            ax.legend(loc="best")

        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        print(f"Topologia salvata: {output_path}")

    def _plot_profile_set(self, series, name, ylabel, output_path, winter_day, summer_day):
        x = pd.to_datetime([f"2030-{ts}" for ts in series.index], format="%Y-%m-%d %H:%M:%S")
        values = series.astype(float).values

        fig, axes = plt.subplots(3, 1, figsize=(14, 10), constrained_layout=True)
        axes[0].plot(x, values, linewidth=0.8)
        axes[0].set_title(f"{name.replace('_', ' ').title()} — profilo annuale")
        axes[0].set_ylabel(ylabel)
        axes[0].grid(True, alpha=0.3)

        for ax, day, title in [(axes[1], winter_day, "giorno invernale"), (axes[2], summer_day, "giorno estivo")]:
            day_cols = [c for c in series.index if c.startswith(day)]
            day_x = [self._parse_month_day(c).hour for c in day_cols]
            ax.plot(day_x, series.loc[day_cols].astype(float).values, marker="o", linewidth=1.5)
            ax.set_title(f"{name.replace('_', ' ').title()} — {title} ({day})")
            ax.set_xlabel("Ora")
            ax.set_ylabel(ylabel)
            ax.set_xticks(range(0, 24, 2))
            ax.grid(True, alpha=0.3)

        fig.savefig(output_path, dpi=160)
        plt.close(fig)

    # ══════════════════════════════════════════════════════════════
    # STATIC
    # ══════════════════════════════════════════════════════════════

    @staticmethod
    def _parse_month_day(value):
        raw = str(value).strip()
        if len(raw) == len("01-01 00:00:00"):
            return datetime.strptime(f"2030-{raw}", "%Y-%m-%d %H:%M:%S")
        return datetime.strptime(raw, "%Y-%m-%d %H:%M:%S")

    @staticmethod
    def _natural_sort_key(value):
        text = str(value)
        return (0, int(text)) if text.isdigit() else (1, text)

    @staticmethod
    def _geojson_to_dataframe(path):
        with open(path, "r") as f:
            geojson = json.load(f)
        rows = []
        for feature in geojson["features"]:
            row = dict(feature["properties"])
            row["geometry"] = feature["geometry"]
            rows.append(row)
        return pd.DataFrame(rows)


if __name__ == "__main__":
    loader = Grid20Loader(
        grid_name="20_0",
        data_year=2030,
        start_date="01-01 00:00:00",
        end_date="12-31 23:00:00",
        setpoint_c=20.0,
    )
    loader.build_all_profiles()
    output_dir = loader.save_outputs()
    node_output_dir = loader.save_node_outputs("122")
    print(f"Output in: {output_dir}")
    print(f"Output nodo 122 in: {node_output_dir}")
