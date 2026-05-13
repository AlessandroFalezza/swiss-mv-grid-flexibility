"""
grid_loader.py
Adattamento del Grid20Loader del compagno per il PC di Alessandro.

Differenze rispetto all'originale:
  - I path dei dataset vengono letti da config.py (SWISS_PDGS, ZAPPAROLI)
    invece di usare os.getcwd()
  - I file GeoJSON della rete MV non hanno estensione (niente .txt)
  - I file GeoJSON delle reti LV sono già estratti in
    SWISS_PDGS/grids/geojson_data/LV/ (niente LV.zip)
  - Bug COP corretto: usa Δ = T_set - T_out (lift termico), non T_out

Uso:
    from grid_loader import Grid20Loader
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

# ── Importa i path da config.py ────────────────────────────────────
from config import SWISS_PDGS, ZAPPAROLI, GRID_ID, DATA_YEAR


class Grid20Loader:
    """
    Loader per la rete MV 20_0 — adattato per i path di Alessandro.

    Legge la topologia da Swiss-PDGs (GEO_MV) e i dati DER
    da Zapparoli-Oneto (ZAPPAROLI). Tutti i path sono centralizzati
    in config.py.
    """

    def __init__(
        self,
        grid_name=GRID_ID,
        data_year=DATA_YEAR,
        start_date="01-01 00:00:00",
        end_date="12-31 23:00:00",
        setpoint_c=20.0,
    ):
        self.grid_name   = grid_name
        self.data_year   = data_year
        self.start_date  = self._parse_month_day(start_date)
        self.end_date    = self._parse_month_day(end_date)
        self.setpoint_c  = setpoint_c
        self.heating_limit_c = 14.0

        # ── Path radice dei due dataset ───────────────────────────
        self.swiss_pdgs_path = Path(SWISS_PDGS)
        self.zapparoli_path  = Path(ZAPPAROLI)

        # Cartelle specifiche
        self.geo_mv_path = self.swiss_pdgs_path / "grids" / "geojson_data" / "MV"
        self.geo_lv_path = self.swiss_pdgs_path / "grids" / "geojson_data" / "LV"

        # ── Stato interno ─────────────────────────────────────────
        self.nodes = None
        self.edges = None
        self.mv_to_lv    = {}
        self.time_columns = None

        self.baseline_demand_by_mv_node = None
        self.baseline_demand_total      = None
        self.pv_generation_by_mv_node   = None
        self.pv_generation_total        = None
        self.hp_baseline_by_mv_node     = None
        self.hp_baseline_total          = None
        self.bess_allocation            = None
        self.node_capacity_summary      = None

        if self.data_year not in [2030, 2040, 2050]:
            raise ValueError("data_year deve essere 2030, 2040 o 2050.")

    # ══════════════════════════════════════════════════════════════
    # METODI PUBBLICI
    # ══════════════════════════════════════════════════════════════

    def load_grid(self):
        """Carica nodi e linee della rete MV dal GeoJSON Swiss-PDGs."""
        # I file GeoJSON non hanno estensione nel dataset Swiss-PDGs
        nodes_path = self.geo_mv_path / f"{self.grid_name}_nodes"
        edges_path = self.geo_mv_path / f"{self.grid_name}_edges"

        self.nodes = self._geojson_to_dataframe(str(nodes_path))
        self.edges = self._geojson_to_dataframe(str(edges_path))

        connected_lv = self.nodes[self.nodes["lv_grid"].ne("-1")].copy()
        self.mv_to_lv = dict(
            zip(connected_lv["osmid"].astype(str), connected_lv["lv_grid"])
        )
        return self.nodes, self.edges

    def build_baseline_demand(self):
        """
        Costruisce la domanda baseline convenzionale in kW.
        - Nodi MV puri (lv_grid == -1): MV_load_profile × el_dmd
        - Nodi sottostazione MV/LV: somma profili LV pesati per el_dmd nodo
        """
        if self.nodes is None:
            self.load_grid()

        base_path = self.zapparoli_path / "05_Demand" / str(self.data_year)
        mv_profile = self._read_single_row_profile(
            str(base_path / "MV_load_profile.csv")
        )
        self.time_columns = list(mv_profile.index)

        rows = []

        # Nodi MV puri
        for _, node in self.nodes.iterrows():
            if str(node["lv_grid"]) == "-1":
                peak_kw = float(node["el_dmd"]) * 1000.0
                values  = mv_profile.astype(float) * peak_kw
                rows.append(self._profile_row(str(node["osmid"]), values))

        # Nodi sottostazione MV/LV
        lv_profiles = self._build_all_connected_lv_demand(str(base_path))
        for mv_osmid, lv_grid in self.mv_to_lv.items():
            values = lv_profiles.get(lv_grid)
            if values is None:
                values = pd.Series(0.0, index=self.time_columns)
                print(f"Nessun profilo LV per {lv_grid}; zero al nodo MV {mv_osmid}.")
            rows.append(self._profile_row(mv_osmid, values))

        self.baseline_demand_by_mv_node = pd.DataFrame(rows)
        self.baseline_demand_total = self._sum_profile(
            self.baseline_demand_by_mv_node
        )
        return self.baseline_demand_by_mv_node

    def build_pv_generation(self):
        """Costruisce la generazione PV in kW a livello di nodo MV."""
        if self.nodes is None:
            self.load_grid()

        base_path = self.zapparoli_path / "01_PV" / str(self.data_year)
        rows = {}

        mv_generation = pd.read_csv(str(base_path / "MV_generation.csv"))
        mv_generation = mv_generation[
            mv_generation["MV_grid"].eq(self.grid_name)
        ]
        mv_generation = self._expand_monthly_representative_days(
            mv_generation, ["MV_grid", "MV_osmid"]
        )

        for _, row in mv_generation.iterrows():
            mv_osmid = str(row["MV_osmid"])
            rows[mv_osmid] = (
                rows.get(mv_osmid, self._zero_series())
                + row[self.time_columns].astype(float)
            )

        self.pv_generation_by_mv_node = self._rows_dict_to_dataframe(rows)
        self.pv_generation_total = self._sum_profile(
            self.pv_generation_by_mv_node
        )
        return self.pv_generation_by_mv_node

    def build_hp_baseline(self):
        """
        Costruisce il consumo baseline delle HP in kW (solo riscaldamento).

        Modello corretto (Zapparoli et al. 2025, eq. 3, 6, 8):
          Δ(t)    = T_set - T_out(t)              lift termico [°C]
          Q(t)    = H × Δ(t)                      potenza termica [kW]
          COP(t)  = COP_0 + COP_1·Δ + COP_2·Δ²   dipende dal lift
          P_HP(t) = Q(t) / COP(t)                 potenza elettrica [kW]

        Attivo solo nei giorni con T_media < 14°C (stagione riscaldamento).
        """
        if self.nodes is None:
            self.load_grid()
        if self.time_columns is None:
            self._ensure_time_columns()

        base_path = self.zapparoli_path / "03_HP" / str(self.data_year)
        temperature_profiles = pd.read_csv(
            str(base_path / "Temperature_profiles.csv")
        )
        temperature_profiles = temperature_profiles.set_index(
            "Temperature_profile_name"
        )
        rows = {}

        mv_hp = pd.read_csv(str(base_path / "MV_heat_pump_allocation.csv"))
        mv_hp = mv_hp[mv_hp["MV_grid"].eq(self.grid_name)]
        for _, hp in mv_hp.iterrows():
            mv_osmid = str(hp["MV_osmid"])
            values = self._hp_consumption_for_row(hp, temperature_profiles)
            rows[mv_osmid] = (
                rows.get(mv_osmid, self._zero_series()) + values
            )

        self.hp_baseline_by_mv_node = self._rows_dict_to_dataframe(rows)
        self.hp_baseline_total = self._sum_profile(self.hp_baseline_by_mv_node)
        return self.hp_baseline_by_mv_node

    def load_bess_data(self):
        """Carica i parametri BESS per i nodi MV della rete."""
        base_path    = self.zapparoli_path / "02_BESS" / str(self.data_year)
        alloc_path   = base_path / "BESS_allocation_MV.csv"

        if not alloc_path.exists():
            print(f"File BESS_allocation_MV.csv non trovato in {base_path}.")
            self.bess_allocation = None
            return None

        df = pd.read_csv(str(alloc_path))
        filtered = df[df["MV_grid"].eq(self.grid_name)]

        if filtered.empty:
            print(f"Nessuna BESS per la rete {self.grid_name}.")
            self.bess_allocation = None
        else:
            self.bess_allocation = filtered
        return self.bess_allocation

    def build_node_capacity_summary(self):
        """Tabella riassuntiva PV + HP + BESS installati per nodo MV."""
        if self.nodes is None:
            self.load_grid()

        rows = []
        for _, node in self.nodes.iterrows():
            rows.append({
                "MV_grid"          : self.grid_name,
                "MV_osmid"         : str(node["osmid"]),
                "lv_grid"          : str(node["lv_grid"]),
                "has_lv_connection": str(node["lv_grid"]) != "-1",
                "baseline_el_dmd_MW": float(node.get("el_dmd", 0.0) or 0.0),
            })
        summary = pd.DataFrame(rows)

        # PV
        pv = pd.read_csv(str(
            self.zapparoli_path / "01_PV" / str(self.data_year)
            / "MV_P_installed.csv"
        ))
        pv = pv[pv["MV_grid"].eq(self.grid_name)]
        pv_summary = pv.groupby("MV_osmid", as_index=False).agg(
            pv_count=("P_installed_kW", "size"),
            pv_p_installed_kW=("P_installed_kW", "sum"),
        )
        summary = self._merge_node_summary(summary, pv_summary)

        # HP
        hp = pd.read_csv(str(
            self.zapparoli_path / "03_HP" / str(self.data_year)
            / "MV_heat_pump_allocation.csv"
        ))
        hp = hp[hp["MV_grid"].eq(self.grid_name)]
        hp_summary = hp.groupby("MV_osmid", as_index=False).agg(
            hp_count=("Nominal_power_kW", "size"),
            hp_nominal_power_kW=("Nominal_power_kW", "sum"),
            hp_thermal_capacitance_kWh_per_K=("Thermal_capacitance_KWh/K", "sum"),
            hp_thermal_conductivity_kW_per_K=("Thermal_conductivity_kW/K", "sum"),
        )
        summary = self._merge_node_summary(summary, hp_summary)

        # BESS
        if self.bess_allocation is None:
            self.load_bess_data()
        if self.bess_allocation is None or self.bess_allocation.empty:
            bess_summary = pd.DataFrame(columns=[
                "MV_osmid", "bess_count",
                "bess_capacity_kWh", "bess_nominal_power_kW"
            ])
        else:
            bess_summary = self.bess_allocation.groupby(
                "MV_osmid", as_index=False
            ).agg(
                bess_count=("Battery_capacity_kWh", "size"),
                bess_capacity_kWh=("Battery_capacity_kWh", "sum"),
                bess_nominal_power_kW=("Nominal_power_kW", "sum"),
            )
        summary = self._merge_node_summary(summary, bess_summary)

        cap_cols = [
            "pv_count", "pv_p_installed_kW",
            "hp_count", "hp_nominal_power_kW",
            "hp_thermal_capacitance_kWh_per_K",
            "hp_thermal_conductivity_kW_per_K",
            "bess_count", "bess_capacity_kWh", "bess_nominal_power_kW",
        ]
        for col in cap_cols:
            if col not in summary.columns:
                summary[col] = 0.0
            summary[col] = summary[col].fillna(0.0)

        self.node_capacity_summary = summary.sort_values(
            "MV_osmid",
            key=lambda col: col.astype(str).map(self._natural_sort_key)
        )
        return self.node_capacity_summary

    def build_all_profiles(self):
        """Costruisce tutti i profili in sequenza."""
        self.load_grid()
        self.build_baseline_demand()
        self.build_pv_generation()
        self.build_hp_baseline()
        self.load_bess_data()
        self.build_node_capacity_summary()
        return {
            "baseline_demand_kW": self.baseline_demand_total,
            "pv_generation_kW"  : self.pv_generation_total,
            "hp_baseline_kW"    : self.hp_baseline_total,
        }

    def save_outputs(
        self,
        output_dir=None,
        winter_day="01-15",
        summer_day="07-15"
    ):
        """Salva CSV e figure nella cartella output_dir."""
        if self.baseline_demand_total is None:
            self.build_all_profiles()

        if output_dir is None:
            output_dir = os.path.join(
                os.getcwd(), "outputs", "20_0_profiles"
            )
        os.makedirs(output_dir, exist_ok=True)

        profiles = {
            "baseline_demand": self.baseline_demand_total,
            "pv_generation"  : self.pv_generation_total,
            "hp_baseline"    : self.hp_baseline_total,
        }

        for name, series in profiles.items():
            series.to_csv(
                os.path.join(output_dir, f"{name}_total.csv"),
                header=["value_kW"]
            )
            self._plot_profile_set(
                series=series,
                name=name,
                ylabel="kW",
                output_path=os.path.join(
                    output_dir, f"{name}_annual_winter_summer.png"
                ),
                winter_day=winter_day,
                summer_day=summer_day,
            )

        self.baseline_demand_by_mv_node.to_csv(
            os.path.join(output_dir, "baseline_demand_by_mv_node.csv"),
            index=False
        )
        self.pv_generation_by_mv_node.to_csv(
            os.path.join(output_dir, "pv_generation_by_mv_node.csv"),
            index=False
        )
        self.hp_baseline_by_mv_node.to_csv(
            os.path.join(output_dir, "hp_baseline_by_mv_node.csv"),
            index=False
        )

        if self.bess_allocation is None:
            self.load_bess_data()
        bess_out = self.bess_allocation if self.bess_allocation is not None \
            else pd.DataFrame(columns=[
                "MV_grid", "MV_osmid", "Battery_capacity_kWh",
                "Nominal_power_kW", "Charging_efficiency",
                "Discharging_efficiency"
            ])
        bess_out.to_csv(
            os.path.join(output_dir, "bess_allocation_mv.csv"), index=False
        )

        if self.node_capacity_summary is None:
            self.build_node_capacity_summary()
        self.node_capacity_summary.to_csv(
            os.path.join(output_dir, "node_capacity_summary.csv"), index=False
        )
        self._plot_network_topology(
            os.path.join(output_dir, "network_topology.png")
        )
        return output_dir

    def save_node_outputs(
        self,
        mv_osmid,
        output_dir=None,
        winter_day="01-15",
        summer_day="07-15"
    ):
        """Salva profili e figure per un singolo nodo MV."""
        if self.baseline_demand_total is None:
            self.build_all_profiles()

        mv_osmid = str(mv_osmid)
        if output_dir is None:
            output_dir = os.path.join(
                os.getcwd(), "outputs", "20_0_profiles", f"node_{mv_osmid}"
            )
        os.makedirs(output_dir, exist_ok=True)

        profiles = {
            "baseline_demand": self._node_series(
                self.baseline_demand_by_mv_node, mv_osmid),
            "pv_generation"  : self._node_series(
                self.pv_generation_by_mv_node, mv_osmid),
            "hp_baseline"    : self._node_series(
                self.hp_baseline_by_mv_node, mv_osmid),
        }

        for name, series in profiles.items():
            series.to_csv(
                os.path.join(output_dir, f"{name}_node_{mv_osmid}.csv"),
                header=["value_kW"]
            )
            self._plot_profile_set(
                series=series,
                name=f"{name}_node_{mv_osmid}",
                ylabel="kW",
                output_path=os.path.join(
                    output_dir,
                    f"{name}_node_{mv_osmid}_annual_winter_summer.png"
                ),
                winter_day=winter_day,
                summer_day=summer_day,
            )

        if self.node_capacity_summary is None:
            self.build_node_capacity_summary()
        node_summary = self.node_capacity_summary[
            self.node_capacity_summary["MV_osmid"].astype(str).eq(mv_osmid)
        ]
        node_summary.to_csv(
            os.path.join(output_dir, f"capacity_summary_node_{mv_osmid}.csv"),
            index=False
        )
        return output_dir

    # ══════════════════════════════════════════════════════════════
    # METODI PRIVATI — DOMANDA LV
    # ══════════════════════════════════════════════════════════════

    def _build_all_connected_lv_demand(self, base_path):
        """Aggrega la domanda LV pesata per el_dmd ai nodi MV sottostazione."""
        commercial_profiles = pd.read_csv(
            os.path.join(base_path, "Commercial_profiles.csv")
        )
        residential_profiles = pd.read_csv(
            os.path.join(base_path, "Residential_profiles.csv")
        )
        basicload_shares = pd.read_csv(
            os.path.join(base_path, "LV_basicload_shares.csv")
        )

        connected_lv = set(self.mv_to_lv.values())
        basicload_shares = basicload_shares[
            basicload_shares["LV_grid"].isin(connected_lv)
        ]
        lv_peak_maps = self._load_lv_peak_maps(connected_lv)

        result = {}
        for lv_grid, shares in basicload_shares.groupby("LV_grid"):
            bfs_code  = int(str(lv_grid).split("-")[0])
            commercial = commercial_profiles[
                commercial_profiles["BFS_municipality_code"].eq(bfs_code)
            ]
            residential = residential_profiles[
                residential_profiles["BFS_municipality_code"].eq(bfs_code)
            ]
            if commercial.empty or residential.empty:
                print(f"Profilo comm/res mancante per BFS {bfs_code}.")
                continue

            commercial  = commercial.iloc[0][self.time_columns].astype(float)
            residential = residential.iloc[0][self.time_columns].astype(float)
            peak_map    = lv_peak_maps.get(str(lv_grid), {})
            total       = pd.Series(0.0, index=self.time_columns)

            for _, share in shares.iterrows():
                lv_osmid  = str(share["LV_osmid"])
                peak_kw   = float(peak_map.get(lv_osmid, 0.0)) * 1000.0
                profile_pu = (
                    commercial  * float(share["Commercial_demand_share"])
                    + residential * float(share["Residential_demand_share"])
                )
                total = total + profile_pu * peak_kw
            result[str(lv_grid)] = total
        return result

    def _load_lv_peak_maps(self, lv_grids):
        """
        Carica el_dmd per ogni nodo LV dalle reti già estratte in
        SWISS_PDGS/grids/geojson_data/LV/ (struttura con sottocartelle
        per regione/urbanizzazione come nel dataset Swiss-PDGs originale).

        Fallback: usa dict_folder.json + LV.zip se presente nella cartella
        06_Grids del dataset Zapparoli.
        """
        # Prova prima con LV.zip nel dataset Zapparoli (come nell'originale)
        grids_path = str(self.zapparoli_path / "06_Grids")
        dict_path  = os.path.join(grids_path, "dict_folder.json")
        zip_path   = os.path.join(grids_path, "LV.zip")

        # Fallback: LV GeoJSON estratti in Swiss-PDGs
        # (struttura piatta: tutti i file _nodes direttamente in LV/)
        extracted_flat = self.geo_lv_path

        if not os.path.exists(dict_path):
            # dict_folder.json non trovato → usa flat search in Swiss-PDGs LV
            return self._load_lv_peak_maps_flat(lv_grids, extracted_flat)

        with open(dict_path, "r") as f:
            folder_dict = json.load(f)

        peak_maps = {}
        zip_ref = (
            zipfile.ZipFile(zip_path, "r")
            if os.path.exists(zip_path)
            else None
        )
        entries = (
            {entry.filename for entry in zip_ref.infolist()}
            if zip_ref is not None
            else set()
        )

        try:
            for lv_grid in lv_grids:
                municipality_code = str(lv_grid).split("-")[0]
                folder_name = folder_dict.get(municipality_code)
                if folder_name is None:
                    print(f"Nessuna cartella LV per municipio {municipality_code}.")
                    continue

                entry_name     = f"LV/{folder_name}/{lv_grid}_nodes"
                extracted_path = os.path.join(
                    grids_path, "geojson_data", "LV",
                    folder_name, f"{lv_grid}_nodes"
                )

                if zip_ref is not None and entry_name in entries:
                    with zip_ref.open(entry_name) as f:
                        geojson = json.load(f)
                elif os.path.exists(extracted_path):
                    with open(extracted_path, "r") as f:
                        geojson = json.load(f)
                else:
                    # Fallback: cerca nel Swiss-PDGs estratto (struttura piatta)
                    flat_path = self._find_lv_nodes_flat(
                        lv_grid, extracted_flat
                    )
                    if flat_path is None:
                        print(f"{lv_grid}_nodes non trovato.")
                        continue
                    with open(flat_path, "r") as f:
                        geojson = json.load(f)

                peak_maps[str(lv_grid)] = {
                    str(feat["properties"]["osmid"]): float(
                        feat["properties"].get("el_dmd", 0.0) or 0.0
                    )
                    for feat in geojson["features"]
                }
        finally:
            if zip_ref is not None:
                zip_ref.close()

        return peak_maps

    def _load_lv_peak_maps_flat(self, lv_grids, lv_root):
        """
        Ricerca ricorsiva dei file _nodes nelle sottocartelle di Swiss-PDGs LV.
        Usato come fallback quando dict_folder.json non è disponibile.
        """
        # Costruisce un indice nome→path una volta sola
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
                str(feat["properties"]["osmid"]): float(
                    feat["properties"].get("el_dmd", 0.0) or 0.0
                )
                for feat in geojson["features"]
            }
        return peak_maps

    def _find_lv_nodes_flat(self, lv_grid, lv_root):
        """Cerca ricorsivamente il file {lv_grid}_nodes in lv_root."""
        for root, _, files in os.walk(str(lv_root)):
            for fname in files:
                base = fname.replace(".geojson", "").replace("_nodes", "")
                if base == str(lv_grid):
                    return os.path.join(root, fname)
        return None

    # ══════════════════════════════════════════════════════════════
    # METODI PRIVATI — MODELLO HP (corretto)
    # ══════════════════════════════════════════════════════════════

    def _hp_consumption_for_row(self, hp, temperature_profiles):
        """
        Calcola il consumo elettrico baseline per un nodo HP.

        Modello corretto:
          Q(t)   = H × max(T_set - T_out, 0)          [kW termici]
          Δ(t)   = T_set - T_out(t)                    [lift termico]
          COP(t) = COP_0 + COP_1·Δ + COP_2·Δ²         [eq. 8 paper]
          P(t)   = Q(t) / COP(t), clip [0, P_nom]
        Attivo solo nei giorni con T_media_giornaliera < 14°C.
        """
        profile_name = hp["Temperature_profile_name"]
        if profile_name not in temperature_profiles.index:
            return self._zero_series()

        temp = temperature_profiles.loc[
            profile_name, self.time_columns
        ].astype(float)

        # Filtro stagione riscaldamento: giorni con T_media < 14°C
        day_keys = pd.Index(temp.index).str.slice(0, 5)
        daily_mean_temp = temp.groupby(day_keys).transform("mean")
        heating_active  = daily_mean_temp < self.heating_limit_c

        # Potenza termica: Q = H × ΔT (moltiplica per conduttanza)
        thermal_kw = (
            float(hp["Thermal_conductivity_kW/K"])
            * (self.setpoint_c - temp).clip(lower=0.0)
            * heating_active.astype(float)
        )

        # COP funzione del lift Δ = T_set - T_out (eq. 8 del paper)
        # NON della temperatura esterna da sola
        if {"COP_0", "COP_1", "COP_2"}.issubset(hp.index):
            delta = self.setpoint_c - temp          # lift termico [°C]
            cop   = (
                float(hp["COP_0"])
                + float(hp["COP_1"]) * delta
                + float(hp["COP_2"]) * delta * delta
            ).clip(lower=1.0)
        else:
            cop = pd.Series(float(hp["COP"]), index=self.time_columns)

        electric_kw = thermal_kw / cop
        return electric_kw.clip(
            lower=0.0, upper=float(hp["Nominal_power_kW"])
        )

    # ══════════════════════════════════════════════════════════════
    # METODI PRIVATI — UTILITÀ
    # ══════════════════════════════════════════════════════════════

    def _expand_monthly_representative_days(self, df, id_columns):
        if df.empty:
            return pd.DataFrame(columns=id_columns + self._get_time_columns())

        expanded     = {}
        time_columns = [c for c in df.columns if c not in id_columns]
        for col in time_columns:
            month_day, hour = col.split()
            month = int(month_day.split("-")[0])
            days_in_month = calendar.monthrange(2030, month)[1]
            for day in range(1, days_in_month + 1):
                expanded[f"{month:02d}-{day:02d} {hour}"] = df[col].values

        expanded_df = pd.DataFrame(expanded)
        for idx, col in enumerate(id_columns):
            expanded_df.insert(idx, col, df[col].values)
        self.time_columns = self._filter_time_columns(
            [c for c in expanded_df.columns if c not in id_columns]
        )
        return expanded_df[id_columns + self.time_columns]

    def _read_single_row_profile(self, path):
        with open(path, "r") as f:
            timestamps = f.readline().strip().split(",")
            values     = f.readline().strip().split(",")
        full_series = pd.Series(
            [float(v) for v in values], index=timestamps
        )
        filtered = self._filter_time_columns(timestamps)
        return full_series.loc[filtered]

    def _ensure_time_columns(self):
        demand_path = str(
            self.zapparoli_path / "05_Demand"
            / str(self.data_year) / "MV_load_profile.csv"
        )
        self.time_columns = list(
            self._read_single_row_profile(demand_path).index
        )

    def _get_time_columns(self):
        if self.time_columns is None:
            self._ensure_time_columns()
        return self.time_columns

    def _filter_time_columns(self, columns):
        filtered = [
            col for col in columns
            if self.start_date
            <= self._parse_month_day(col)
            <= self.end_date
        ]
        return sorted(filtered, key=self._parse_month_day)

    def _zero_series(self):
        return pd.Series(0.0, index=self._get_time_columns())

    def _profile_row(self, mv_osmid, values):
        row = {"MV_grid": self.grid_name, "MV_osmid": str(mv_osmid)}
        row.update(values.astype(float).to_dict())
        return row

    def _rows_dict_to_dataframe(self, rows):
        return pd.DataFrame([
            self._profile_row(mv_osmid, values)
            for mv_osmid, values in rows.items()
        ])

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
        asset_power = dict(
            zip(summary["MV_osmid"].astype(str),
                summary["total_asset_power_kW"])
        )

        fig, ax = plt.subplots(figsize=(11, 9), constrained_layout=True)

        for _, edge in self.edges.iterrows():
            coords = edge["geometry"].get("coordinates", [])
            if not coords:
                continue
            xs = [p[0] for p in coords]
            ys = [p[1] for p in coords]
            ax.plot(xs, ys, color="#7a7a7a", linewidth=0.8,
                    alpha=0.8, zorder=1)

        node_df = self.nodes.copy()
        node_df["osmid"] = node_df["osmid"].astype(str)
        node_df["asset_power_kW"] = (
            node_df["osmid"].map(asset_power).fillna(0.0)
        )
        max_power   = max(float(node_df["asset_power_kW"].max()), 1.0)
        node_sizes  = 35.0 + 180.0 * (
            node_df["asset_power_kW"] / max_power
        )

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
                s=45, marker="s",
                facecolors="none", edgecolors="#d95f02",
                linewidths=1.2, label="MV/LV connection", zorder=4,
            )

        source_nodes = node_df[
            node_df["source"].astype(str).str.lower().eq("true")
        ]
        if not source_nodes.empty:
            ax.scatter(
                source_nodes["x"].astype(float),
                source_nodes["y"].astype(float),
                s=160, marker="*", color="#d62728",
                edgecolors="#202020", linewidths=0.6,
                label="Source (PCC)", zorder=5,
            )

        for _, node in node_df.iterrows():
            ax.annotate(
                str(node["osmid"]),
                (float(node["x"]), float(node["y"])),
                xytext=(3, 3), textcoords="offset points",
                fontsize=6, color="#202020", zorder=6,
            )

        cbar = fig.colorbar(scatter, ax=ax, shrink=0.78)
        cbar.set_label("PV + HP + BESS nominal power (kW)")
        ax.set_title(f"{self.grid_name} — Topologia rete MV")
        ax.set_xlabel("x [m, EPSG:2056]")
        ax.set_ylabel("y [m, EPSG:2056]")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.2)
        if not connected_lv.empty or not source_nodes.empty:
            ax.legend(loc="best")

        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        print(f"Topologia salvata: {output_path}")

    def _plot_profile_set(
        self, series, name, ylabel, output_path, winter_day, summer_day
    ):
        x = pd.to_datetime(
            [f"2030-{ts}" for ts in series.index],
            format="%Y-%m-%d %H:%M:%S"
        )
        values = series.astype(float).values

        fig, axes = plt.subplots(
            3, 1, figsize=(14, 10), constrained_layout=True
        )

        axes[0].plot(x, values, linewidth=0.8)
        axes[0].set_title(
            f"{name.replace('_', ' ').title()} — profilo annuale"
        )
        axes[0].set_ylabel(ylabel)
        axes[0].grid(True, alpha=0.3)

        for ax, day, title in [
            (axes[1], winter_day, "giorno invernale"),
            (axes[2], summer_day, "giorno estivo"),
        ]:
            day_cols = [c for c in series.index if c.startswith(day)]
            day_x    = [self._parse_month_day(c).hour for c in day_cols]
            ax.plot(
                day_x,
                series.loc[day_cols].astype(float).values,
                marker="o", linewidth=1.5,
            )
            ax.set_title(
                f"{name.replace('_', ' ').title()} — {title} ({day})"
            )
            ax.set_xlabel("Ora")
            ax.set_ylabel(ylabel)
            ax.set_xticks(range(0, 24, 2))
            ax.grid(True, alpha=0.3)

        fig.savefig(output_path, dpi=160)
        plt.close(fig)

    @staticmethod
    def _parse_month_day(value):
        # Usa 2030 come anno base (coerente con il dataset)
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


# ══════════════════════════════════════════════════════════════════
# TEST
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    loader = Grid20Loader(
        grid_name="20_0",
        data_year=2030,
        start_date="01-01 00:00:00",
        end_date="12-31 23:00:00",
        setpoint_c=20.0,
    )
    loader.build_all_profiles()
    output_dir      = loader.save_outputs()
    node_output_dir = loader.save_node_outputs("122")
    print(f"Output in: {output_dir}")
    print(f"Output nodo 122 in: {node_output_dir}")