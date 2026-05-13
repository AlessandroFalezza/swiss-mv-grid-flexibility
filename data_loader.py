"""
data_loader.py
Carica i dati DER per la rete 20_0 dal dataset Zapparoli & Oneto (2025).
Adattato da 08_Data_loader.py degli autori.

Uso:
    from data_loader import load_all
    data = load_all()   # carica tutto per GRID_ID e DATA_YEAR da config.py
"""

import calendar
import json
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from config import (BESS_DIR, DATA_YEAR, DEM_DIR, GEO_MV, GRID_ID, HP_DIR,
                    PP_MV, PV_DIR)

   # ══════════════════════════════════════════════════════════════════
# PLOT — Profilo annuale nodo 122 (sottostazione LV con PV + HP)
# ══════════════════════════════════════════════════════════════════

def plot_node_annual_profile(data, osmid=122):
    """
    Visualizza il profilo di consumo/generazione annuale di un nodo
    collegato a una rete LV, scomponendo i contributi:
      - Carico passivo (P_load)
      - Pompa di calore (P_HP)
      - Fotovoltaico (P_PV, negativo = generazione)
      - Potenza netta (P_net = somma dei tre)

    Mostra tre pannelli:
      1. Anno intero (profilo orario)
      2. Settimana invernale (15 dic ± 3 giorni)
      3. Settimana estiva  (15 lug ± 3 giorni)
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import numpy as np

    # ── Ricostruisci i componenti separati ───────────────────────
    baseline     = data['baseline']
    pv_gen       = data['pv_gen']       # kW
    hp_params    = data['hp_params']
    node_profiles= data['node_profiles']
    el_dmd       = data['el_dmd']
    idx          = data['time_index']

    el_dmd_k  = el_dmd[osmid]
    prof_k    = node_profiles[osmid]    # p.u.
    P_load    = el_dmd_k * prof_k       # MW

    # HP
    if osmid in hp_params.index:
        from data_loader import load_hp_params, compute_hp_baseline
        hp_p, temp_d = load_hp_params()
        hp_b = compute_hp_baseline(hp_p, temp_d)
        P_hp = hp_b[osmid].values / 1000 if osmid in hp_b.columns else np.zeros(8760)
    else:
        P_hp = np.zeros(8760)

    # PV (kW → MW)
    if osmid in pv_gen.index:
        P_pv = pv_gen.loc[osmid].values / 1000
    else:
        P_pv = np.zeros(8760)

    P_net = P_load + P_hp - P_pv

    # ── Setup figure ──────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        f'Profilo annuale — Nodo {osmid}  '
        f'(Sottostazione LV {data["nodes"][data["nodes"]["osmid"].astype(int)==osmid]["lv_grid"].values[0]})\n'
        f'el_dmd picco = {el_dmd_k*1000:.0f} kW  |  '
        f'PV = {pv_gen.loc[osmid].max():.0f} kW max  |  Scenario 2030',
        fontsize=13, fontweight='bold', y=0.98
    )

    colors = {
        'load': '#4878CF',
        'hp'  : '#D65F5F',
        'pv'  : '#F5A623',
        'net' : '#2CA02C',
    }

    # ── Pannello 1: Anno intero ────────────────────────────────────
    ax1 = fig.add_subplot(2, 1, 1)

    ax1.fill_between(idx, 0, P_load,
                     alpha=0.35, color=colors['load'],
                     label='Carico passivo')
    ax1.fill_between(idx, P_load, P_load + P_hp,
                     alpha=0.35, color=colors['hp'],
                     label='Pompa di calore')
    ax1.fill_between(idx, 0, -P_pv,
                     alpha=0.35, color=colors['pv'],
                     label='PV (generazione)')
    ax1.plot(idx, P_net,
             color=colors['net'], linewidth=0.7, alpha=0.9,
             label='P netta al nodo (P_net)')
    ax1.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)

    ax1.set_ylabel('Potenza [MW]', fontsize=11)
    ax1.set_title('Profilo orario annuale (8760 ore)', fontsize=11)
    ax1.legend(loc='upper right', fontsize=9, ncol=4)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.set_xlim(idx[0], idx[-1])
    ax1.grid(True, alpha=0.3)

    # Evidenzia le zone negative (nodo che esporta)
    ax1.fill_between(idx, P_net, 0,
                     where=(P_net < 0),
                     alpha=0.25, color='gold',
                     label='Esportazione netta')

    # ── Pannello 2: Settimana invernale (dic 13–19) ───────────────
    ax2 = fig.add_subplot(2, 2, 3)
    mask_w = (idx >= '2030-12-13') & (idx <= '2030-12-19')

    ax2.stackplot(idx[mask_w],
                  P_load[mask_w], P_hp[mask_w],
                  labels=['Carico', 'HP'],
                  colors=[colors['load'], colors['hp']],
                  alpha=0.6)
    ax2.fill_between(idx[mask_w], 0, -P_pv[mask_w],
                     alpha=0.5, color=colors['pv'], label='PV')
    ax2.plot(idx[mask_w], P_net[mask_w],
             color=colors['net'], linewidth=2, label='P netta', zorder=5)
    ax2.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)

    ax2.set_ylabel('Potenza [MW]', fontsize=10)
    ax2.set_title('Settimana invernale (13–19 dic)', fontsize=11)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d %b\n%H:%M'))
    ax2.xaxis.set_major_locator(mdates.DayLocator())
    ax2.legend(fontsize=8, loc='upper right')
    ax2.grid(True, alpha=0.3)

    # ── Pannello 3: Settimana estiva (lug 13–19) ─────────────────
    ax3 = fig.add_subplot(2, 2, 4)
    mask_s = (idx >= '2030-07-13') & (idx <= '2030-07-19')

    ax3.stackplot(idx[mask_s],
                  P_load[mask_s], P_hp[mask_s],
                  labels=['Carico', 'HP'],
                  colors=[colors['load'], colors['hp']],
                  alpha=0.6)
    ax3.fill_between(idx[mask_s], 0, -P_pv[mask_s],
                     alpha=0.5, color=colors['pv'], label='PV')
    ax3.plot(idx[mask_s], P_net[mask_s],
             color=colors['net'], linewidth=2, label='P netta', zorder=5)
    ax3.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax3.fill_between(idx[mask_s], P_net[mask_s], 0,
                     where=(P_net[mask_s] < 0),
                     alpha=0.3, color='gold', label='Esportazione')

    ax3.set_ylabel('Potenza [MW]', fontsize=10)
    ax3.set_title('Settimana estiva (13–19 lug)', fontsize=11)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%d %b\n%H:%M'))
    ax3.xaxis.set_major_locator(mdates.DayLocator())
    ax3.legend(fontsize=8, loc='upper right')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('node_122_annual_profile.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Salvato: node_122_annual_profile.png")


# ══════════════════════════════════════════════════════════════════
# 1. TOPOLOGIA DELLA RETE
# ══════════════════════════════════════════════════════════════════

def load_network_topology(grid=GRID_ID):
    """
    Carica nodi e linee della rete MV dal GeoJSON Swiss-PDGs.

    Returns
    -------
    nodes : GeoDataFrame  — 51 nodi con osmid, el_dmd, lv_grid, consumers, voltage
    edges : GeoDataFrame  — 50 linee con u, v, r, x, b, s_nom, load, OHL
    """
    nodes = gpd.read_file(GEO_MV / f"{grid}_nodes")
    edges = gpd.read_file(GEO_MV / f"{grid}_edges")
    nodes['osmid'] = nodes['osmid'].astype(int)
    edges['u'] = edges['u'].astype(int)
    edges['v'] = edges['v'].astype(int)
    return nodes, edges


def get_consumer_nodes(nodes):
    """
    Restituisce il dict {osmid: el_dmd_MW} per i nodi consumer.
    Esclude PCC (source=True) e nodi passivi (consumers=False).
    """
    cons = nodes[nodes['consumers'] == 'True']
    return dict(zip(cons['osmid'].astype(int), cons['el_dmd'].astype(float)))


# ══════════════════════════════════════════════════════════════════
# 2. PROFILI DI CARICO — differenziati per tipo di nodo
# ══════════════════════════════════════════════════════════════════

def _load_mv_profile(year=DATA_YEAR):
    """
    Legge MV_load_profile.csv → Series 8760h p.u.
    Usato per i nodi MV puri (lv_grid == '-1').
    """
    path = DEM_DIR / str(year) / 'MV_load_profile.csv'
    with open(path, 'r') as f:
        lines = f.readlines()
    timestamps = lines[0].strip().split(',')
    values     = [float(v) for v in lines[1].strip().split(',')]
    idx = pd.to_datetime(
        ['2030-' + t for t in timestamps], format='%Y-%m-%d %H:%M:%S')
    return pd.Series(values, index=idx, name='load_pu')


def _load_lv_profiles_for_node(lv_grid_id, year, res_df, com_df, shares_df):
    """
    Calcola il profilo p.u. aggregato per un nodo MV sottostazione.

    Logica (come in Zapparoli 08_Data_loader.py):
      1. Estrae il codice BFS dalla stringa lv_grid_id (es. '1345-2_0_3' → 1345).
      2. Recupera i profili commerciali e residenziali per quel BFS.
      3. Filtra le share dei nodi LV appartenenti a quel lv_grid.
      4. Calcola la share media commerciale e residenziale
         (media non pesata sui nodi LV con carico > 0).
      5. Profilo aggregato = share_comm × comm_profile + share_res × res_profile.

    Returns
    -------
    Series 8760h p.u. con lo stesso indice dei profili res/com.
    """
    bfs = int(lv_grid_id.split('-')[0])

    # Profili del municipio
    comm_row = com_df[com_df['BFS_municipality_code'] == bfs]
    res_row  = res_df[res_df['BFS_municipality_code'] == bfs]
    if comm_row.empty or res_row.empty:
        return None

    comm_profile = comm_row.iloc[0, 1:].astype(float).values  # 8760 valori
    res_profile  = res_row.iloc[0, 1:].astype(float).values

    # Share per i nodi LV di questo grid
    node_shares = shares_df[shares_df['LV_grid'] == lv_grid_id]
    # Escludi nodi con entrambe le share a zero (nodi senza carico classificato)
    node_shares = node_shares[
        (node_shares['Commercial_demand_share'] > 0) |
        (node_shares['Residential_demand_share'] > 0)
    ]

    if node_shares.empty:
        return None

    # Share medie (non pesate — non abbiamo el_dmd LV per ogni nodo)
    mean_comm = node_shares['Commercial_demand_share'].mean()
    mean_res  = node_shares['Residential_demand_share'].mean()

    # Normalizza (le share potrebbero non sommare esattamente a 1)
    total = mean_comm + mean_res
    if total > 0:
        mean_comm /= total
        mean_res  /= total

    profile_values = mean_comm * comm_profile + mean_res * res_profile
    return profile_values


def load_node_demand_profiles(nodes_df, year=DATA_YEAR):
    """
    Calcola un profilo di carico p.u. specifico per ogni nodo consumer.

    - Nodi MV puri (lv_grid == '-1'):
        usano MV_load_profile.csv  (profilo standard aggregato)
    - Nodi sottostazione MV/LV (lv_grid != '-1'):
        usano la combinazione pesata di Commercial_profiles e
        Residential_profiles del municipio BFS corrispondente,
        con le share di LV_basicload_shares.csv

    Returns
    -------
    dict {osmid: np.ndarray 8760 valori p.u.}
    Indice temporale: lo stesso di _load_mv_profile()
    """
    mv_profile = _load_mv_profile(year)              # Series 8760h
    idx        = mv_profile.index
    mv_vals    = mv_profile.values                   # numpy array

    # Carica i file di profilo LV (sono grandi — caricali una volta sola)
    base = DEM_DIR / str(year)
    res_df    = pd.read_csv(base / 'Residential_profiles.csv')
    com_df    = pd.read_csv(base / 'Commercial_profiles.csv')
    shares_df = pd.read_csv(base / 'LV_basicload_shares.csv')

    consumer_nodes = nodes_df[nodes_df['consumers'] == 'True'].copy()
    consumer_nodes['osmid'] = consumer_nodes['osmid'].astype(int)

    profiles = {}
    lv_cache = {}   # cache per non ricalcolare lo stesso lv_grid più volte

    for _, row in consumer_nodes.iterrows():
        osmid   = int(row['osmid'])
        lv_grid = str(row['lv_grid'])

        if lv_grid == '-1':
            # Nodo MV puro → profilo standard
            profiles[osmid] = mv_vals
        else:
            # Nodo sottostazione MV/LV → profilo LV aggregato
            if lv_grid not in lv_cache:
                lv_prof = _load_lv_profiles_for_node(
                    lv_grid, year, res_df, com_df, shares_df)
                lv_cache[lv_grid] = lv_prof if lv_prof is not None else mv_vals
            profiles[osmid] = lv_cache[lv_grid]

    return profiles, idx


# ══════════════════════════════════════════════════════════════════
# 3. PROFILI PV  (espansi a 8760 ore, kW per nodo)
# ══════════════════════════════════════════════════════════════════

def _expand_pv_to_8760(df_12days):
    """
    Espande i 12 giorni rappresentativi (uno per mese) a 8760 ore.
    Ogni giorno del mese X riceve il profilo del giorno 15 del mese X.

    Il DataFrame in input ha colonne tipo '07-15 10:00:00'.
    L'output ha 8760 colonne tipo '07-23 10:00:00'.
    """
    expanded = {}
    for col in df_12days.columns:
        month  = int(col.split('-')[0])
        hour   = col.split(' ')[1]
        n_days = calendar.monthrange(2030, month)[1]
        for day in range(1, n_days + 1):
            new_col          = f"{month:02d}-{day:02d} {hour}"
            expanded[new_col] = df_12days[col].values

    expanded_df = pd.DataFrame(expanded, index=df_12days.index)
    cols_sorted  = sorted(expanded_df.columns,
                          key=lambda x: datetime.strptime('2030-' + x, '%Y-%m-%d %H:%M:%S'))
    return expanded_df[cols_sorted]


def load_pv_profiles(grid=GRID_ID, year=DATA_YEAR):
    """
    Carica i profili di generazione PV per ogni nodo MV con impianto installato.

    Returns
    -------
    pv_gen : DataFrame  — indice = MV_osmid (int), colonne = 8760 timestamps
                          valori in kW (generazione media del giorno rappresentativo)
    pv_installed : Series — {osmid: P_installed_kW}
    """
    gen_path  = PV_DIR / str(year) / 'MV_generation.csv'
    inst_path = PV_DIR / str(year) / 'MV_P_installed.csv'

    gen  = pd.read_csv(gen_path)
    inst = pd.read_csv(inst_path)

    gen  = gen[gen['MV_grid'] == grid].set_index('MV_osmid').drop(columns=['MV_grid'])
    inst = inst[inst['MV_grid'] == grid].set_index('MV_osmid').drop(columns=['MV_grid'])

    # Espandi 288 → 8760 ore
    pv_gen = _expand_pv_to_8760(gen)
    pv_gen.index = pv_gen.index.astype(int)

    return pv_gen, inst['P_installed_kW']


# ══════════════════════════════════════════════════════════════════
# 4. PARAMETRI BESS
# ══════════════════════════════════════════════════════════════════

def load_bess_params(grid=GRID_ID, year=DATA_YEAR):
    """
    Carica i parametri delle batterie per ogni nodo MV con BESS.

    Returns
    -------
    DataFrame con colonne:
        Battery_capacity_kWh, Nominal_power_kW,
        Charging_efficiency, Discharging_efficiency
    Indice: MV_osmid (int)
    """
    path = BESS_DIR / str(year) / 'BESS_allocation_MV.csv'
    df   = pd.read_csv(path)
    df   = df[df['MV_grid'] == grid].set_index('MV_osmid').drop(columns=['MV_grid'])
    df.index = df.index.astype(int)
    return df


# ══════════════════════════════════════════════════════════════════
# 5. PARAMETRI HP + TEMPERATURE
# ══════════════════════════════════════════════════════════════════

def load_hp_params(grid=GRID_ID, year=DATA_YEAR):
    """
    Carica i parametri termici delle pompe di calore e i profili
    di temperatura delle stazioni meteo associate.

    Returns
    -------
    hp : DataFrame  — indice = MV_osmid (int), colonne:
            Nominal_power_kW, Thermal_capacitance_KWh/K,
            Thermal_conductivity_kW/K, COP, COP_0, COP_1, COP_2,
            Temperature_profile_name
    temp : dict {station_name: Series 8760h °C con indice datetime}
    """
    hp_path   = HP_DIR / str(year) / 'MV_heat_pump_allocation.csv'
    temp_path = HP_DIR / str(year) / 'Temperature_profiles.csv'

    hp   = pd.read_csv(hp_path)
    temp = pd.read_csv(temp_path, index_col=0)

    hp = hp[hp['MV_grid'] == grid].set_index('MV_osmid').drop(columns=['MV_grid'])
    hp.index = hp.index.astype(int)

    # Carica solo le stazioni usate dalla rete
    stations = hp['Temperature_profile_name'].unique()
    temp_dict = {}
    for st in stations:
        vals = temp.loc[st].values.astype(float)
        idx = pd.to_datetime(['2030-' + c for c in temp.columns],
                     format='%Y-%m-%d %H:%M:%S')
        temp_dict[st] = pd.Series(vals, index=idx, name=st)

    return hp, temp_dict


# ══════════════════════════════════════════════════════════════════
# 6. BASELINE HP  (potenza elettrica che mantiene T_indoor costante)
# ══════════════════════════════════════════════════════════════════

def compute_hp_baseline(hp_params, temp_dict, T_set=20.0):
    """
    Calcola il profilo di consumo elettrico baseline delle HP
    usando il modello RC del primo ordine in regime stazionario.

    Modello (Zapparoli et al. 2025, eq. 3, 6, 8):
        H        = Thermal_conductivity_kW/K    [conduttanza termica, kW/K]
        Δ(t)     = T_set - T_out(t)             [lift termico, °C]
        Q(t)     = H × Δ(t)                     [potenza termica, kW]
        COP(t)   = COP_0 + COP_1·Δ + COP_2·Δ²  [dipende dal lift, non da T_out]
        P_HP(t)  = Q(t) / COP(t)                [potenza elettrica, kW]

    Note:
    - COP usa il lift termico Δ = T_indoor - T_outdoor (eq. 8 del paper),
      NON la temperatura esterna da sola. A lift maggiore il COP diminuisce.
    - Q usa la conduttanza H (moltiplicazione), non la resistenza (divisione).
    - Clipping [0, P_nom]: solo riscaldamento, mai oltre il nominale.

    Returns
    -------
    DataFrame  — indice = datetime (8760h), colonne = MV_osmid (int)
                 valori in kW elettrici
    """
    hp_baseline = {}
    ref_idx = list(temp_dict.values())[0].index

    for node, row in hp_params.iterrows():
        H       = float(row['Thermal_conductivity_kW/K'])  # conduttanza [kW/K]
        P_nom   = float(row['Nominal_power_kW'])
        COP_0   = float(row['COP_0'])
        COP_1   = float(row['COP_1'])
        COP_2   = float(row['COP_2'])
        station = row['Temperature_profile_name']

        T_out   = temp_dict[station].values.astype(float)  # °C

        # Lift termico Δ = T_indoor - T_outdoor (eq. 8 del paper)
        delta   = T_set - T_out                            # °C

        # COP funzione del lift — diminuisce con freddo crescente (fisicamente corretto)
        COP_t   = COP_0 + COP_1 * delta + COP_2 * delta**2
        COP_t   = np.maximum(COP_t, 1.0)                  # guard: COP ≥ 1

        # Potenza termica per mantenere T_set: Q = H × Δ (eq. 3 + 6 del paper)
        Q       = H * delta                                # kW termici

        # Potenza elettrica
        P_HP    = Q / COP_t                                # kW elettrici

        # Clipping: solo riscaldamento, non oltre il nominale installato
        P_HP    = np.clip(P_HP, 0.0, P_nom)

        hp_baseline[node] = P_HP

    return pd.DataFrame(hp_baseline, index=ref_idx)

# ══════════════════════════════════════════════════════════════════
# 7. BASELINE COMPLETA DELLA RETE
# ══════════════════════════════════════════════════════════════════

def build_baseline(el_dmd_dict, nodes_df, year=DATA_YEAR, grid=GRID_ID, T_set=20.0):
    """
    Costruisce la potenza netta baseline per ogni nodo consumer (8760 ore).

    P_net_k(t) = el_dmd_k × demand_profile_k(t)  [carico passivo, MW]
               + P_HP_k(t) / 1000                 [HP baseline, MW]
               - P_PV_k(t) / 1000                 [PV baseline, MW]

    demand_profile_k è specifico per nodo:
      - MV puri (lv_grid=='-1')   → MV_load_profile (standard)
      - Sottostazioni MV/LV       → mix commerciale/residenziale per BFS
    BESS: baseline = 0 (come in Brändle et al.)

    Parameters
    ----------
    el_dmd_dict : dict {osmid: el_dmd_MW}
    nodes_df    : GeoDataFrame — nodi della rete (serve per lv_grid)

    Returns
    -------
    baseline : DataFrame  8760h × n_nodi (MW)
    bess     : DataFrame  parametri BESS
    pv_inst  : Series     potenza installata PV (kWp)
    hp_params: DataFrame  parametri HP
    """
    # Profili per nodo (distingue MV puri da sottostazioni)
    node_profiles, idx = load_node_demand_profiles(nodes_df, year)

    pv_gen, pv_inst    = load_pv_profiles(grid, year)
    bess               = load_bess_params(grid, year)
    hp_params, temp_d  = load_hp_params(grid, year)
    hp_base            = compute_hp_baseline(hp_params, temp_d, T_set)

    baseline = pd.DataFrame(0.0, index=idx,
                             columns=list(el_dmd_dict.keys()))

    for node, el_dmd_mw in el_dmd_dict.items():

        # 1. Carico passivo con profilo specifico del nodo
        profile = node_profiles.get(node, list(node_profiles.values())[0])
        baseline[node] += el_dmd_mw * profile

        # 2. HP baseline
        if node in hp_base.columns:
            hp_series = hp_base[node].values
            baseline[node] += hp_series / 1000      # kW → MW

        # 3. PV (si sottrae)
        if node in pv_gen.index:
            baseline[node] -= pv_gen.loc[node].values / 1000   # kW → MW

    return baseline, bess, pv_inst, hp_params


# ══════════════════════════════════════════════════════════════════
# 8. FUNZIONE PRINCIPALE — carica tutto in un colpo
# ══════════════════════════════════════════════════════════════════

def load_all(grid=GRID_ID, year=DATA_YEAR, T_set=20.0):
    """
    Carica tutti i dati necessari per la pipeline FFOR.

    Returns
    -------
    dict con chiavi:
        'nodes'     : GeoDataFrame nodi
        'edges'     : GeoDataFrame linee
        'el_dmd'    : dict {osmid: el_dmd_MW}
        'baseline'  : DataFrame 8760h × n_nodi (MW)
        'bess'      : DataFrame parametri BESS
        'pv_inst'   : Series potenza installata PV (kWp)
        'hp_params' : DataFrame parametri HP
        'pv_gen'    : DataFrame generazione PV 8760h (kW)
        'node_profiles': dict {osmid: np.array 8760h p.u.}
    """
    print(f"Caricamento dati per rete {grid}, scenario {year}...")

    nodes, edges         = load_network_topology(grid)
    el_dmd               = get_consumer_nodes(nodes)
    pv_gen, pv_inst      = load_pv_profiles(grid, year)
    bess                 = load_bess_params(grid, year)
    hp_params, temp_dict = load_hp_params(grid, year)
    node_profiles, idx   = load_node_demand_profiles(nodes, year)

    baseline, bess, pv_inst, hp_params = build_baseline(
        el_dmd, nodes, year, grid, T_set)

    # Statistiche sui profili nodo
    mv_count  = sum(1 for n in nodes[nodes['consumers']=='True']['osmid'].astype(int)
                    if str(nodes[nodes['osmid'].astype(int)==n]['lv_grid'].values[0]) == '-1')
    lv_count  = sum(1 for n in nodes[nodes['consumers']=='True']['osmid'].astype(int)
                    if str(nodes[nodes['osmid'].astype(int)==n]['lv_grid'].values[0]) != '-1')

    print(f"  Nodi consumer : {len(el_dmd)}")
    print(f"    di cui MV puri (profilo MV):       {mv_count}")
    print(f"    di cui sottostazioni MV/LV (LV):   {lv_count}")
    print(f"  Nodi con PV   : {len(pv_inst)}")
    print(f"  Nodi con BESS : {len(bess)}")
    print(f"  Nodi con HP   : {len(hp_params)}")
    print(f"  Baseline shape: {baseline.shape}")
    print("Caricamento completato.")

    return {
        'nodes'        : nodes,
        'edges'        : edges,
        'el_dmd'       : el_dmd,
        'baseline'     : baseline,
        'bess'         : bess,
        'pv_inst'      : pv_inst,
        'hp_params'    : hp_params,
        'pv_gen'       : pv_gen,
        'node_profiles': node_profiles,
        'time_index'   : idx,
    }


# ══════════════════════════════════════════════════════════════════
# TEST
# ══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    data = load_all()

    baseline      = data['baseline']
    bess          = data['bess']
    node_profiles = data['node_profiles']

    print("\n=== PROFILI DI CARICO PER NODO ===")
    print("Confronto MV puro vs sottostazione MV/LV alla stessa ora:")
    t_str = '2030-12-15 19:00:00'
    t = pd.Timestamp(t_str)
    if t in baseline.index:
        # Nodo 103 (MV puro, lv_grid==-1)
        p103 = node_profiles[103]
        idx  = baseline.index.get_loc(t)
        # Nodo 0 (sottostazione, lv_grid='1345-2_0_3')
        p0   = node_profiles[0]
        print(f"  Ora: {t_str}")
        print(f"  Nodo 103 (MV puro)     profilo p.u.: {p103[idx]:.4f}")
        print(f"  Nodo 0   (sottostation) profilo p.u.: {p0[idx]:.4f}")
        print(f"  (se diversi, la logica LV è attiva)")

    print("\n=== SNAPSHOT: 15 dicembre ore 19:00 ===")
    snap = baseline.loc[baseline.index == t]
    if not snap.empty:
        row = snap.iloc[0].sort_values(ascending=False)
        print(row.to_string())
        print(f"\nPotenza totale al PCC: {row.sum():.3f} MW")

    print("\n=== SNAPSHOT: 15 luglio ore 12:00 ===")
    t2 = pd.Timestamp('2030-07-15 12:00:00')
    snap2 = baseline.loc[baseline.index == t2]
    if not snap2.empty:
        row2 = snap2.iloc[0].sort_values(ascending=False)
        print(row2.to_string())
        print(f"\nPotenza totale al PCC: {row2.sum():.3f} MW")

    print("\n=== PARAMETRI BESS ===")
    print(bess.to_string())

    # ← AGGIUNGI QUESTE DUE RIGHE QUI, allo stesso livello di rientro
    print("\nGenerazione plot profilo annuale nodo 122...")
    plot_node_annual_profile(data, osmid=122)


 