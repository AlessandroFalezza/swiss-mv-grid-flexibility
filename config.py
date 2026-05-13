# config.py — Path centralizzati per il progetto OES
from pathlib import Path

# ── DATASET ────────────────────────────────────────────────────
SWISS_PDGS = Path(r"C:\Users\Alessandro\aeonetos-Swiss-PDGs-3354b4d")
ZAPPAROLI  = Path(r"C:\Users\Alessandro\SwissDN_DERs")

# ── RETE ───────────────────────────────────────────────────────
GRID_ID   = '20_0'
DATA_YEAR = 2030

# ── PATH DERIVATI ──────────────────────────────────────────────
GEO_MV   = SWISS_PDGS / 'grids' / 'geojson_data' / 'MV'
PP_MV    = SWISS_PDGS / 'grids' / 'pandapower_data' / 'MV'

PV_DIR   = ZAPPAROLI / '01_PV'
BESS_DIR = ZAPPAROLI / '02_BESS'
HP_DIR   = ZAPPAROLI / '03_HP'
DEM_DIR  = ZAPPAROLI / '05_Demand'

# ── VERIFICA ───────────────────────────────────────────────────
if __name__ == '__main__':
    paths = {
        'Swiss-PDGs geojson': GEO_MV,
        'Swiss-PDGs pandapower': PP_MV,
        'Zapparoli PV': PV_DIR / str(DATA_YEAR),
        'Zapparoli BESS': BESS_DIR / str(DATA_YEAR),
        'Zapparoli HP': HP_DIR / str(DATA_YEAR),
        'Zapparoli Demand': DEM_DIR / str(DATA_YEAR),
    }
    print("=== VERIFICA PATH ===")
    all_ok = True
    for name, path in paths.items():
        ok = path.exists()
        print(f"  {'OK' if ok else 'MANCA'} — {name}: {path}")
        if not ok:
            all_ok = False
    print()
    print("Tutti i path OK!" if all_ok else "ATTENZIONE: alcuni path mancanti")