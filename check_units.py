from grid_loader_mveq import Grid20Loader
import pandas as pd
import numpy as np

loader = Grid20Loader(
    grid_name="20_0",
    data_year=2030,
    start_date="01-01 00:00:00",
    end_date="12-31 23:00:00",
    setpoint_c=20.0,
)

nodes, edges = loader.load_grid()

print("\n=== NODES COLUMNS ===")
print(nodes.columns.tolist())

print("\n=== EDGES COLUMNS ===")
print(edges.columns.tolist())

print("\n=== NODES HEAD ===")
print(nodes.head().to_string())

print("\n=== EDGES HEAD ===")
print(edges.head().to_string())

# Colonne candidate
candidate_node_cols = ["osmid", "voltage", "el_dmd", "source", "lv_grid", "x", "y"]
candidate_edge_cols = ["u", "v", "r", "x", "b", "s_nom", "load", "OHL", "length"]

print("\n=== NODE NUMERIC SUMMARY ===")
for col in candidate_node_cols:
    if col in nodes.columns:
        print(f"\n--- nodes[{col}] ---")
        print(nodes[col].describe())

print("\n=== EDGE NUMERIC SUMMARY ===")
for col in candidate_edge_cols:
    if col in edges.columns:
        print(f"\n--- edges[{col}] ---")
        print(edges[col].describe())

print("\n=== UNIQUE VOLTAGES ===")
if "voltage" in nodes.columns:
    print(sorted(nodes["voltage"].dropna().unique()))

print("\n=== EDGE PARAMETER RAW RANGES ===")
for col in ["r", "x", "b", "s_nom"]:
    if col in edges.columns:
        vals = pd.to_numeric(edges[col], errors="coerce").dropna()
        print(
            f"{col}: min={vals.min():.8g}, "
            f"median={vals.median():.8g}, "
            f"max={vals.max():.8g}"
        )

print("\n=== QUICK HEURISTICS ===")

if "r" in edges.columns and "x" in edges.columns:
    r = pd.to_numeric(edges["r"], errors="coerce")
    x = pd.to_numeric(edges["x"], errors="coerce")
    print("r/x median:", (r / x.replace(0, np.nan)).median())
    print("x/r median:", (x / r.replace(0, np.nan)).median())

if "s_nom" in edges.columns:
    s = pd.to_numeric(edges["s_nom"], errors="coerce").dropna()
    print("s_nom values sample:", s.head(10).to_list())