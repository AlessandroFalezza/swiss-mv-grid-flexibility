import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

BASE_GEO = Path(r"C:\Users\Alessandro\aeonetos-Swiss-PDGs-3354b4d\grids\geojson_data\MV")

CANDIDATES = ['138_3', '20_0', '490_0']

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, grid_id in zip(axes, CANDIDATES):
    # Carica nodi e linee
    nodes = gpd.read_file(BASE_GEO / f"{grid_id}_nodes")
    edges = gpd.read_file(BASE_GEO / f"{grid_id}_edges")
    
    # Disegna linee
    edges.plot(ax=ax, color='steelblue', linewidth=1.2, alpha=0.7)
    
    # Disegna nodi consumer
    consumers = nodes[nodes['consumers'] == 'True']
    consumers.plot(ax=ax, color='orange', markersize=20, alpha=0.8, label='Consumer')
    
    # Evidenzia PCC (source)
    pcc = nodes[nodes['source'] == True]
    pcc.plot(ax=ax, color='red', markersize=120, marker='*', label='PCC')
    
    # Info
    n_bus = len(nodes)
    total_mw = nodes['el_dmd'].sum()
    min_v = nodes['voltage'].min()
    
    ax.set_title(f"Rete {grid_id}\n{n_bus} nodi | {total_mw:.1f} MW | min_v={min_v:.4f}", 
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.set_axis_off()

plt.tight_layout()
plt.savefig('candidates_topology.png', dpi=150, bbox_inches='tight')
plt.show()
print("Salvato: candidates_topology.png")