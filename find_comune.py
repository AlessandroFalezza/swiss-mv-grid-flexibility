from pathlib import Path
import geopandas as gpd

BASE_GEO = Path(r"C:\Users\Alessandro\aeonetos-Swiss-PDGs-3354b4d\grids\geojson_data\MV")
nodes = gpd.read_file(BASE_GEO / "20_0_nodes")
nodes_wgs = nodes.to_crs('EPSG:4326')
lat = nodes_wgs.geometry.y.mean()
lon = nodes_wgs.geometry.x.mean()
print(f"lat={lat:.5f}, lon={lon:.5f}")
print(f"Google Maps: https://www.google.com/maps?q={lat},{lon}")