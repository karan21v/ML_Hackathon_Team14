
from pathlib import Path
import numpy as np
from pyproj import Transformer

trip = Path(r"C:\Users\dhanu\Desktop\Student folder\train\00000057")

track = np.loadtxt(trip / f"{trip.name}.track")
nodes = np.loadtxt(trip / f"{trip.name}.nodes")
arcs = np.loadtxt(trip / f"{trip.name}.arcs", dtype=int)
route = np.loadtxt(trip / f"{trip.name}.route", dtype=int).reshape(-1)

transformer = Transformer.from_crs(
    "EPSG:4326", "EPSG:32632", always_xy=True
)

print("=== RAW COORDINATE RANGES ===")
print("GPS longitude:", track[:, 0].min(), track[:, 0].max())
print("GPS latitude:", track[:, 1].min(), track[:, 1].max())
print("Node longitude:", nodes[:, 0].min(), nodes[:, 0].max())
print("Node latitude:", nodes[:, 1].min(), nodes[:, 1].max())

gps_x, gps_y = transformer.transform(track[:, 0], track[:, 1])
node_x, node_y = transformer.transform(nodes[:, 0], nodes[:, 1])

gps_xy = np.column_stack((gps_x, gps_y))
node_xy = np.column_stack((node_x, node_y))

print("\n=== PROJECTED COORDINATES ===")
print("Invalid GPS coordinates:", (~np.isfinite(gps_xy)).any(axis=1).sum())
print("Invalid node coordinates:", (~np.isfinite(node_xy)).any(axis=1).sum())

print("\n=== ROAD GEOMETRY ===")
print("Arc index minimum:", arcs.min())
print("Arc index maximum:", arcs.max())
print("Node count:", len(nodes))
print("Invalid arc references:", ((arcs < 0) | (arcs >= len(nodes))).any(axis=1).sum())

valid_arcs = arcs[
    ((arcs >= 0) & (arcs < len(nodes))).all(axis=1)
]

if len(valid_arcs):
    dx = node_x[valid_arcs[:, 1]] - node_x[valid_arcs[:, 0]]
    dy = node_y[valid_arcs[:, 1]] - node_y[valid_arcs[:, 0]]
    lengths = np.hypot(dx, dy)

    print("Invalid segment lengths:", (~np.isfinite(lengths)).sum())
    print("Zero-length segments:", (lengths == 0).sum())
    print("Maximum segment length:", np.nanmax(lengths))

print("\n=== ROUTE REFERENCES ===")
print("Route minimum:", route.min())
print("Route maximum:", route.max())
print("Arc count:", len(arcs))
print("Invalid route references:", ((route < 0) | (route >= len(arcs))).sum())
