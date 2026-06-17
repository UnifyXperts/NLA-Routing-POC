import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import pandas as pd
import numpy as np
import requests

OSRM_URL   = "http://router.project-osrm.org/table/v1/driving/"
NULL_VALUE = 999_999.0

techs = pd.read_csv("data/technicians.csv")
jobs  = pd.read_csv("data/jobs.csv")

print("=" * 60)
print("INPUT COORDINATES  (no defaults — straight from CSV)")
print("=" * 60)

print("\nTECHNICIAN DEPOT LOCATIONS")
print(f"  {'ID':<6} {'Name':<8} {'lat':>10} {'lng':>11}")
print(f"  {'-'*6} {'-'*8} {'-'*10} {'-'*11}")
for _, r in techs.iterrows():
    print(f"  {r['tech_id']:<6} {r['name']:<8} {r['start_lat']:>10.4f} {r['start_lng']:>11.4f}")

print("\nJOB LOCATIONS")
print(f"  {'ID':<6} {'Customer':<18} {'lat':>10} {'lng':>11}")
print(f"  {'-'*6} {'-'*18} {'-'*10} {'-'*11}")
for _, r in jobs.iterrows():
    print(f"  {r['job_id']:<6} {r['customer_name']:<18} {r['lat']:>10.4f} {r['lng']:>11.4f}")

# Build location list: tech depots first, then jobs
tech_locs = list(zip(techs["start_lat"], techs["start_lng"]))
job_locs  = list(zip(jobs["lat"], jobs["lng"]))
locations = tech_locs + job_locs
n_techs   = len(techs)
n_jobs    = len(jobs)
n_locs    = len(locations)

labels = list(techs["tech_id"]) + list(jobs["job_id"])

print(f"\nTotal locations sent to OSRM: {n_locs}  ({n_techs} tech depots + {n_jobs} jobs)")

# OSRM call — lng,lat order
coords = ";".join(f"{lng},{lat}" for lat, lng in locations)
url    = f"{OSRM_URL}{coords}?annotations=distance,duration"
print(f"\nOSRM endpoint: {url[:80]}…")

resp = requests.get(url, timeout=30)
data = resp.json()

if data.get("code") != "Ok":
    print(f"OSRM error: {data}")
    sys.exit(1)

print(f"OSRM status  : {data['code']}")

def clean(matrix):
    n   = len(matrix)
    arr = np.zeros((n, n), dtype=float)
    null_count = 0
    for i in range(n):
        for j in range(n):
            v = matrix[i][j]
            if v is None:
                arr[i][j] = NULL_VALUE
                null_count += 1
            else:
                arr[i][j] = float(v)
    return arr, null_count

distance_m, d_nulls = clean(data["distances"])
duration_s, t_nulls = clean(data["durations"])

print(f"Null distances replaced : {d_nulls}  (-> {NULL_VALUE} m)")
print(f"Null durations replaced : {t_nulls}  (-> {NULL_VALUE} s)")

# ── Duration matrix (minutes) ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("OSRM DURATION MATRIX  (minutes, rounded)  — REAL road-network")
print("=" * 60)

col_w = 7
header = f"{'From/To':<8}" + "".join(f"{lbl:>{col_w}}" for lbl in labels)
print(header)
print("-" * len(header))
for i, from_lbl in enumerate(labels):
    row = f"{from_lbl:<8}"
    for j in range(n_locs):
        val = duration_s[i][j] / 60
        if i == j:
            row += f"{'–':>{col_w}}"
        elif val >= 9999:
            row += f"{'BIG':>{col_w}}"
        else:
            row += f"{val:>{col_w}.1f}"
    print(row)

# ── Distance matrix (km) ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("OSRM DISTANCE MATRIX  (km, rounded)  — REAL road-network")
print("=" * 60)

print(header)
print("-" * len(header))
for i, from_lbl in enumerate(labels):
    row = f"{from_lbl:<8}"
    for j in range(n_locs):
        val = distance_m[i][j] / 1000
        if i == j:
            row += f"{'–':>{col_w}}"
        elif val >= 9999:
            row += f"{'BIG':>{col_w}}"
        else:
            row += f"{val:>{col_w}.1f}"
    print(row)

# ── Spot-check: tech-to-job summary ───────────────────────────────────────
print("\n" + "=" * 60)
print("TECH DEPOT -> EACH JOB  (drive minutes, OSRM)")
print("=" * 60)
print(f"  {'Job':<6} " + "".join(f"{tid:>8}" for tid in techs['tech_id']))
print(f"  {'-'*6} " + "".join(f"{'--------':>8}" for _ in techs.iterrows()))
for j_idx, jid in enumerate(jobs["job_id"]):
    node_j = n_techs + j_idx
    row = f"  {jid:<6} "
    for t_idx in range(n_techs):
        mins = duration_s[t_idx][node_j] / 60
        row += f"{mins:>8.1f}"
    print(row)

print("\nAll values above come directly from OSRM. No defaults assumed.")
print("NULL_VALUE placeholder only appears if OSRM cannot route between two points.")
