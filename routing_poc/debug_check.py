import sys
sys.path.insert(0, "src")
from config_loader import load_config
from data_loader import load_all
from distance_matrix import build_distance_matrix
from phase1_priority import compute_priority
from phase2_matching import match_resources
from datetime import date

cfg  = load_config("config/routing_rules.json")
data = load_all("data")
dist = build_distance_matrix(data["technicians"], data["jobs"])
pq   = compute_priority(data["jobs"], cfg)
elig = match_resources(pq, data["technicians"], data["trucks"], data["programs"], dist, cfg, date.fromisoformat("2026-06-12"))

ow = cfg["objective_weights"]
w_revenue = float(ow["maximize_revenue"])
REVENUE_SCALE = 10

print(f"w_revenue = {w_revenue}  (type: {type(ow['maximize_revenue'])})")
print(f"raw config value: {ow['maximize_revenue']!r}")

rev_j004 = dict(zip(pq["job_id"], pq["revenue"]))["J004"]
penalty_j004 = int(round(w_revenue * rev_j004 * REVENUE_SCALE))
print(f"J004 revenue: {rev_j004}")
print(f"J004 expected penalty: int(round({w_revenue} * {rev_j004} * {REVENUE_SCALE})) = {penalty_j004}")

# Check what priority_queue looks like at first 3 positions
print()
print("Priority queue first 5 positions:")
for i in range(5):
    row = pq.iloc[i]
    rev = row["revenue"]
    pen = int(round(w_revenue * rev * REVENUE_SCALE))
    print(f"  pos {i}: {row['job_id']}  revenue={rev}  penalty={pen}")

# Check arc costs from T02 depot
t02_idx = dist["tech_index"]["T02"]
print()
print("Arc costs T02 -> node (n_techs+j_idx) for first 5 PQ positions:")
n_techs = dist["n_techs"]
for j_idx in range(5):
    node = n_techs + j_idx
    arc  = int(round(float(ow["minimize_drive_time"]) * dist["duration_s"][t02_idx][node]))
    job_at_pq_pos = pq.iloc[j_idx]["job_id"]
    job_at_matrix_pos = list(data["jobs"]["job_id"])[j_idx]
    print(f"  node {node} (PQ pos {j_idx} = {job_at_pq_pos}, matrix pos {j_idx} = {job_at_matrix_pos}): arc={arc}")
