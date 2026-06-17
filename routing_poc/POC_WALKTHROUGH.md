# TouchTurf Routing Engine POC — Step-by-Step Walkthrough

## What This POC Does

Proves that a field-service routing engine can prioritize jobs, match technicians, compute distances, optimize routes, and visualize results — all without ERPNext.

---

## Execution Flow (`main.py`)

### Step 0 — Load Config (`config/routing_rules.json`)

Reads three rule blocks before anything else:

| Block | Key Settings |
|---|---|
| `priority_weights` | `new_customer=1000`, `asap=100`, `last_service_days=1` |
| `optimization_weights` | `workload_balance_weight=100` (balances load across techs) |
| `operational_constraints` | `max_hours=8h`, `lunch_after=240 min` |

All downstream phases read from this config — changing a number here changes behavior everywhere.

---

### Step 1 — Data Loader (`src/data_loader.py`)

Reads 4 CSV files from `data/`:

| File | What it contains |
|---|---|
| `jobs.csv` | 20 Richmond VA jobs — coords, revenue, flags, program |
| `technicians.csv` | Tech IDs, home depot coords, skills, licenses |
| `trucks.csv` | Truck-to-tech mapping, capacity, equipment |
| `programs.csv` | Program requirements — skill, equipment, material |

Validates required columns on load; raises immediately if anything is missing.

---

### Step 2 — Phase I: Queue Priority (`src/phase1_priority.py`)

**Input:** `jobs.csv`

**Logic:** Scores every job with a weighted formula:

```
priority_score = (new_customer × 1000) + (asap × 100) + (last_service_days × 1)
```

- New customer flag dominates (1000 pts) — always rises to top
- ASAP flag second (100 pts)
- Days since last visit breaks ties

**Output:** Jobs sorted highest-score-first → this is the service queue order.

---

### Step 3 — Phase II: Capacity Matching (`src/phase2_matching.py`)

**Input:** jobs + technicians + trucks + programs

For every job, checks each technician against 3 gates:

1. **Skill gate** — tech's `skills` list must contain the program's `required_skill`
2. **Equipment gate** — tech's truck must have the program's `required_equipment`
3. **Capacity gate** — truck `capacity` ≥ program `material_required`

**Output:** Per-job list of eligible tech IDs (e.g. `J003 → [T01, T03]`). Jobs with zero eligible techs show `NONE`.

---

### Step 4 — Distance Matrix (`src/distance_matrix.py`)

**Input:** job coordinates + fixed depot (`37.5407, -77.4360` — Richmond VA)

Computes a full **(N+1) × (N+1)** matrix using **Haversine formula** (straight-line miles on a sphere). No external API needed.

- Row/column 0 = Depot
- Rows/columns 1–N = each job
- Values in miles, rounded to 3 decimal places

---

### Step 5 — Phase III: Route Optimization (`src/optimizer.py`)

Uses **Google OR-Tools** vehicle routing (VRP) solver.

**Setup:**
- Nodes = Depot + all jobs
- Vehicles = one per technician
- Distances scaled to integers (`miles × 100`) for OR-Tools

**Constraints enforced:**
- Time dimension: total route ≤ `max_hours_per_tech × 60` minutes
- Eligibility: each job node is locked to only its eligible techs via `VehicleVar`
- Disjunctions: every job has a penalty of 10,000 if skipped (soft constraint — solver may drop a job if it truly can't fit)

**Objectives (in priority order):**
1. Minimize total drive distance (arc cost)
2. Balance workload — `GlobalSpanCostCoefficient=100` on Distance dimension penalizes uneven routes

**Search strategy:** `PATH_CHEAPEST_ARC` for initial solution → `GUIDED_LOCAL_SEARCH` to improve → 10-second time limit.

**Lunch break (post-processing):** After solving, if a route's total time > 240 min, a `BREAK` token is inserted at the midpoint of the stop list.

**Output:** Per-tech route dict:
```
T01 → [Depot, J003, J011, BREAK, J001, Depot]  (310 min)
```

---

### Step 6 — Visualization (`src/visualization.py`)

Uses **Folium** to render an interactive HTML map:

- Red home icon = Depot
- Numbered circle markers (color-coded per tech) = job stops
- Polylines connecting stops in route order

Output: `output/route_map.html` — open in any browser.

---

### Step 7 — KPI Dashboard (`src/dashboard.py`)

Computes 6 metrics from the final routes and prints as a pandas DataFrame:

| Metric | How computed |
|---|---|
| Total Jobs | `len(jobs)` |
| Assigned Jobs | Unique job IDs across all routes |
| Unassigned Jobs | Total − Assigned |
| Total Revenue ($) | Sum of `revenue` for assigned jobs |
| Total Distance (mi) | Sum of matrix lookups across each route's consecutive stops |
| Avg Stops / Technician | Assigned stops ÷ number of techs |

---

## Data Flow Summary

```
routing_rules.json
        │
        ▼
jobs/technicians/trucks/programs CSVs
        │
        ▼
Phase I  →  priority_score per job  →  sorted queue
        │
        ▼
Phase II →  eligible techs per job  →  eligibility_df
        │
        ▼
Haversine distance matrix  (depot + all jobs)
        │
        ▼
OR-Tools VRP solver  →  optimized routes (with lunch break)
        │
        ├──▶  Folium map  →  output/route_map.html
        │
        └──▶  KPI DataFrame  →  printed to console
```

---

## Key Files

| File | Role |
|---|---|
| `main.py` | Orchestrator — runs all steps in order |
| `config/routing_rules.json` | All tunable weights and constraints |
| `data/*.csv` | Sample data — 20 Richmond VA jobs |
| `src/phase1_priority.py` | Weighted scoring and sort |
| `src/phase2_matching.py` | Skill / equipment / capacity gating |
| `src/distance_matrix.py` | Haversine N×N matrix |
| `src/optimizer.py` | OR-Tools VRP model + lunch post-processing |
| `src/visualization.py` | Folium interactive map |
| `src/dashboard.py` | KPI summary table |
| `output/route_map.html` | Generated map (created at runtime) |
