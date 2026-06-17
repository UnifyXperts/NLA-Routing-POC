import sys
import os
import time
from datetime import date

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from config_loader   import load_config, print_config
from data_loader     import load_all, print_summary
from phase1_priority import compute_priority, print_queue
from phase2_matching import match_resources, print_matches
from distance_matrix import build_distance_matrix
from optimizer       import optimize, print_routes
from visualization   import build_route_map
from dashboard       import build_kpi, build_optimization_summary
from llm_summary     import generate_narrative


def section(title: str) -> None:
    print(f"\n{'='*56}")
    print(f"  {title}")
    print(f"{'='*56}")


def main():
    t0       = time.time()
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "data")
    cfg_path = os.path.join(base_dir, "config", "routing_rules.json")
    out_dir  = os.path.join(base_dir, "output")
    os.makedirs(out_dir, exist_ok=True)

    # ── Configuration ──────────────────────────────────────────────────────────
    section("CONFIGURATION · routing_rules.json")
    cfg = load_config(cfg_path)
    print_config(cfg)

    # ── Task 1: Load data ──────────────────────────────────────────────────────
    section("TASK 1 · Data Loader")
    data = load_all(data_dir)
    print_summary(data)

    # ── Distance Matrix (OSRM) — must come before Phase II ────────────────────
    section("DISTANCE MATRIX · OSRM")
    print("  Calling OSRM Table API …")
    dist_data = build_distance_matrix(data["technicians"], data["jobs"])
    n_locs    = dist_data["n_techs"] + dist_data["n_jobs"]
    print(f"  Matrix size : {n_locs}×{n_locs}  "
          f"({dist_data['n_techs']} tech depots + {dist_data['n_jobs']} jobs)")
    # Sample: T01 → J001
    t01_idx  = dist_data["tech_index"].get("T01")
    j001_idx = dist_data["job_index"].get("J001")
    if t01_idx is not None and j001_idx is not None:
        d_km  = dist_data["distance_m"][t01_idx][j001_idx] / 1000
        dur_m = dist_data["duration_s"][t01_idx][j001_idx] / 60
        print(f"  Sample T01->J001 : {d_km:.2f} km  /  {dur_m:.1f} min\n")

    # ── Phase I: Priority queue ────────────────────────────────────────────────
    section("PHASE I · Queue Priority")
    priority_queue = compute_priority(data["jobs"], cfg)
    print_queue(priority_queue)

    # ── Phase II: Capacity matching ────────────────────────────────────────────
    section("PHASE II · Capacity Matching")
    route_date     = date.fromisoformat(cfg["operational_constraints"]["route_date"])
    eligibility_df = match_resources(
        priority_queue,
        data["technicians"],
        data["trucks"],
        data["programs"],
        dist_data,
        cfg,
        route_date,
    )
    print_matches(eligibility_df)

    # ── Phase III: Route optimization ──────────────────────────────────────────
    section("PHASE III · Route Optimization  (OR-Tools VRP)")
    routes, unassigned = optimize(
        dist_data,
        priority_queue,
        data["technicians"],
        data["trucks"],
        data["programs"],
        eligibility_df,
        cfg,
    )
    print_routes(routes, unassigned)

    # ── Visualization ──────────────────────────────────────────────────────────
    section("VISUALIZATION · Folium Map")
    map_path = os.path.join(out_dir, "route_map.html")
    if routes:
        build_route_map(routes, data["jobs"], data["technicians"], output_path=map_path)
    else:
        from visualization import build_jobs_map
        build_jobs_map(data["jobs"], output_path=map_path)
    print(f"  Saved -> {map_path}\n")

    # ── Optimization Summary ───────────────────────────────────────────────────
    section("OPTIMIZATION SUMMARY")
    if routes:
        summary = build_optimization_summary(
            jobs           = data["jobs"],
            priority_queue = priority_queue,
            technicians    = data["technicians"],
            routes         = routes,
            unassigned     = unassigned,
            dist_data      = dist_data,
            cfg            = cfg,
            programs       = data["programs"],
            eligibility_df = eligibility_df,
        )
        print(summary)
    else:
        print("  No routes — skipping summary.")

    # ── KPI Dashboard ──────────────────────────────────────────────────────────
    section("KPI DASHBOARD")
    if routes:
        kpi_df = build_kpi(data["jobs"], data["technicians"], routes, dist_data)
        print(kpi_df.to_string(index=False))
    else:
        print("  No routes — skipping KPI.")

    # ── LLM Narrative ─────────────────────────────────────────────────────────
    section("LLM NARRATIVE")
    if routes and cfg.get("llm", {}).get("enabled", False):
        narrative = generate_narrative(
            routes=routes,
            unassigned=unassigned,
            jobs=data["jobs"],
            technicians=data["technicians"],
            dist_data=dist_data,
            cfg=cfg,
        )
        print(narrative)
    else:
        print("  LLM disabled — set llm.enabled=true and llm.provider=groq in routing_rules.json")

    section(f"DONE  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
