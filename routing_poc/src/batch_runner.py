"""
Batched pipeline for large job sets.

Why batching?
  • OSRM table API has URL length limits (~4 KB). The 2-D OSRM block approach
    in distance_matrix.py now handles any total matrix size, but OR-Tools
    performance degrades with 200+ nodes and a 60-second time limit.
  • Splitting into batches of BATCH_SIZE (default 45) keeps each OR-Tools
    sub-problem at ≤ 61 nodes (16 techs + 45 jobs), solving in < 5 seconds.
  • Remaining technician capacity is tracked across batches so techs are never
    over-scheduled.

Public API
----------
extract_sub_distdata(full_dist_data, batch_jobs)  ->  sub_dist_data dict
optimize_batched(full_dist_data, priority_queue, technicians, trucks,
                 programs, eligibility_df, cfg,
                 batch_size=45, status_cb=None)    ->  (routes, unassigned)
"""

import copy
import numpy as np
import pandas as pd

from optimizer import optimize as _optimize, _shift_seconds


# ── Sub-matrix extraction ─────────────────────────────────────────────────────

def extract_sub_distdata(full: dict, batch_jobs: pd.DataFrame) -> dict:
    """
    Extract a (n_techs + len(batch_jobs)) × (n_techs + len(batch_jobs)) sub-matrix
    from the full NxN dist_data.

    Tech nodes always occupy indices 0..n_techs-1 (unchanged from full matrix).
    Batch job nodes are remapped to n_techs..n_techs+len(batch_jobs)-1.
    """
    n_techs   = full["n_techs"]
    full_dur  = full["duration_s"]
    full_dist = full["distance_m"]

    batch_jids  = list(batch_jobs["job_id"])
    global_idx  = list(range(n_techs)) + [full["job_index"][jid] for jid in batch_jids]

    gix      = np.array(global_idx, dtype=int)
    sub_dur  = full_dur[np.ix_(gix, gix)]
    sub_dist = full_dist[np.ix_(gix, gix)]

    sub_job_index = {jid: n_techs + k for k, jid in enumerate(batch_jids)}

    return {
        "duration_s": sub_dur,
        "distance_m": sub_dist,
        "n_techs":    n_techs,
        "n_jobs":     len(batch_jids),
        "tech_index": full["tech_index"],   # unchanged (0..n_techs-1)
        "job_index":  sub_job_index,
    }


# ── Batched optimiser ─────────────────────────────────────────────────────────

def optimize_batched(
    full_dist_data: dict,
    priority_queue: pd.DataFrame,
    technicians: pd.DataFrame,
    trucks: pd.DataFrame,
    programs: pd.DataFrame,
    eligibility_df: pd.DataFrame,
    cfg: dict,
    batch_size: int = 45,
    status_cb=None,
    job_cluster_map: dict = None,
) -> tuple:
    """
    Optimise jobs in priority-ordered batches of `batch_size`.

    Each batch:
      1. Extracts sub-matrix from full_dist_data  (no extra OSRM call)
      2. Filters eligibility_df to batch jobs
      3. Injects remaining tech capacity into cfg  (2nd batch onwards)
      4. Calls OR-Tools optimizer on ≤ 61 nodes

    Remaining capacity is tracked per-tech across batches so techs
    are never over-scheduled on a single day.

    Returns
    -------
    all_routes   : dict  {tech_id: route_info}   — merged across all batches
    all_unassigned : list  — job_ids that could not be scheduled
    """
    all_routes:     dict = {}
    all_unassigned: list = []

    # Full-shift seconds per tech derived from shift_start/shift_end.
    # _shift_seconds falls back to max_hours (default 8 h) when those columns
    # are absent or cannot be parsed, so any shift length is supported.
    full_shift: dict = {
        row["tech_id"]: _shift_seconds(row)
        for _, row in technicians.iterrows()
    }

    # Budget pool = target_pct × actual shift, not 100% of shift.
    # Initialising here (not lazily) ensures batch 1 is also capped correctly.
    target_pct_val = float(cfg.get("utilization", {}).get("target_pct", 90)) / 100.0
    tech_remaining_s: dict = {
        tid: int(target_pct_val * s)
        for tid, s in full_shift.items()
    }

    jobs_list = priority_queue.reset_index(drop=True)
    n_total   = len(jobs_list)
    batches   = [
        jobs_list.iloc[i:i + batch_size].reset_index(drop=True)
        for i in range(0, n_total, batch_size)
    ]
    n_batches = len(batches)

    for b_idx, batch_jobs in enumerate(batches):
        if batch_jobs.empty:
            continue

        if status_cb:
            status_cb(
                f"   Batch {b_idx + 1}/{n_batches}  "
                f"({len(batch_jobs)} jobs, "
                f"jobs {b_idx * batch_size + 1}–{min((b_idx + 1) * batch_size, n_total)} "
                f"by priority)…"
            )

        # ── Sub-matrix for this batch ─────────────────────────────────────────
        sub_dist = extract_sub_distdata(full_dist_data, batch_jobs)

        # ── Eligibility slice ─────────────────────────────────────────────────
        batch_jids = set(batch_jobs["job_id"])
        batch_elig = (
            eligibility_df[eligibility_df["job_id"].isin(batch_jids)]
            .reset_index(drop=True)
        )

        # ── Inject remaining capacity into every batch (including batch 1) ──────
        # optimizer.py derives shift_s_list from this override, so the hard
        # time-dimension cap is always target_pct × actual_shift − time_used_so_far.
        batch_cfg = copy.deepcopy(cfg)
        batch_cfg["_tech_remaining_s"] = {
            tid: max(0, tech_remaining_s.get(tid, int(target_pct_val * full_shift.get(tid, 28800))))
            for tid in full_shift
        }

        # ── OR-Tools optimisation ─────────────────────────────────────────────
        routes_b, unassigned_b = _optimize(
            sub_dist, batch_jobs, technicians, trucks,
            programs, batch_elig, batch_cfg,
            job_cluster_map=job_cluster_map,
        )

        all_unassigned.extend(unassigned_b)

        if not routes_b:
            continue

        for tid, info in routes_b.items():
            stops = [s for s in info["route"] if s not in ("Depot", "BREAK")]
            used_s = info["total_minutes"] * 60

            # Deplete remaining capacity for next batch
            prev_s = tech_remaining_s.get(tid, full_shift.get(tid, 28800))
            tech_remaining_s[tid] = max(0, prev_s - used_s)

            if not stops:
                continue

            if tid not in all_routes:
                # First batch where this tech has stops — copy route info as-is.
                # shift_minutes from batch 1 = full shift (no override injected).
                all_routes[tid] = {
                    "route":             ["Depot"] + stops + ["Depot"],
                    "total_minutes":     round(info["total_minutes"], 1),
                    "shift_minutes":     info.get("shift_minutes", full_shift.get(tid, 28800) / 60),
                    "eff_shift_minutes": info.get("eff_shift_minutes", info.get("shift_minutes", 480)),
                    "target_minutes":    info.get("target_minutes", info.get("shift_minutes", 480) * 0.9),
                    "lunch_minutes":     round(info.get("lunch_minutes", 0), 1),
                }
            else:
                # Merge: append new stops between existing stops and final Depot
                existing_stops = [
                    s for s in all_routes[tid]["route"]
                    if s not in ("Depot", "BREAK")
                ]
                all_routes[tid]["route"] = ["Depot"] + existing_stops + stops + ["Depot"]
                all_routes[tid]["total_minutes"] = round(
                    all_routes[tid]["total_minutes"] + info["total_minutes"], 1
                )
                all_routes[tid]["lunch_minutes"] = round(
                    all_routes[tid].get("lunch_minutes", 0) + info.get("lunch_minutes", 0), 1
                )
                # shift_minutes / target_minutes stay from batch-1 (= full day shift)

    return all_routes, all_unassigned
