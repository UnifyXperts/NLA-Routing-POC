from ortools.constraint_solver import routing_enums_pb2, pywrapcp
import pandas as pd
import numpy as np
from datetime import datetime

REVENUE_SCALE = 10


def _shift_seconds(tech_row) -> int:
    """Compute available shift seconds from shift_start/shift_end."""
    try:
        s = datetime.strptime(str(tech_row["shift_start"]).strip(), "%H:%M")
        e = datetime.strptime(str(tech_row["shift_end"]).strip(), "%H:%M")
        delta_min = (e.hour * 60 + e.minute) - (s.hour * 60 + s.minute)
        return max(delta_min, 0) * 60
    except Exception:
        return int(float(tech_row.get("max_hours", 8)) * 3600)


def _build_demands(jobs: pd.DataFrame, programs: pd.DataFrame) -> list:
    # Prefer per-job material_required computed from lawn_area_sqft in data_loader.
    # Fall back to flat program-level value for backward compatibility.
    if "material_required" in jobs.columns:
        return [int(v) for v in jobs["material_required"]]
    prog_map = programs.set_index("program_id").get("material_per_1000sqft",
               programs.set_index("program_id").get("material_required", {})).to_dict()
    return [int(prog_map.get(pid, 0)) for pid in jobs["program_id"]]


def _build_capacities(technicians: pd.DataFrame, trucks: pd.DataFrame) -> list:
    truck_map = trucks.set_index("tech_id")["capacity"].to_dict()
    return [int(truck_map.get(tid, 0)) for tid in technicians["tech_id"]]


def _pre_assign_jobs(jobs_list: pd.DataFrame, elig_map: dict,
                     tech_id_to_vidx: dict, n_techs: int) -> dict:
    """
    Greedy min-load pre-assignment: each job goes to the eligible tech that currently
    has the fewest jobs.  This guarantees geographic spreading when total work is well
    below what all techs can absorb, avoiding the OR-Tools tendency to pack all jobs
    into the first few vehicles found by PATH_CHEAPEST_ARC.

    Returns: {job_array_index: vehicle_index}
    """
    tech_load = [0] * n_techs
    assignment = {}
    for j_idx in range(len(jobs_list)):
        job_id  = jobs_list.iloc[j_idx]["job_id"]
        allowed = [tech_id_to_vidx[t] for t in elig_map.get(job_id, [])
                   if t in tech_id_to_vidx]
        if allowed:
            best_v = min(allowed, key=lambda v: tech_load[v])
            assignment[j_idx] = best_v
            tech_load[best_v] += 1
    return assignment


def _insert_lunch_break(stops: list, total_min: float,
                        threshold_min: int, break_min: int) -> tuple:
    if total_min <= threshold_min or len(stops) < 2:
        return stops, 0
    mid = len(stops) // 2
    return stops[:mid] + ["BREAK"] + stops[mid:], break_min


def optimize(dist_data: dict, jobs: pd.DataFrame, technicians: pd.DataFrame,
             trucks: pd.DataFrame, programs: pd.DataFrame,
             eligibility_df: pd.DataFrame, cfg: dict,
             job_cluster_map: dict = None):
    """
    Build and solve the VRP.

    Utilization modelling:
      - shift hours are derived from shift_start/shift_end (calendar) or max_hours
      - Hard upper cap: band_high_pct * shift_seconds (Time dimension limit)
      - Soft lower bound: band_low_pct * shift_seconds (penalise under-utilisation)
      - Variance: SetGlobalSpanCostCoefficient pushes routes toward equal length

    All hard constraints (work hours, capacity) are optional via constraint_flags.

    Returns (routes, unassigned_job_ids)
      routes: {tech_id: {"route": [...], "total_minutes": N,
                          "shift_minutes": M, "target_minutes": T}}
    """
    ow     = cfg["objective_weights"]
    oc     = cfg["operational_constraints"]
    cflags = cfg.get("constraint_flags", {})
    ucfg   = cfg.get("utilization", {})

    w_revenue            = float(ow["maximize_revenue"])
    w_drive              = float(ow["minimize_drive_time"])
    w_util               = float(ow["maximize_tech_utilization"])
    service_call_penalty = float(ow.get("service_call_penalty", 500))

    use_calendar      = bool(ucfg.get("use_shift_calendar", True))
    target_pct        = float(ucfg.get("target_pct", 90)) / 100.0
    band_low_pct      = float(ucfg.get("band_low_pct", 85)) / 100.0
    band_high_pct     = float(ucfg.get("band_high_pct", 95)) / 100.0
    underutil_pen     = int(ucfg.get("underutil_penalty_per_minute", 10))
    minimize_variance = bool(ucfg.get("minimize_variance", True))
    span_coeff        = int(ucfg.get("variance_span_coefficient", 0))

    enable_hours    = bool(cflags.get("enable_work_hours", True))
    enable_capacity = bool(cflags.get("enable_capacity", True))
    enable_lunch    = bool(cflags.get("enable_lunch_break", True))
    lunch_thr       = int(oc.get("lunch_after_minutes", 240))
    lunch_dur       = int(oc.get("lunch_break_minutes", 30))

    duration_s = dist_data["duration_s"]
    n_techs    = dist_data["n_techs"]

    techs_list = technicians.reset_index(drop=True)
    jobs_list  = jobs.reset_index(drop=True)
    n_jobs     = len(jobs_list)
    n_nodes    = n_techs + n_jobs

    tech_id_to_vidx = {row["tech_id"]: i for i, (_, row) in enumerate(techs_list.iterrows())}

    service_times = (
        [0] * n_techs +
        [int(jobs_list.iloc[j]["duration_minutes"]) * 60 for j in range(n_jobs)]
    )

    # Per-tech shift seconds — two separate lists:
    #   full_shift_display_s : actual shift duration (shift_start→shift_end or max_hours fallback)
    #                          used only for display fields (shift_minutes, target_minutes).
    #   shift_s_list         : remaining time budget injected by batch_runner
    #                          (= target_pct × full_shift for batch 1, less for batch 2+)
    #                          used for the OR-Tools hard cap.
    _remaining_override: dict = cfg.get("_tech_remaining_s", {})
    shift_s_list: list[int] = []
    full_shift_display_s: list[int] = []
    for _, tech in techs_list.iterrows():
        tid = tech["tech_id"]
        full_s = (
            _shift_seconds(tech)
            if use_calendar
            else int(float(tech.get("max_hours", 8)) * 3600)
        )
        full_shift_display_s.append(full_s)
        if _remaining_override and tid in _remaining_override:
            shift_s_list.append(max(int(_remaining_override[tid]), 0))
        else:
            shift_s_list.append(full_s)

    demands      = _build_demands(jobs_list, programs)
    vehicle_caps = _build_capacities(techs_list, trucks)
    revenue_map   = dict(zip(jobs_list["job_id"], jobs_list["revenue"]))
    job_type_map  = dict(zip(jobs_list["job_id"],
                             jobs_list["job_type"] if "job_type" in jobs_list.columns else ["regular"] * len(jobs_list)))

    # Effective shift seconds: subtract lunch when it would apply to that shift.
    # This ensures OR-Tools never schedules more job+drive time than what is
    # actually available after the mandatory break.
    lunch_s_fixed = lunch_dur * 60 if enable_lunch else 0
    # eff_shift_s is shift minus lunch, used only for the utilisation soft lower bound.
    # Minimum 300 s (5 min) so OR-Tools always gets a valid positive bound even when
    # the remaining budget from a previous batch is nearly exhausted.
    eff_shift_s: list[int] = [
        max(s - (lunch_s_fixed if enable_lunch and s > lunch_thr * 60 else 0), 300)
        for s in shift_s_list
    ]

    starts  = list(range(n_techs))
    ends    = list(range(n_techs))
    manager = pywrapcp.RoutingIndexManager(n_nodes, n_techs, starts, ends)
    routing = pywrapcp.RoutingModel(manager)

    # O2: arc cost = weighted OSRM drive time
    def arc_cost_cb(fi, ti):
        fn = manager.IndexToNode(fi)
        tn = manager.IndexToNode(ti)
        return int(round(w_drive * duration_s[fn][tn]))

    arc_id = routing.RegisterTransitCallback(arc_cost_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(arc_id)

    # Time dimension: transit = service_time[from] + drive_time[from][to]
    def time_cb(fi, ti):
        fn = manager.IndexToNode(fi)
        tn = manager.IndexToNode(ti)
        return service_times[fn] + int(round(duration_s[fn][tn]))

    time_id = routing.RegisterTransitCallback(time_cb)

    if enable_hours:
        # Hard cap = budget passed in (target_pct × full_shift, injected by batch_runner
        # for every batch) minus a lunch reservation when the shift is long enough.
        # After adding lunch back in post-processing: drive+service+lunch
        # ≤ target_pct × actual_shift_seconds for that technician.
        max_times_s = [
            max(
                shift_s_list[v]
                - (lunch_s_fixed if enable_lunch and shift_s_list[v] > lunch_thr * 60 else 0),
                300,
            )
            for v in range(n_techs)
        ]
    else:
        max_times_s = [int(24 * 3600)] * n_techs  # effectively no limit

    routing.AddDimensionWithVehicleCapacity(time_id, 0, max_times_s, True, "Time")
    time_dim = routing.GetDimensionOrDie("Time")

    # O4: Soft utilisation — penalise routes that end below band_low.
    # Only a LOWER bound is used here.  Adding a soft upper bound with the same
    # high per-second coefficient would make staying below target far cheaper
    # than dropping a job, causing mass job-drops on any dense schedule.
    # The hard upper cap (band_high_pct via max_times_s) already enforces the
    # ceiling; the soft lower bound nudges idle techs to absorb more work.
    if w_util > 0:
        pen_per_s = max(1, int(w_util * underutil_pen))
        for v in range(n_techs):
            end_idx = routing.End(v)
            low_s   = int(band_low_pct * eff_shift_s[v])
            time_dim.SetCumulVarSoftLowerBound(end_idx, low_s, pen_per_s)

        # Variance minimisation: penalises the gap between the longest and
        # shortest route.  Keep coefficient small — span_coeff × max_route_time
        # must stay below the minimum disjunction penalty to avoid job drops.
        # Safe range for Richmond routes: coefficient ≤ 1.  Default 0 (off).
        if minimize_variance and span_coeff > 0:
            time_dim.SetGlobalSpanCostCoefficient(span_coeff)

    # Capacity dimension
    def demand_cb(fi):
        node = manager.IndexToNode(fi)
        return 0 if node < n_techs else demands[node - n_techs]

    demand_id = routing.RegisterUnaryTransitCallback(demand_cb)
    if enable_capacity:
        routing.AddDimensionWithVehicleCapacity(demand_id, 0, vehicle_caps, True, "Capacity")

    # O1 + eligibility: disjunction penalty + allowed vehicles per job
    elig_map = eligibility_df.set_index("job_id")["eligible_technicians"].to_dict()

    # Pre-assign when work density is sparse (< 10 jobs/tech) to force distribution.
    # Without this, PATH_CHEAPEST_ARC packs all jobs into the first 3-4 vehicles.
    use_pre_assign = (n_jobs / max(n_techs, 1)) < 10
    pre_assign = (
        _pre_assign_jobs(jobs_list, elig_map, tech_id_to_vidx, n_techs)
        if use_pre_assign else {}
    )

    # Penalty floor: when a job is locked to a specific tech via pre-assignment, its
    # disjunction penalty must exceed the worst-case round-trip arc cost so the solver
    # never drops it purely due to drive overhead.
    # Max one-way drive in the Richmond service area ≈ 45 min → 2× = 90 min = 5400s.
    route_overhead_s = 5400
    min_disjunction_penalty = int(w_drive * route_overhead_s * 2) if pre_assign else 0

    for j_idx in range(n_jobs):
        job_id        = jobs_list.iloc[j_idx]["job_id"]
        node          = n_techs + j_idx
        index         = manager.NodeToIndex(node)
        revenue   = revenue_map.get(job_id, 0)
        job_type  = job_type_map.get(job_id, "regular")
        # Service calls have $0 billing revenue but must not be dropped freely.
        # Treat them as worth at least service_call_penalty dollars so the solver
        # accepts detours up to the same cost it would pay for a high-value regular job.
        effective_revenue = max(revenue, service_call_penalty) if job_type == "service_call" else revenue
        base_penalty      = int(round(w_revenue * effective_revenue * REVENUE_SCALE))
        penalty           = max(base_penalty, min_disjunction_penalty)
        eligible_tids = elig_map.get(job_id, [])
        allowed_v     = [tech_id_to_vidx[t] for t in eligible_tids if t in tech_id_to_vidx]

        # Enforce k-means cluster: restrict to the cluster-assigned tech if they are eligible
        if job_cluster_map and job_id in job_cluster_map:
            cluster_tid  = job_cluster_map[job_id]
            cluster_vidx = tech_id_to_vidx.get(cluster_tid)
            if cluster_vidx is not None and cluster_vidx in allowed_v:
                allowed_v = [cluster_vidx]

        if not allowed_v:
            routing.AddDisjunction([index], 0)
        else:
            if j_idx in pre_assign:
                # Lock to pre-assigned vehicle; -1 still allows drop if truly infeasible
                routing.VehicleVar(index).SetValues([-1, pre_assign[j_idx]])
            else:
                routing.VehicleVar(index).SetValues([-1] + allowed_v)
            routing.AddDisjunction([index], penalty)

    params = pywrapcp.DefaultRoutingSearchParameters()
    # LOCAL_CHEAPEST_INSERTION is required when all nodes are optional (disjunctions).
    # PATH_CHEAPEST_ARC greedily picks arc(depot→depot)=0 over any job arc, producing
    # an all-empty first solution that GLS cannot escape from.  LCI compares insertion
    # cost vs. disjunction penalty and correctly inserts jobs whenever penalty > arc delta.
    params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.LOCAL_CHEAPEST_INSERTION
    )
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    params.time_limit.seconds = int(oc.get("solver_time_limit", 60))

    solution = routing.SolveWithParameters(params)
    if not solution:
        return None, list(jobs_list["job_id"])

    time_dim = routing.GetDimensionOrDie("Time")
    routes   = {}
    served   = set()

    for v in range(n_techs):
        tid   = techs_list.iloc[v]["tech_id"]
        index = routing.Start(v)
        stops = []

        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            if node >= n_techs:
                stops.append(jobs_list.iloc[node - n_techs]["job_id"])
            index = solution.Value(routing.NextVar(index))

        end_idx   = routing.End(v)
        total_s   = solution.Value(time_dim.CumulVar(end_idx))
        total_min = total_s / 60.0

        if enable_lunch:
            stops_final, extra_min = _insert_lunch_break(stops, total_min, lunch_thr, lunch_dur)
        else:
            stops_final, extra_min = stops, 0

        served.update(s for s in stops_final if s != "BREAK")
        routes[tid] = {
            "route":              ["Depot"] + stops_final + ["Depot"],
            "total_minutes":      round(total_min + extra_min, 1),
            "shift_minutes":      round(full_shift_display_s[v] / 60.0, 1),  # actual full shift
            "eff_shift_minutes":  round(eff_shift_s[v] / 60.0, 1),           # remaining budget − lunch
            "target_minutes":     round(target_pct * full_shift_display_s[v] / 60.0, 1),  # 90% of full shift
            "lunch_minutes":      round(extra_min, 1),
        }

    unassigned = sorted(set(jobs_list["job_id"]) - served)
    return routes, unassigned


def print_routes(routes, unassigned):
    if routes is None:
        print("No solution found.")
        return
    print("=== Phase III: Optimized Routes ===\n")
    for tid, info in routes.items():
        shift_min  = info.get("shift_minutes", "?")
        target_min = info.get("target_minutes", "?")
        util_pct   = round(info["total_minutes"] / shift_min * 100, 1) if shift_min else "?"
        print(f"Tech {tid}  ({info['total_minutes']} / {shift_min} min  |  {util_pct}%  |  target {target_min} min)")
        for stop in info["route"]:
            print(f"  {stop}")
        print()

    if unassigned:
        print(f"Unassigned ({len(unassigned)}): {unassigned}")
    else:
        print("All jobs assigned.")
