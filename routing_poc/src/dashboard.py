import pandas as pd


_OBJECTIVE_LABEL = {
    "maximize_revenue":          "Maximize Revenue",
    "minimize_drive_time":       "Minimize Drive Time",
    "meet_customer_preference":  "Honor Customer Preferences",
    "maximize_tech_utilization": "Maximize Tech Utilization",
}


def _detect_mode(weights: dict) -> str:
    w = {k: float(v) for k, v in weights.items()}
    max_w = max(w.values())
    dominant = [k for k, v in w.items() if v == max_w]
    if len(dominant) == 4 or (max_w == 1.0 and len(set(w.values())) == 1):
        return "Balanced (all objectives weighted equally)"
    label = _OBJECTIVE_LABEL.get(dominant[0], dominant[0])
    return f"'{label}'-focused (weight {max_w}x)"


def build_optimization_summary(
    jobs: pd.DataFrame,
    priority_queue: pd.DataFrame,
    technicians: pd.DataFrame,
    routes: dict,
    unassigned: list,
    dist_data: dict,
    cfg: dict,
    programs: pd.DataFrame = None,
    eligibility_df: pd.DataFrame = None,
) -> str:
    """Return a human-readable narrative explaining the optimisation outcome."""
    ow           = cfg["objective_weights"]
    cflags       = cfg.get("constraint_flags", {})
    ucfg         = cfg.get("utilization", {})
    tech_index   = dist_data["tech_index"]
    job_index    = dist_data["job_index"]
    duration_s   = dist_data["duration_s"]

    job_rev_map   = dict(zip(jobs["job_id"], jobs["revenue"]))
    job_dur_map   = dict(zip(jobs["job_id"], jobs["duration_minutes"]))
    job_cust_map  = dict(zip(jobs["job_id"], jobs["customer_name"]))
    job_prog_map  = dict(zip(jobs["job_id"], jobs["program_id"]))
    job_pref_tech = dict(zip(
        jobs["job_id"],
        jobs["preferred_tech_id"].fillna("").astype(str).str.strip(),
    ))
    tech_max_map  = dict(zip(technicians["tech_id"], technicians["max_hours"]))
    tech_name_map = dict(zip(technicians["tech_id"], technicians["name"]))

    prog_svc_map: dict = {}
    if programs is not None:
        rate_col = (
            "minutes_per_1000sqft" if "minutes_per_1000sqft" in programs.columns
            else "service_time_minutes" if "service_time_minutes" in programs.columns
            else None
        )
        if rate_col:
            prog_svc_map = dict(zip(programs["program_id"], programs[rate_col]))

    lines: list = []
    sep = "-" * 58

    def h(title: str) -> None:
        lines.append("")
        lines.append(sep)
        lines.append(f"  {title}")
        lines.append(sep)

    # ── Section 1: Objective Configuration ────────────────────────────────────
    h("1. OBJECTIVE CONFIGURATION")
    mode = _detect_mode(ow)
    lines.append(f"  Mode : {mode}")
    lines.append("")
    weights_sorted = sorted(ow.items(), key=lambda x: -float(x[1]))
    for key, val in weights_sorted:
        bar = "#" * int(float(val) * 4)
        lines.append(f"  {_OBJECTIVE_LABEL.get(key, key):<30s}  w={float(val):.1f}  {bar}")
    lines.append("")

    w_rev  = float(ow.get("maximize_revenue", 1))
    w_drv  = float(ow.get("minimize_drive_time", 1))
    w_pref = float(ow.get("meet_customer_preference", 1))
    w_util = float(ow.get("maximize_tech_utilization", 1))

    if w_rev > max(w_drv, w_pref, w_util):
        lines.append("  Reasoning: Revenue dominates — the solver strongly prefers")
        lines.append("  including high-value jobs even if they add drive time.")
    elif w_drv > max(w_rev, w_pref, w_util):
        lines.append("  Reasoning: Drive time dominates — the solver tightens routes")
        lines.append("  geographically, accepting lower revenue for fuel savings.")
    elif w_pref > max(w_rev, w_drv, w_util):
        lines.append("  Reasoning: Customer preference dominates — preferred techs")
        lines.append("  and time windows are honored wherever possible.")
    else:
        lines.append("  Reasoning: All objectives carry equal influence. The solver")
        lines.append("  balances revenue, efficiency, preferences, and utilization.")

    # ── Section 1b: Utilization Settings ──────────────────────────────────────
    if ucfg:
        lines.append("")
        lines.append(f"  Utilization target : {ucfg.get('target_pct', 90)}%"
                     f"  (band {ucfg.get('band_low_pct', 85)}%–{ucfg.get('band_high_pct', 95)}%)")
        cal = "shift calendar" if ucfg.get("use_shift_calendar", True) else "flat max_hours"
        lines.append(f"  Capacity source    : {cal}")
        var = "on" if ucfg.get("minimize_variance", True) else "off"
        lines.append(f"  Variance min.      : {var}")

    # ── Section 1c: Active Constraint Flags ───────────────────────────────────
    if cflags:
        lines.append("")
        lines.append("  Constraints active:")
        flag_labels = {
            "enable_work_hours":         "Work-hour limit",
            "enable_capacity":           "Truck capacity",
            "enable_location_threshold": "Location threshold",
            "enable_lunch_break":        "Lunch break",
            "enable_skill_check":        "Skill check",
            "enable_license_check":      "License check",
            "enable_equipment_check":    "Equipment check",
            "enable_availability_check": "Availability check",
        }
        for flag, label in flag_labels.items():
            state = "ON " if cflags.get(flag, True) else "OFF"
            lines.append(f"    [{state}]  {label}")

    # ── Section 2: Program Service-Time Baselines ──────────────────────────────
    if prog_svc_map:
        h("2. PROGRAM SERVICE-TIME RATES")
        lines.append("  Duration is computed per-job: lawn_area_sqft / 1000 * rate.")
        lines.append("")
        prog_descs = {
            "FERT": "Fertilizer - load spreader, walk property, clean up.",
            "MOSQ": "Mosquito   - mix solution, spray perimeter & shrubs.",
            "AER":  "Aeration   - transport aerator, full coverage pass, tidy up.",
        }
        for pid, svc in sorted(prog_svc_map.items()):
            desc = prog_descs.get(pid, pid)
            lines.append(f"  {pid:<6s}  {svc:>3} min/1,000 sqft   {desc}")

    # ── Section 3: Queue Prioritization ───────────────────────────────────────
    h("3. QUEUE PRIORITIZATION  (Phase I output)")
    n_new  = int(priority_queue["new_customer"].sum())
    n_asap = int(priority_queue["asap"].sum())
    n_svc  = int((priority_queue["job_type"] == "service_call").sum())
    n_reg  = int((priority_queue["job_type"] == "regular").sum())
    lines.append(f"  Total jobs in queue : {len(priority_queue)}")
    lines.append(
        f"  New customers       : {n_new}"
        f"  -> top of queue (weight {cfg['priority_weights']['new_customer']})"
    )
    lines.append(
        f"  ASAP-flagged        : {n_asap}"
        f"  -> second tier (weight {cfg['priority_weights']['asap']})"
    )
    lines.append(
        f"  Service calls       : {n_svc}"
        f"  -> ahead of regular within each tier"
    )
    lines.append(f"  Regular maintenance : {n_reg}")
    lines.append("")
    lines.append("  Top 5 priority jobs:")
    top5 = priority_queue.head(5)
    for _, row in top5.iterrows():
        reasons = []
        if row["new_customer"]:
            reasons.append("new customer")
        if row["asap"]:
            reasons.append("ASAP")
        if row["job_type"] == "service_call":
            reasons.append("service call")
        if row["last_service_days"] > 0:
            reasons.append(f"{int(row['last_service_days'])}d overdue")
        reason_str = ", ".join(reasons) if reasons else "standard scheduling"
        lines.append(
            f"    #{int(row['priority_rank']):<3d}  {row['job_id']}  "
            f"{row['customer_name']:<22s}  [{reason_str}]"
        )

    # ── Section 4: Per-Technician Route Breakdown ──────────────────────────────
    h("4. PER-TECHNICIAN ROUTE BREAKDOWN")
    total_assigned    = 0
    total_revenue     = 0.0
    all_util_pct      = []
    total_drive_s_all = 0.0
    total_svc_min_all = 0.0

    for tid, info in routes.items():
        route     = info["route"]
        t_idx     = tech_index[tid]
        stops     = [s for s in route if s not in ("Depot", "BREAK")]
        has_break = "BREAK" in route
        total_min = info["total_minutes"]

        # Prefer calendar shift_minutes from route output over flat max_hours
        shift_min  = info.get("shift_minutes") or tech_max_map.get(tid, 8) * 60
        target_min = info.get("target_minutes") or shift_min * 0.9
        util_pct   = min(100.0, round(total_min / shift_min * 100, 1)) if shift_min else 0
        all_util_pct.append(util_pct)
        total_assigned += len(stops)

        # Drive time for this tech
        prev_i  = t_idx
        drive_s = 0.0
        for stop in route:
            if stop == "Depot":
                cur_i = t_idx
            elif stop == "BREAK":
                continue
            else:
                cur_i = job_index[stop]
            if prev_i != cur_i:
                drive_s += duration_s[prev_i][cur_i]
            prev_i = cur_i
        total_drive_s_all += drive_s

        svc_min = sum(job_dur_map.get(s, 0) for s in stops)
        total_svc_min_all += svc_min
        rev = sum(job_rev_map.get(s, 0) for s in stops)
        total_revenue += rev

        pref_vio = sum(
            1 for s in stops
            if job_pref_tech.get(s, "") not in ("", tid)
        )

        # Utilisation band indicator
        target_pct_val = ucfg.get("target_pct", 90)
        band_low       = ucfg.get("band_low_pct", 85)
        band_high      = ucfg.get("band_high_pct", 95)
        if util_pct >= band_low and util_pct <= band_high:
            band_note = f"[OK  band {band_low}%-{band_high}%]"
        elif util_pct < band_low:
            band_note = f"[LOW target {target_pct_val}%]"
        else:
            band_note = f"[HIGH target {target_pct_val}%]"

        name = tech_name_map.get(tid, tid)
        lines.append(f"  {tid} - {name}")
        brk_note = "  (+ lunch break)" if has_break else ""
        lines.append(f"    Stops          : {len(stops)}{brk_note}")
        lines.append(
            f"    Route time     : {total_min:.0f} / {shift_min:.0f} min"
            f"  ({drive_s/60:.0f} min drive + {svc_min} min service)"
        )
        lines.append(
            f"    Utilization    : {util_pct}%  {band_note}"
        )
        lines.append(f"    Revenue        : ${rev:,.2f}")
        if pref_vio:
            lines.append(
                f"    Pref. violations: {pref_vio} job(s) not with preferred tech"
            )
        else:
            lines.append("    Preferences    : all honored")
        for stop in stops:
            cust      = job_cust_map.get(stop, stop)
            prog      = job_prog_map.get(stop, "")
            pref      = job_pref_tech.get(stop, "")
            pref_note = f"  [pref:{pref}]" if pref and pref != tid else ""
            lines.append(
                f"      {stop}  {cust:<22s}  {prog:<5s}"
                f"  {job_dur_map.get(stop,0):>2} min"
                f"  ${job_rev_map.get(stop,0):.2f}"
                + pref_note
            )
        lines.append("")

    # ── Section 5: Fleet Efficiency ────────────────────────────────────────────
    h("5. FLEET EFFICIENCY")
    avg_util  = round(sum(all_util_pct) / len(all_util_pct), 1) if all_util_pct else 0
    util_var  = (
        round(max(all_util_pct) - min(all_util_pct), 1) if len(all_util_pct) > 1 else 0
    )
    denom     = total_drive_s_all / 60 + total_svc_min_all
    drive_pct = round(total_drive_s_all / 60 / denom * 100, 1) if denom > 0 else 0
    svc_pct   = 100.0 - drive_pct

    total_shift_h = sum(
        info.get("shift_minutes", tech_max_map.get(tid, 8) * 60) / 60
        for tid, info in routes.items()
    )
    rev_per_hour = round(total_revenue / (total_assigned / max(len(routes), 1) * avg_util / 100 * total_shift_h / max(len(routes),1)), 2) if avg_util > 0 else 0

    lines.append(f"  Active technicians : {len(routes)}")
    lines.append(f"  Assigned jobs      : {total_assigned}")
    lines.append(f"  Unassigned jobs    : {len(unassigned)}")
    lines.append(f"  Total revenue      : ${total_revenue:,.2f}")
    lines.append(f"  Avg utilization    : {avg_util}%")
    lines.append(f"  Utilization spread : {util_var}% (max–min gap)")
    lines.append(
        f"  Time split         : {svc_pct:.0f}% productive service"
        f"  /  {drive_pct:.0f}% driving"
    )
    lines.append("")

    target_pct_val = ucfg.get("target_pct", 90)
    band_low       = ucfg.get("band_low_pct", 85)
    if avg_util >= band_low:
        lines.append(f"  Assessment: Fleet utilization is within or above target band.")
    elif avg_util >= 70:
        lines.append("  Assessment: Good utilization. Minor idle windows exist —")
        lines.append("  consider adding nearby jobs or adjusting objective weights.")
    else:
        lines.append("  Assessment: Utilization below target. Job volume may be")
        lines.append("  insufficient or eligibility filters are too restrictive.")

    if util_var <= 10:
        lines.append("  Variance is low — utilization is balanced across technicians.")
    elif util_var <= 20:
        lines.append("  Moderate variance — some techs are running longer than others.")
    else:
        lines.append(
            "  High variance — consider enabling 'minimize variance' or "
            "adjusting geo-clustering."
        )

    if drive_pct <= 20:
        lines.append("  Route density is high — stops are geographically tight.")
    elif drive_pct <= 35:
        lines.append("  Drive time is moderate — typical for a dispersed service area.")
    else:
        lines.append("  High drive ratio — consider tighter geographic clustering.")

    # ── Section 6: Unassigned Jobs ─────────────────────────────────────────────
    h("6. UNASSIGNED JOBS")
    if unassigned:
        elig_reason_map: dict = {}
        elig_techs_map:  dict = {}
        if eligibility_df is not None and not eligibility_df.empty:
            for _, row in eligibility_df.iterrows():
                elig_reason_map[row["job_id"]] = row.get("rejection_reason", "")
                elig_techs_map[row["job_id"]]  = row.get("eligible_technicians", [])

        tech_util_map: dict = {}
        for tid, info in routes.items():
            shift_min = info.get("shift_minutes") or tech_max_map.get(tid, 8) * 60
            used_min  = info["total_minutes"]
            tech_util_map[tid] = round(used_min / shift_min * 100, 1) if shift_min else 0

        lines.append(f"  {len(unassigned)} job(s) not included in today's routes.")
        lines.append("")

        for jid in unassigned:
            cust     = job_cust_map.get(jid, jid)
            prog     = job_prog_map.get(jid, "")
            rev      = job_rev_map.get(jid, 0)
            rank_row = priority_queue[priority_queue["job_id"] == jid]
            rank     = int(rank_row["priority_rank"].iloc[0]) if not rank_row.empty else "?"

            p2_reason  = elig_reason_map.get(jid, "")
            elig_techs = elig_techs_map.get(jid, [])

            if p2_reason:
                reason = f"Phase II: {p2_reason}"
            elif not elig_techs:
                reason = "Phase II: no technician passed all eligibility checks"
            else:
                busy_techs = [t for t in elig_techs if tech_util_map.get(t, 0) >= 90]
                if len(busy_techs) == len(elig_techs):
                    reason = (
                        f"Solver: all {len(elig_techs)} eligible tech(s) near/at capacity"
                        f" ({', '.join(f'{t}={tech_util_map[t]:.0f}%' for t in elig_techs)})"
                    )
                else:
                    reason = (
                        f"Solver trade-off: revenue ${rev:.2f} did not outweigh"
                        f" drive-time cost vs. already-assigned jobs"
                        f" (eligible: {', '.join(elig_techs)})"
                    )

            lines.append(f"  {jid}  {cust:<22s}  {prog:<5s}  ${rev:.2f}  priority #{rank}")
            lines.append(f"    Reason: {reason}")

        lines.append("")
        lines.append("  Actions to reduce unassigned jobs:")
        lines.append("  - Disable location threshold to allow all depots")
        lines.append("  - Raise maximize_revenue weight to include lower-margin stops")
        lines.append("  - Extend max_hours_per_tech if techs have headroom")
    else:
        lines.append("  All jobs successfully assigned — full fleet coverage achieved.")

    lines.append("")
    return "\n".join(lines)


def build_kpi(jobs: pd.DataFrame, technicians: pd.DataFrame,
              routes: dict, dist_data: dict) -> pd.DataFrame:
    tech_index = dist_data["tech_index"]
    job_index  = dist_data["job_index"]
    distance_m = dist_data["distance_m"]
    duration_s = dist_data["duration_s"]

    job_rev_map    = dict(zip(jobs["job_id"], jobs["revenue"]))
    job_pref_tech  = dict(zip(jobs["job_id"], jobs["preferred_tech_id"].fillna("")))
    tech_max_hours = dict(zip(technicians["tech_id"], technicians["max_hours"]))

    total_jobs        = len(jobs)
    assigned_ids: set = set()
    total_distance_m  = 0.0
    total_drive_s     = 0.0
    tech_stop_counts  = []
    pref_violations   = 0
    total_idle_s      = 0.0
    total_util_pct    = []

    for tid, info in routes.items():
        route   = info["route"]
        t_idx   = tech_index[tid]
        t_stops = [s for s in route if s not in ("Depot", "BREAK")]
        tech_stop_counts.append(len(t_stops))
        assigned_ids.update(t_stops)

        prev_idx = t_idx
        for stop in route:
            if stop == "Depot":
                cur_idx = t_idx
            elif stop == "BREAK":
                continue
            else:
                cur_idx = job_index[stop]

            if prev_idx != cur_idx:
                total_distance_m += distance_m[prev_idx][cur_idx]
                total_drive_s    += duration_s[prev_idx][cur_idx]
            prev_idx = cur_idx

        # Preference violations
        for job_id in t_stops:
            pref = str(job_pref_tech.get(job_id, "")).strip()
            if pref and pref != tid:
                pref_violations += 1

        # Utilisation (using calendar shift_minutes when available)
        route_s   = info["total_minutes"] * 60
        shift_min = info.get("shift_minutes") or tech_max_hours.get(tid, 8) * 60
        shift_s   = shift_min * 60
        idle_s    = max(0.0, shift_s - route_s)
        total_idle_s += idle_s
        total_util_pct.append(round(route_s / shift_s * 100, 1) if shift_s else 0)

    assigned_jobs   = len(assigned_ids)
    unassigned_jobs = total_jobs - assigned_jobs
    total_revenue   = sum(job_rev_map.get(j, 0) for j in assigned_ids)
    avg_stops       = round(sum(tech_stop_counts) / len(tech_stop_counts), 1) if tech_stop_counts else 0
    total_drive_min = round(total_drive_s / 60, 1)
    total_idle_hrs  = round(total_idle_s / 3600, 2)
    avg_util_pct    = round(sum(total_util_pct) / len(total_util_pct), 1) if total_util_pct else 0
    util_spread     = round(max(total_util_pct) - min(total_util_pct), 1) if len(total_util_pct) > 1 else 0

    kpi = pd.DataFrame([
        {"Metric": "Total Jobs",                   "Value": total_jobs},
        {"Metric": "Assigned Jobs",                "Value": assigned_jobs},
        {"Metric": "Unassigned Jobs",              "Value": unassigned_jobs},
        {"Metric": "Total Revenue ($)",            "Value": f"${total_revenue:,}"},
        {"Metric": "Total Distance (km)",          "Value": round(total_distance_m / 1000, 2)},
        {"Metric": "Total Drive Time (min)",       "Value": total_drive_min},
        {"Metric": "Avg Stops / Technician",       "Value": avg_stops},
        {"Metric": "Avg Utilization (%)",          "Value": avg_util_pct},
        {"Metric": "Utilization Spread (%)",       "Value": util_spread},
        {"Metric": "Preference Violations (O3)",   "Value": pref_violations},
        {"Metric": "Total Idle Time (hrs) (O4)",   "Value": total_idle_hrs},
    ])
    return kpi
