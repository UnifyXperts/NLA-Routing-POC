import pandas as pd
from datetime import date, datetime


def _parse_time(t_str: str) -> datetime:
    return datetime.strptime(t_str.strip(), "%H:%M")


def _window_within_shift(preferred_window: str, shift_start: str, shift_end: str) -> bool:
    """Return True if preferred_window falls entirely inside the shift."""
    if not preferred_window:
        return True
    try:
        pw_start, pw_end = preferred_window.split("-")
        p0 = _parse_time(pw_start)
        p1 = _parse_time(pw_end)
        s0 = _parse_time(shift_start)
        s1 = _parse_time(shift_end)
        return s0 <= p0 and p1 <= s1
    except Exception:
        return True  # malformed window — don't block the tech


def _location_eligible(tech_idx: int, n_techs: int, n_jobs: int,
                       duration_s, threshold_s: float) -> bool:
    """True if the tech can reach at least one job within threshold_s seconds."""
    min_time = min(
        duration_s[tech_idx][n_techs + j] for j in range(n_jobs)
    )
    return min_time <= threshold_s


def match_resources(jobs: pd.DataFrame, technicians: pd.DataFrame,
                    trucks: pd.DataFrame, programs: pd.DataFrame,
                    dist_data: dict, cfg: dict, route_date: date,
                    job_cluster_map: dict = None) -> pd.DataFrame:
    """
    For each job, find all eligible (tech_id) entries.

    Each check can be disabled individually via cfg['constraint_flags'].
    All checks default to enabled when the flag is absent.

    Checks (all optional via flags):
      1. Location threshold: OSRM drive time depot → any job <= threshold
      2. Skill match
      3. License match
      4. Availability: day-of-week + time window vs shift
      5. Equipment match (truck) — handles comma-separated multi-item requirements
      6. Capacity: truck capacity >= material_required

    Note: job_cluster_map is accepted for API compatibility but is NOT used as a
    hard filter here — it caused 0 eligible techs when geographic zones didn't align
    with skill requirements. The optimizer's drive-time objective naturally clusters
    jobs geographically without blocking eligibility.
    """
    duration_s = dist_data["duration_s"]
    tech_index = dist_data["tech_index"]
    n_techs    = dist_data["n_techs"]
    n_jobs     = len(jobs)

    cflags       = cfg.get("constraint_flags", {})
    enable_loc   = bool(cflags.get("enable_location_threshold", True))
    enable_skill = bool(cflags.get("enable_skill_check", True))
    enable_lic   = bool(cflags.get("enable_license_check", True))
    enable_equip = bool(cflags.get("enable_equipment_check", True))
    enable_avail = bool(cflags.get("enable_availability_check", True))
    enable_cap   = bool(cflags.get("enable_capacity", True))

    threshold_s = cfg["operational_constraints"]["location_threshold_minutes"] * 60
    route_day   = route_date.strftime("%a")

    prog_map  = programs.set_index("program_id").to_dict("index")
    truck_map = trucks.set_index("tech_id").to_dict("index")

    # Pre-compute per-tech location eligibility once (cluster-level)
    if enable_loc:
        loc_eligible = {
            tech["tech_id"]: _location_eligible(
                tech_index[tech["tech_id"]], n_techs, n_jobs, duration_s, threshold_s
            )
            for _, tech in technicians.iterrows()
        }
    else:
        loc_eligible = {tech["tech_id"]: True for _, tech in technicians.iterrows()}

    results = []
    for _, job in jobs.iterrows():
        pid  = job["program_id"]
        prog = prog_map.get(pid)
        if prog is None:
            results.append({"job_id": job["job_id"], "program_id": pid,
                             "eligible_technicians": [], "eligible_count": 0,
                             "rejection_reason": "unknown program"})
            continue

        req_skill   = prog.get("required_skill", "")
        req_license = prog.get("required_license", "")
        req_equip_raw = prog.get("required_equipment", "")
        # Parse multi-item equipment requirements (e.g. "Aerator,Seeder" → ["Aerator","Seeder"])
        req_equip_items = [e.strip() for e in str(req_equip_raw).split(",") if e.strip()] if req_equip_raw else []
        req_mat     = float(prog.get("material_required", 0))
        pref_window = str(job.get("preferred_time_window", "")).strip()

        eligible   = []
        rejections = []

        for _, tech in technicians.iterrows():
            tid = tech["tech_id"]

            # 1. Location threshold
            if enable_loc and not loc_eligible.get(tid, False):
                rejections.append(f"depot too far from job cluster (>{int(threshold_s/60)} min)")
                continue

            # 2. Skill check
            if enable_skill and req_skill and req_skill not in tech["skills"]:
                rejections.append(f"missing skill '{req_skill}'")
                continue

            # 3. License check
            if enable_lic and req_license and req_license not in tech["licenses"]:
                rejections.append(f"missing license '{req_license}'")
                continue

            # 4a. Availability — day of week
            if enable_avail and route_day not in tech["available_days"]:
                rejections.append(f"not available on {route_day}")
                continue

            # 4b. Availability — preferred time window must fit within shift
            if enable_avail and not _window_within_shift(
                pref_window, tech["shift_start"], tech["shift_end"]
            ):
                rejections.append("preferred time window outside tech shift")
                continue

            # Truck checks
            truck = truck_map.get(tid)
            if truck is None:
                rejections.append("no truck assigned")
                continue

            # 5. Equipment check — all required items must be present on the truck
            truck_equip = (
                truck["equipment"] if isinstance(truck["equipment"], list)
                else [e.strip() for e in str(truck["equipment"]).split(",")]
            )
            if enable_equip and req_equip_items:
                missing = [e for e in req_equip_items if e not in truck_equip]
                if missing:
                    rejections.append(f"truck missing equipment {missing}")
                    continue

            # 6. Capacity check
            if enable_cap and truck["capacity"] < req_mat:
                rejections.append(
                    f"truck capacity {truck['capacity']} < required {int(req_mat)}"
                )
                continue

            eligible.append(tid)

        # Most common rejection reason summary
        rejection_reason = ""
        if not eligible and rejections:
            from collections import Counter
            top_reason, count = Counter(rejections).most_common(1)[0]
            n_checked = len(rejections)
            rejection_reason = f"{top_reason} ({count}/{n_checked} techs)"

        results.append({
            "job_id":               job["job_id"],
            "program_id":           pid,
            "eligible_technicians": eligible,
            "eligible_count":       len(eligible),
            "rejection_reason":     rejection_reason,
        })

    return pd.DataFrame(results)


def print_matches(df: pd.DataFrame) -> None:
    print("\n=== Phase II: Eligible Resources ===")
    for _, row in df.iterrows():
        techs = ", ".join(row["eligible_technicians"]) if row["eligible_technicians"] else "NONE"
        print(f"  {row['job_id']} ({row['program_id']}): {techs}")
    print("====================================\n")
