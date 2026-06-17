import math
import pandas as pd
import os

# Jobs: duration_minutes is derived at load-time from lawn_area_sqft × program rate.
# programs: material_per_1000sqft and minutes_per_1000sqft replace flat columns.
REQUIRED_COLUMNS = {
    "jobs": [
        "job_id", "customer_name", "lat", "lng", "revenue",
        "lawn_area_sqft", "new_customer", "asap", "last_service_days",
        "program_id", "job_type", "preferred_tech_id", "preferred_time_window",
    ],
    "technicians": [
        "tech_id", "name", "start_lat", "start_lng", "max_hours",
        "skills", "licenses", "available_days", "shift_start", "shift_end",
    ],
    "trucks": ["truck_id", "tech_id", "capacity", "equipment"],
    "programs": [
        "program_id", "required_skill", "required_license",
        "required_equipment", "material_per_1000sqft", "minutes_per_1000sqft",
    ],
}

# Minimum service time so no job gets rounded to 0 minutes.
MIN_DURATION_MINUTES = 5


def _parse_list(value) -> list:
    if pd.isna(value) or str(value).strip() == "":
        return []
    return [v.strip() for v in str(value).split(",") if v.strip()]


def load_csv(path: str, name: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS[name] if c not in df.columns]
    if missing:
        raise ValueError(f"{name}.csv missing columns: {missing}")
    return df


def _compute_job_fields(jobs: pd.DataFrame, programs: pd.DataFrame) -> pd.DataFrame:
    """
    Derive duration_minutes and material_required per job from lawn_area_sqft.

    Formula:
        duration_minutes  = ceil(lawn_area_sqft / 1000 × minutes_per_1000sqft)
        material_required = round(lawn_area_sqft / 1000 × material_per_1000sqft)

    Both values are clamped to a minimum of MIN_DURATION_MINUTES / 0 respectively.
    """
    rate_map = programs.set_index("program_id")[
        ["minutes_per_1000sqft", "material_per_1000sqft"]
    ].to_dict("index")

    durations = []
    materials = []
    for _, row in jobs.iterrows():
        rates   = rate_map.get(row["program_id"], {})
        area_k  = float(row["lawn_area_sqft"]) / 1000.0
        dur     = math.ceil(area_k * float(rates.get("minutes_per_1000sqft", 0)))
        mat     = round(area_k * float(rates.get("material_per_1000sqft", 0)))
        durations.append(max(dur, MIN_DURATION_MINUTES))
        materials.append(max(mat, 0))

    jobs = jobs.copy()
    jobs["duration_minutes"]  = durations
    jobs["material_required"] = materials
    return jobs


def load_all(data_dir: str) -> dict:
    data = {name: load_csv(os.path.join(data_dir, f"{name}.csv"), name)
            for name in REQUIRED_COLUMNS}

    # Normalise optional string columns
    for col in ("preferred_tech_id", "preferred_time_window"):
        data["jobs"][col] = data["jobs"][col].fillna("").astype(str).str.strip()

    # Parse list-valued columns into actual Python lists
    for col in ("skills", "licenses", "available_days"):
        data["technicians"][col] = data["technicians"][col].apply(_parse_list)

    data["trucks"]["equipment"] = data["trucks"]["equipment"].apply(_parse_list)

    for col in ("required_skill", "required_license", "required_equipment"):
        data["programs"][col] = data["programs"][col].fillna("").astype(str).str.strip()

    # Compute area-based duration and material per job
    data["jobs"] = _compute_job_fields(data["jobs"], data["programs"])

    return data


def print_summary(data: dict) -> None:
    summary = {
        "Total Jobs":        len(data["jobs"]),
        "Total Technicians": len(data["technicians"]),
        "Total Trucks":      len(data["trucks"]),
        "Total Programs":    len(data["programs"]),
    }
    print("\n=== Dataset Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # Show area → duration breakdown per program
    prog = data["programs"].set_index("program_id")
    print("\n  Service-time rates (from programs.csv):")
    for pid, row in prog.iterrows():
        print(f"    {pid}: {row['minutes_per_1000sqft']} min / 1,000 sqft"
              f"  |  {row['material_per_1000sqft']} units / 1,000 sqft")

    # Show a few computed durations
    print("\n  Sample job durations (computed from lawn area):")
    for _, row in data["jobs"].head(5).iterrows():
        print(f"    {row['job_id']}  {row['lawn_area_sqft']:>6,} sqft"
              f"  => {row['duration_minutes']:>3} min"
              f"  | material: {row['material_required']:>3}")
    print("======================\n")
    return summary
