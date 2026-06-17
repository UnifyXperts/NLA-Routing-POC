import pandas as pd


def compute_priority(jobs: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Sort order:
      1. new_customer desc          (Rule 2 — highest priority)
      2. asap desc                  (Rule 3 — second highest)
      3. job_type: service_call < regular  (Rule 4 — service calls first)
      4. last_service_days desc     (Rule 5 — most overdue first, within each group)
    """
    w = cfg["priority_weights"]
    df = jobs.copy()

    # Numeric type rank: 0 = service_call (higher priority), 1 = regular
    df["_type_rank"] = df["job_type"].apply(lambda x: 0 if str(x).strip() == "service_call" else 1)

    # Priority score drives the primary sort; type_rank and days refine within ties
    df["priority_score"] = (
        df["new_customer"]      * w["new_customer"] +
        df["asap"]              * w["asap"] +
        df["last_service_days"] * w["last_service_days"]
    )

    df = df.sort_values(
        by=["priority_score", "_type_rank", "last_service_days"],
        ascending=[False, True, False],
    ).reset_index(drop=True)

    df = df.drop(columns=["_type_rank"])
    df["priority_rank"] = range(1, len(df) + 1)
    return df


def print_queue(df: pd.DataFrame) -> None:
    print("\n=== Phase I: Prioritized Job Queue ===")
    cols = ["priority_rank", "job_id", "job_type", "program_id",
            "new_customer", "asap", "last_service_days", "priority_score"]
    print(df[cols].to_string(index=False))
    print("======================================\n")
