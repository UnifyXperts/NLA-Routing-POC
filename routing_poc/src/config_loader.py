import json
import os

REQUIRED = {
    "priority_weights":        ["new_customer", "asap", "last_service_days"],
    "objective_weights":       ["maximize_revenue", "minimize_drive_time",
                                "meet_customer_preference", "maximize_tech_utilization"],
    "operational_constraints": ["max_hours_per_tech", "lunch_after_minutes",
                                "lunch_break_minutes", "location_threshold_minutes", "route_date"],
}

# Optional sections — present in routing_rules.json but not required by CLI runner
_OPTIONAL_SECTIONS = ("utilization", "constraint_flags", "llm")

_CONSTRAINT_FLAG_DEFAULTS = {
    "enable_work_hours":         True,
    "enable_capacity":           True,
    "enable_location_threshold": True,
    "enable_lunch_break":        True,
    "enable_skill_check":        True,
    "enable_license_check":      True,
    "enable_equipment_check":    True,
    "enable_availability_check": True,
}

_UTILIZATION_DEFAULTS = {
    "target_pct":                   90,
    "band_low_pct":                 85,
    "band_high_pct":                95,
    "underutil_penalty_per_minute": 10,
    "use_shift_calendar":           True,
    "minimize_variance":            True,
}


def load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Routing config not found: {config_path}")
    with open(config_path) as f:
        cfg = json.load(f)
    _validate(cfg, config_path)
    _apply_defaults(cfg)
    return cfg


def _validate(cfg: dict, path: str) -> None:
    errors = []
    for section, keys in REQUIRED.items():
        if section not in cfg:
            errors.append(f"Missing section '{section}'")
            continue
        for key in keys:
            if key not in cfg[section]:
                errors.append(f"Missing key '{section}.{key}'")
    if errors:
        raise ValueError(
            f"Invalid config at {path}:\n" + "\n".join(f"  - {e}" for e in errors)
        )


def _apply_defaults(cfg: dict) -> None:
    """Merge optional sections with defaults so downstream code can rely on them."""
    cfg.setdefault("constraint_flags", {})
    for k, v in _CONSTRAINT_FLAG_DEFAULTS.items():
        cfg["constraint_flags"].setdefault(k, v)

    cfg.setdefault("utilization", {})
    for k, v in _UTILIZATION_DEFAULTS.items():
        cfg["utilization"].setdefault(k, v)

    cfg.setdefault("llm", {"provider": "none", "enabled": False})


def print_config(cfg: dict) -> None:
    col = 34
    print("\n  Priority Weights:")
    for k, v in cfg["priority_weights"].items():
        print(f"    {k:<{col}} = {v}")
    print("\n  Objective Weights:")
    for k, v in cfg["objective_weights"].items():
        print(f"    {k:<{col}} = {v}")
    print("\n  Operational Constraints:")
    for k, v in cfg["operational_constraints"].items():
        print(f"    {k:<{col}} = {v}")
    print("\n  Utilization Settings:")
    for k, v in cfg.get("utilization", {}).items():
        print(f"    {k:<{col}} = {v}")
    print("\n  Constraint Flags:")
    for k, v in cfg.get("constraint_flags", {}).items():
        state = "ON " if v else "OFF"
        print(f"    [{state}]  {k}")
    print()
