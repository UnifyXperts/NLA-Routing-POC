"""
LLM executive summary for route optimization results.

Supported providers (set via cfg['llm']['provider']):
  groq    — Groq Cloud, model llama-3.1-8b-instant (free tier, needs GROQ_API_KEY)
  gemini  — Google Gemini 1.5 Flash (free tier, needs GOOGLE_API_KEY)
  ollama  — local Ollama server (http://localhost:11434, model llama3)
  none    — skip LLM, return empty string

The provider can be swapped by changing LLM_PROVIDER in the cfg or the
environment variable LLM_PROVIDER.
"""

import os
import pandas as pd

_FALLBACK_NOTE = "[LLM unavailable — showing static fallback]"


def build_prompt(routes: dict, unassigned: list, jobs: pd.DataFrame,
                 technicians: pd.DataFrame, dist_data: dict, cfg: dict,
                 priority_queue: pd.DataFrame = None,
                 eligibility_df: pd.DataFrame = None) -> str:
    """Construct a WHY-focused executive prompt explaining optimizer decisions."""
    ow       = cfg.get("objective_weights", {})
    pw       = cfg.get("priority_weights", {})
    ucfg     = cfg.get("utilization", {})
    cflags   = cfg.get("constraint_flags", {})
    route_dt = cfg.get("operational_constraints", {}).get("route_date", "today")

    job_rev_map  = dict(zip(jobs["job_id"], jobs["revenue"]))
    job_dur_map  = (dict(zip(jobs["job_id"], jobs["duration_minutes"]))
                    if "duration_minutes" in jobs.columns else {})
    job_cust_map = dict(zip(jobs["job_id"], jobs["customer_name"]))
    job_prog_map = dict(zip(jobs["job_id"], jobs["program_id"]))
    job_pref_map = dict(zip(jobs["job_id"],
                            jobs["preferred_tech_id"].fillna("").astype(str).str.strip()))
    tech_name_map = dict(zip(technicians["tech_id"], technicians["name"]))

    total_jobs = len(jobs)
    assigned   = sum(
        len([s for s in info["route"] if s not in ("Depot", "BREAK")])
        for info in routes.values()
    )
    total_rev = sum(
        job_rev_map.get(s, 0)
        for info in routes.values()
        for s in info["route"] if s not in ("Depot", "BREAK")
    )

    # ── Section 1: Coverage ───────────────────────────────────────────────────
    lines = [
        "You are a field operations analyst. A service route optimizer just ran.",
        "Explain in 4–6 sentences WHY these routing decisions were made — not just what happened.",
        "Cover: which jobs were prioritized and why, why specific techs were chosen,",
        "what trade-offs occurred (revenue vs. drive time, capacity limits), and",
        "one concrete action the manager can take to improve coverage if jobs are unassigned.",
        "Be specific and use job/tech IDs. Do not use bullet points.",
        "",
        f"ROUTE DATE: {route_dt}",
        f"COVERAGE: {assigned} of {total_jobs} jobs scheduled  ({len(unassigned)} unassigned)",
        f"TOTAL REVENUE CAPTURED: ${total_rev:,.2f}",
        "",
    ]

    # ── Section 2: Objective weight trade-offs ────────────────────────────────
    lines.append("OBJECTIVE WEIGHTS (higher = stronger influence):")
    for k, v in ow.items():
        lines.append(f"  {k.replace('_',' ')}: {float(v):.1f}")
    dominant = max(ow, key=lambda k: float(ow[k]))
    lines.append(
        f"  → Dominant objective: '{dominant.replace('_',' ')}'. "
        "The optimizer favoured decisions that most improved this objective."
    )

    # ── Section 3: Priority queue rationale ──────────────────────────────────
    lines.append("")
    lines.append("PRIORITY QUEUE — WHY THESE JOBS CAME FIRST:")
    lines.append(f"  Weights: new_customer={pw.get('new_customer',1000)}, "
                 f"asap={pw.get('asap',100)}, days_overdue×{pw.get('last_service_days',1)}")
    if priority_queue is not None and not priority_queue.empty:
        top5 = priority_queue.head(5)
        for _, row in top5.iterrows():
            reasons = []
            if row.get("new_customer"):
                reasons.append(f"new customer (w={pw.get('new_customer',1000)})")
            if row.get("asap"):
                reasons.append(f"ASAP flag (w={pw.get('asap',100)})")
            if row.get("job_type") == "service_call":
                reasons.append("service call (ahead of regular)")
            days = int(row.get("last_service_days", 0))
            if days > 0:
                reasons.append(f"{days}d since last visit")
            lines.append(
                f"  #{int(row['priority_rank'])}: {row['job_id']} "
                f"({job_cust_map.get(row['job_id'], '')}) — {', '.join(reasons) or 'standard order'}"
            )

    # ── Section 4: Per-technician assignments and WHY ─────────────────────────
    lines.append("")
    lines.append("TECHNICIAN ASSIGNMENTS — WHY EACH TECH GOT THEIR STOPS:")
    for tid, info in routes.items():
        stops     = [s for s in info["route"] if s not in ("Depot", "BREAK")]
        shift_min = info.get("shift_minutes", 480)
        eff_min   = info.get("eff_shift_minutes", shift_min)
        used_min  = info["total_minutes"]
        util_pct  = round(used_min / shift_min * 100, 1) if shift_min else 0
        svc_min   = sum(job_dur_map.get(s, 0) for s in stops)
        drive_min = round(used_min - svc_min, 0)
        pref_vio  = [s for s in stops if job_pref_map.get(s, "") not in ("", tid)]
        name      = tech_name_map.get(tid, tid)
        lines.append(
            f"  {tid} ({name}): {len(stops)} stops | "
            f"{used_min:.0f}/{shift_min:.0f} min ({util_pct}% util) | "
            f"{drive_min:.0f} min drive | "
            f"OR-Tools budget: {eff_min:.0f} min | "
            f"${sum(job_rev_map.get(s,0) for s in stops):,.0f} revenue"
        )
        if pref_vio:
            lines.append(
                f"    ⚠ Preference violations: {', '.join(pref_vio)} "
                f"(customer requested different tech but {tid} was assigned — capacity/eligibility trade-off)"
            )
        for s in stops:
            lines.append(
                f"    → {s} ({job_cust_map.get(s,'')}) | {job_prog_map.get(s,'')} | "
                f"{job_dur_map.get(s,0)} min service | ${job_rev_map.get(s,0):.0f}"
            )

    # ── Section 5: Unassigned jobs WHY ───────────────────────────────────────
    if unassigned:
        lines.append("")
        lines.append("UNASSIGNED JOBS — WHY THE OPTIMIZER EXCLUDED THEM:")
        elig_map = {}
        reason_map = {}
        if eligibility_df is not None and not eligibility_df.empty:
            for _, row in eligibility_df.iterrows():
                elig_map[row["job_id"]]   = row.get("eligible_technicians", [])
                reason_map[row["job_id"]] = row.get("rejection_reason", "")
        for jid in unassigned[:8]:
            rev  = job_rev_map.get(jid, 0)
            elig = elig_map.get(jid, [])
            p2r  = reason_map.get(jid, "")
            if p2r:
                why = f"Phase II eligibility failure: {p2r}"
            elif not elig:
                why = "No technician passed eligibility checks (skill/license/equipment/availability)"
            else:
                why = (
                    f"Optimizer trade-off: all {len(elig)} eligible tech(s) ({', '.join(elig)}) "
                    "were at/near capacity or adding this stop would increase total cost"
                )
            lines.append(
                f"  {jid} ({job_cust_map.get(jid,'')}) | "
                f"${rev:.0f} | {job_prog_map.get(jid,'')} — {why}"
            )
        if len(unassigned) > 8:
            lines.append(f"  ... and {len(unassigned)-8} more unassigned jobs")

    # ── Section 6: Constraint context ────────────────────────────────────────
    disabled = [k.replace("enable_","").replace("_"," ")
                for k, v in cflags.items() if not v]
    if disabled:
        lines.append("")
        lines.append(f"DISABLED CONSTRAINTS: {', '.join(disabled)}")
        lines.append(
            "  These checks were turned OFF — eligibility was broader than default."
        )

    lines.append("")
    lines.append(f"UTILIZATION TARGET: {ucfg.get('target_pct',90)}% "
                 f"(band {ucfg.get('band_low_pct',85)}%–{ucfg.get('band_high_pct',95)}%)")
    lines.append("Lunch break is subtracted from OR-Tools scheduling budget when enabled.")
    lines.append("")
    lines.append("Write the 4–6 sentence executive WHY summary now:")
    return "\n".join(lines)


def call_llm(prompt: str, cfg: dict, api_key: str = None) -> str:
    """
    Call the configured LLM provider. Returns narrative string.
    Raises on failure so the caller can fall back gracefully.
    """
    llm_cfg  = cfg.get("llm", {})
    provider = llm_cfg.get("provider", "none").lower()
    key      = api_key or llm_cfg.get("api_key", "") or ""

    if provider == "groq":
        return _call_groq(prompt, key)
    elif provider == "gemini":
        return _call_gemini(prompt, key)
    elif provider == "ollama":
        return _call_ollama(prompt, llm_cfg.get("model", "llama3"))
    else:
        raise ValueError(f"LLM provider '{provider}' not supported or disabled.")


def generate_narrative(routes: dict, unassigned: list, jobs: pd.DataFrame,
                       technicians: pd.DataFrame, dist_data: dict, cfg: dict,
                       api_key: str = None,
                       priority_queue: "pd.DataFrame | None" = None,
                       eligibility_df: "pd.DataFrame | None" = None) -> str:
    """
    High-level entry point. Returns LLM narrative or a static fallback.
    Never raises — always returns a string.
    """
    llm_cfg = cfg.get("llm", {})
    if not llm_cfg.get("enabled", True):
        return ""
    if llm_cfg.get("provider", "none").lower() == "none":
        return ""

    try:
        prompt = build_prompt(
            routes, unassigned, jobs, technicians, dist_data, cfg,
            priority_queue=priority_queue,
            eligibility_df=eligibility_df,
        )
        return call_llm(prompt, cfg, api_key)
    except Exception as exc:
        return (
            _static_fallback(routes, unassigned, jobs, cfg)
            + f"\n\n{_FALLBACK_NOTE}\n(Error: {exc})"
        )


# ── Provider implementations ──────────────────────────────────────────────────

def _call_groq(prompt: str, api_key: str) -> str:
    try:
        from groq import Groq  # type: ignore
    except ImportError:
        raise ImportError("groq package not installed. Run: pip install groq")
    key = api_key or os.environ.get("GROQ_API_KEY", "")
    if not key:
        raise ValueError("No Groq API key. Set GROQ_API_KEY or enter it in the sidebar.")
    client = Groq(api_key=key)
    resp   = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()


def _call_gemini(prompt: str, api_key: str) -> str:
    try:
        import google.generativeai as genai  # type: ignore
    except ImportError:
        raise ImportError("google-generativeai not installed. Run: pip install google-generativeai")
    key = api_key or os.environ.get("GOOGLE_API_KEY", "")
    if not key:
        raise ValueError("No Gemini API key. Set GOOGLE_API_KEY or enter it in the sidebar.")
    genai.configure(api_key=key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    resp  = model.generate_content(prompt)
    return resp.text.strip()


def _call_ollama(prompt: str, model: str = "llama3") -> str:
    try:
        import requests  # type: ignore
    except ImportError:
        raise ImportError("requests not installed.")
    resp = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=90,
    )
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


# ── Static fallback ───────────────────────────────────────────────────────────

def _static_fallback(routes: dict, unassigned: list, jobs: pd.DataFrame, cfg: dict) -> str:
    """Minimal narrative derived directly from KPI numbers — no LLM needed."""
    job_rev_map = dict(zip(jobs["job_id"], jobs["revenue"]))
    total_jobs  = len(jobs)
    assigned    = sum(
        len([s for s in info["route"] if s not in ("Depot", "BREAK")])
        for info in routes.values()
    )
    total_rev = sum(
        job_rev_map.get(s, 0)
        for info in routes.values()
        for s in info["route"] if s not in ("Depot", "BREAK")
    )
    n_unassigned = len(unassigned)
    n_techs      = len(routes)

    util_pcts = []
    for tid, info in routes.items():
        shift_min = info.get("shift_minutes", 480)
        if shift_min:
            util_pcts.append(info["total_minutes"] / shift_min * 100)
    avg_util = round(sum(util_pcts) / len(util_pcts), 1) if util_pcts else 0

    route_dt = cfg.get("operational_constraints", {}).get("route_date", "today")
    lines = [
        f"Today's route ({route_dt}) covers {assigned} of {total_jobs} jobs across "
        f"{n_techs} technician{'s' if n_techs != 1 else ''}, capturing ${total_rev:,.2f} in revenue.",
        f"Average fleet utilisation is {avg_util}%.",
    ]
    if n_unassigned:
        lines.append(
            f"{n_unassigned} job{'s' if n_unassigned != 1 else ''} "
            f"({', '.join(unassigned[:3])}{'...' if n_unassigned > 3 else ''}) "
            f"could not be scheduled — review eligibility filters or technician capacity."
        )
    else:
        lines.append("All jobs were successfully assigned.")
    return " ".join(lines)
