"""
Groq-powered multi-turn chatbot for the TouchTurf Routing Engine.

Injects the full optimizer run context (routes, KPIs, priority queue,
eligibility, config, and embedded code logic) into the system prompt so
the model can answer precise trade-off and output questions.
"""

from __future__ import annotations

import os
from typing import Generator

import pandas as pd


# ── Embedded routing-engine knowledge ─────────────────────────────────────────

_SYSTEM_TEMPLATE = """\
You are the TouchTurf Routing Engine Assistant for NaturaLawn of America, Richmond VA.

You have full access to the current optimizer run context (routes, KPIs, priority queue, \
eligibility checks, config). Answer every question using that data. Be concise and specific — \
cite job IDs, tech IDs, dollar amounts, minutes, and percentages. Never make up numbers.

━━━ ROUTING ENGINE CODE LOGIC ━━━

PHASE I — Priority Scoring (phase1_priority.py):
  score = (new_customer_flag × W_new) + (asap_flag × W_asap) + (days_since_last_visit × W_days)
  Service calls (job_type="service_call") are grouped before regular jobs within each tier.
  Tie-break within a tier: higher days_since_last_visit wins.

PHASE II — Eligibility Matching (phase2_matching.py) — ALL 6 checks must pass:
  1. Skill        — tech.skills ⊇ program.required_skill
  2. License      — tech.licenses ⊇ program.required_license
  3. Availability — route_date ∈ tech.available_days  AND  preferred_time_window ⊆ [shift_start, shift_end]
  4. Location     — OSRM drive time from tech depot → job cluster ≤ location_threshold_minutes (default 60)
  5. Equipment    — truck.equipment ⊇ program.required_equipment
  6. Capacity     — truck.capacity ≥ job.material_required
  Any check that is disabled in constraint_flags is automatically treated as passed.

PHASE III — OR-Tools VRP (optimizer.py):
  minimize total_cost = (−W_revenue × revenue_scale × revenue)
                      + (W_drive    × drive_time_seconds)
                      + (W_pref    × preference_penalty)
                      − (W_util    × utilization_term)

  utilization_term  = max(0, target_seconds − actual_route_seconds)
  actual_route_time = Σ drive_time + Σ job_duration_minutes×60
  preference_penalty fires when: assigned_tech ≠ preferred_tech_id  OR  arrival > preferred_time_window

  Batch processing: jobs are processed in priority-ordered batches (default 45 jobs/batch).
  Each batch gets its own OSRM matrix call and OR-Tools VRP solve.
  Geo-clustering (k-means) pre-assigns jobs to technician zones before the VRP solve.

KEY TRADE-OFFS — how weight changes shift the solution:
  W_revenue ↑  → high-$ jobs fill routes first; low-revenue jobs dropped when techs near capacity
  W_drive ↑    → geographically compact routes; distant jobs skipped even if high revenue
  W_util ↑     → optimizer packs routes toward target %; may pull lower-revenue jobs to fill idle time
  W_pref ↑     → avoids wrong-tech assignments; may accept longer drive times to honour preferences
  Geo clustering ON → faster solve, but cross-zone assignments are prevented even when optimal
  Batch size ↓  → faster but loses global view across batches
  Disabling constraints → more jobs assigned but may violate real-world feasibility

DISTANCE NOTES:
  All distances use OSRM real road network (not haversine/straight-line).
  distance_matrix = meters | duration_matrix = seconds
  Utilisation % = total_route_time / shift_minutes × 100
  Lunch break subtracts from OR-Tools scheduling budget but NOT from the utilisation denominator.

━━━ CURRENT OPTIMIZER RUN CONTEXT ━━━

{context}
"""


# ── Context builder ───────────────────────────────────────────────────────────

def build_context(results: dict | None) -> str:
    """Serialise optimizer session-state results into structured text for the LLM."""
    if not results:
        return (
            "No optimizer run has been completed in this session yet. "
            "Answer general questions about how the routing engine works."
        )

    lines: list[str] = []

    cfg        = results.get("cfg", {})
    routes     = results.get("routes", {}) or {}
    jobs_df    = results.get("data", {}).get("jobs", pd.DataFrame())
    techs_df   = results.get("data", {}).get("technicians", pd.DataFrame())
    progs_df   = results.get("data", {}).get("programs", pd.DataFrame())
    elig_df    = results.get("eligibility_df", pd.DataFrame())
    pq         = results.get("priority_queue", pd.DataFrame())
    unassigned = results.get("unassigned", []) or []
    kpi_df     = results.get("kpi_df")

    oc    = cfg.get("operational_constraints", {})
    ow    = cfg.get("objective_weights", {})
    pw    = cfg.get("priority_weights", {})
    ucfg  = cfg.get("utilization", {})
    cf    = cfg.get("constraint_flags", {})

    # ── Config ────────────────────────────────────────────────────────────────
    lines += [
        "── CONFIG ──────────────────────────────────────────────",
        f"Route date        : {oc.get('route_date','N/A')}",
        f"Filter by date    : {oc.get('filter_by_date', True)}",
        f"W_revenue={ow.get('maximize_revenue',1)}  W_drive={ow.get('minimize_drive_time',1)}  "
        f"W_preference={ow.get('meet_customer_preference',1)}  W_utilization={ow.get('maximize_tech_utilization',1)}",
        f"Priority weights  : new_customer={pw.get('new_customer',1000)}  "
        f"asap={pw.get('asap',100)}  days×{pw.get('last_service_days',1)}",
        f"Utilization target: {ucfg.get('target_pct',90)}%  "
        f"band={ucfg.get('band_low_pct',85)}%–{ucfg.get('band_high_pct',95)}%  "
        f"calendar={'ON' if ucfg.get('use_shift_calendar') else 'OFF'}",
        f"Max hours/tech    : {oc.get('max_hours_per_tech',8)}h  "
        f"Lunch: {oc.get('lunch_break_minutes',0)} min after {oc.get('lunch_after_minutes',240)} min",
        f"Active constraints: {', '.join(k.replace('enable_','') for k,v in cf.items() if v)}",
    ]
    disabled_c = [k.replace("enable_","") for k,v in cf.items() if not v]
    if disabled_c:
        lines.append(f"DISABLED constraints: {', '.join(disabled_c)}")

    # ── KPI ───────────────────────────────────────────────────────────────────
    if kpi_df is not None and not kpi_df.empty:
        lines.append("\n── KPI SUMMARY ─────────────────────────────────────────")
        for _, row in kpi_df.iterrows():
            lines.append(f"  {row['Metric']}: {row['Value']}")

    # ── Per-tech routes ───────────────────────────────────────────────────────
    if routes:
        job_rev  = {} if jobs_df.empty else dict(zip(jobs_df["job_id"], jobs_df["revenue"]))
        job_cust = {} if jobs_df.empty else dict(zip(jobs_df["job_id"], jobs_df["customer_name"]))
        job_prog = {} if jobs_df.empty else dict(zip(jobs_df["job_id"], jobs_df["program_id"]))
        job_dur  = ({} if jobs_df.empty or "duration_minutes" not in jobs_df.columns
                    else dict(zip(jobs_df["job_id"], jobs_df["duration_minutes"])))
        tech_name = ({} if techs_df.empty
                     else dict(zip(techs_df["tech_id"], techs_df["name"])))

        lines.append("\n── TECHNICIAN ROUTES ───────────────────────────────────")
        for tid, info in routes.items():
            stops     = [s for s in info["route"] if s not in ("Depot", "BREAK")]
            shift_min = info.get("shift_minutes", 480)
            eff_min   = info.get("eff_shift_minutes", shift_min)
            used_min  = info["total_minutes"]
            util_pct  = round(used_min / shift_min * 100, 1) if shift_min else 0
            svc_min   = sum(job_dur.get(s, 0) for s in stops)
            drive_min = round(used_min - svc_min, 1)
            rev       = sum(job_rev.get(s, 0) for s in stops)
            name      = tech_name.get(tid, tid)
            lines.append(
                f"\n{tid} ({name})  {len(stops)} stops  "
                f"{used_min:.0f}/{shift_min:.0f} min ({util_pct}% util)  ${rev:,.0f} revenue"
            )
            lines.append(f"  Drive: {drive_min:.0f} min | Service: {svc_min:.0f} min | OR-Tools budget: {eff_min:.0f} min")
            lines.append(f"  Sequence: {' → '.join(info['route'])}")
            for s in stops:
                lines.append(
                    f"  • {s}  {job_cust.get(s,'')}  {job_prog.get(s,'')}  "
                    f"{job_dur.get(s,0)} min  ${job_rev.get(s,0):.0f}"
                )

    # ── Unassigned ────────────────────────────────────────────────────────────
    if unassigned:
        lines.append(f"\n── UNASSIGNED JOBS ({len(unassigned)}) ─────────────────────────")
        job_rev  = {} if jobs_df.empty else dict(zip(jobs_df["job_id"], jobs_df["revenue"]))
        job_cust = {} if jobs_df.empty else dict(zip(jobs_df["job_id"], jobs_df["customer_name"]))
        job_prog = {} if jobs_df.empty else dict(zip(jobs_df["job_id"], jobs_df["program_id"]))
        elig_map   = {}
        reason_map = {}
        if elig_df is not None and not elig_df.empty:
            for _, row in elig_df.iterrows():
                elig_map[row["job_id"]]   = row.get("eligible_technicians", [])
                reason_map[row["job_id"]] = row.get("rejection_reason", "")
        for jid in unassigned:
            elig = elig_map.get(jid, [])
            p2r  = reason_map.get(jid, "")
            if p2r:
                why = f"Phase II blocked: {p2r}"
            elif not elig:
                why = "No eligible tech (all 6 eligibility checks failed)"
            else:
                why = (f"Optimizer trade-off: {len(elig)} eligible tech(s) "
                       f"({', '.join(elig)}) but all at/near capacity or adding this "
                       "stop would increase total_cost")
            lines.append(
                f"  {jid}  {job_cust.get(jid,'')}  "
                f"${job_rev.get(jid,0):.0f}  {job_prog.get(jid,'')} → {why}"
            )

    # ── Priority queue ────────────────────────────────────────────────────────
    if pq is not None and not pq.empty:
        lines.append(f"\n── PRIORITY QUEUE (top 20 of {len(pq)}) ────────────────────")
        cols = ["priority_rank","job_id","customer_name","job_type",
                "new_customer","asap","last_service_days","priority_score"]
        cols = [c for c in cols if c in pq.columns]
        for _, row in pq.head(20).iterrows():
            lines.append("  " + "  ".join(f"{c}={row[c]}" for c in cols))

    # ── Eligibility detail ────────────────────────────────────────────────────
    if elig_df is not None and not elig_df.empty:
        lines.append(f"\n── ELIGIBILITY (first 20 jobs) ─────────────────────────")
        for _, row in elig_df.head(20).iterrows():
            elig   = row.get("eligible_technicians", [])
            reason = row.get("rejection_reason", "")
            lines.append(
                f"  {row['job_id']}: eligible={elig or 'NONE'}"
                + (f"  blocked: {reason}" if reason else "")
            )

    # ── Programs reference ────────────────────────────────────────────────────
    if not progs_df.empty:
        lines.append("\n── PROGRAMS REFERENCE ──────────────────────────────────")
        for _, pr in progs_df.iterrows():
            lines.append(
                f"  {pr['program_id']}: skill={pr.get('required_skill','')}  "
                f"license={pr.get('required_license','')}  "
                f"equipment={pr.get('required_equipment','')}  "
                f"material/1k sqft={pr.get('material_per_1000sqft', pr.get('material_required',''))}"
            )

    return "\n".join(lines)


# ── Groq streaming ────────────────────────────────────────────────────────────

_DEFAULT_MODEL = "llama-3.3-70b-versatile"
_FALLBACK_MODELS = ["llama-3.1-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"]


def stream_response(
    messages: list[dict],
    context: str,
    api_key: str,
    model: str = _DEFAULT_MODEL,
) -> Generator[str, None, None]:
    """
    Stream a Groq chat completion.

    messages  — list of {"role": "user"/"assistant", "content": str}
    context   — built by build_context()
    api_key   — Groq API key
    Yields text chunks as they arrive.
    """
    try:
        from groq import Groq  # type: ignore
    except ImportError:
        yield "⚠️ `groq` package not installed. Run: `pip install groq`"
        return

    key = api_key or os.environ.get("GROQ_API_KEY", "")
    if not key:
        yield (
            "⚠️ No Groq API key. "
            "Add `GROQ_API_KEY=your_key` to `routing_poc/.env` "
            "or enter it in the sidebar."
        )
        return

    system_prompt = _SYSTEM_TEMPLATE.format(context=context)
    payload = [{"role": "system", "content": system_prompt}] + messages
    client  = Groq(api_key=key)

    try:
        stream = client.chat.completions.create(
            model=model,
            messages=payload,
            stream=True,
            max_tokens=1200,
            temperature=0.25,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    except Exception as exc:
        err = str(exc)
        # Try fallback models once on model-not-found errors
        if "model" in err.lower() or "not found" in err.lower():
            for fb in _FALLBACK_MODELS:
                if fb == model:
                    continue
                try:
                    stream = client.chat.completions.create(
                        model=fb,
                        messages=payload,
                        stream=True,
                        max_tokens=1200,
                        temperature=0.25,
                    )
                    yield f"*(using {fb})*\n\n"
                    for chunk in stream:
                        delta = chunk.choices[0].delta.content
                        if delta:
                            yield delta
                    return
                except Exception:
                    continue
        yield f"\n\n⚠️ Groq error: {exc}"
