import streamlit as st
import sys
import os
import shutil
import tempfile
import pandas as pd
from datetime import date
from dotenv import load_dotenv

# ── Authentication gate ───────────────────────────────────────────────────────
_AUTH_USERNAME = "Administrator"
_AUTH_PASSWORD = "%fP985t3,2jS"

def _login_screen():
    st.set_page_config(page_title="TouchTurf — Login", layout="centered")
    st.title("TouchTurf Routing Engine")
    st.subheader("Sign in to continue")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign In")
    if submitted:
        if username == _AUTH_USERNAME and password == _AUTH_PASSWORD:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Invalid username or password.")

if not st.session_state.get("authenticated"):
    _login_screen()
    st.stop()
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))
SRC_DIR  = os.path.join(BASE_DIR, "src")
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR  = os.path.join(BASE_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)
sys.path.insert(0, SRC_DIR)

from data_loader          import load_all
from phase1_priority      import compute_priority
from phase2_matching      import match_resources
from geo_clustering       import cluster_jobs_to_techs
from distance_matrix      import build_distance_matrix
from batch_runner         import optimize_batched
from visualization        import build_route_map, build_jobs_map
from dashboard            import build_kpi, build_optimization_summary
from llm_summary          import generate_narrative
from service_history_map  import load_service_data, available_dates, build_sh_map
from chatbot              import build_context, stream_response

# ── Pre-load service history files (cached so they don't reload on every rerun) ─
_SH_PATH     = os.path.join(DATA_DIR, "service_history.xlsx")
_LAT_PATH    = os.path.join(DATA_DIR, "customer_lat_long.xlsx")
_BRANCH_PATH = os.path.join(DATA_DIR, "branch.csv")

@st.cache_data(show_spinner=False)
def _load_sh_data():
    if os.path.exists(_SH_PATH) and os.path.exists(_LAT_PATH) and os.path.exists(_BRANCH_PATH):
        return load_service_data(_SH_PATH, _LAT_PATH, _BRANCH_PATH)
    return None, None, None

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TouchTurf Routing Engine",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("TouchTurf Routing Engine")
st.caption("POC · Richmond VA · OR-Tools + OSRM")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.caption(f"Signed in as **{_AUTH_USERNAME}**")
    if st.button("Sign Out"):
        st.session_state["authenticated"] = False
        st.rerun()
    st.divider()
    st.header("Configuration")

    # ── Data Files ────────────────────────────────────────────────────────────
    st.subheader("Data Files")
    st.caption("Leave blank to use built-in sample data")
    up_jobs   = st.file_uploader("jobs.csv",        type="csv", key="up_jobs")
    up_techs  = st.file_uploader("technicians.csv", type="csv", key="up_techs")
    up_trucks = st.file_uploader("trucks.csv",      type="csv", key="up_trucks")
    up_progs  = st.file_uploader("programs.csv",    type="csv", key="up_progs")

    st.divider()

    # ── Schedule ──────────────────────────────────────────────────────────────
    st.subheader("Schedule")
    filter_by_date = st.toggle("Filter by scheduled date", value=True,
                               help="OFF = run all jobs regardless of scheduled_date column.")
    route_date = st.date_input("Route Date", value=date(2026, 6, 12),
                               disabled=not filter_by_date)

    st.divider()

    # ── Objective Weights ─────────────────────────────────────────────────────
    st.subheader("Objective Weights")
    st.caption("Set to 0 to disable an objective entirely.")
    w_revenue = st.slider("Maximize Revenue",    0.0, 5.0, 2.0, 0.5)
    w_drive   = st.slider("Minimize Drive Time", 0.0, 5.0, 1.0, 0.5)
    w_pref    = st.slider("Customer Preference", 0.0, 5.0, 1.0, 0.5)
    w_util    = st.slider("Tech Utilization",    0.0, 5.0, 1.0, 0.5)

    st.caption("**Service Call Priority**")
    service_call_penalty = st.number_input(
        "Service call minimum value ($-equivalent)",
        min_value=0, max_value=10000, value=500, step=50,
        help=(
            "Service calls typically have $0 revenue, so the optimizer would otherwise "
            "drop them freely. This sets a minimum dollar value the optimizer treats "
            "every service call as worth — e.g. $500 means the solver will accept a "
            "detour up to the same cost it would pay to include a $500 regular job. "
            "Raise to force service calls into routes even if far out of the way."
        ),
    )

    st.divider()

    # ── Utilization Settings ──────────────────────────────────────────────────
    st.subheader("Utilization Settings")
    use_calendar = st.toggle(
        "Use shift calendar (shift_start/shift_end)",
        value=True,
        help="ON = derive available hours from shift times; OFF = use flat max_hours field.",
    )
    target_util_pct = st.slider("Target utilization (%)", 50, 100, 90, 5)
    col_bl, col_bh  = st.columns(2)
    with col_bl:
        util_band_low  = st.number_input("Band low (%)",  min_value=5, max_value=100, value=15, step=5,
                                         help="Set ≈ (avg jobs/tech × avg job time) / shift length. 15% suits ~3 jobs/tech; raise to 85% when techs are fully booked.")
    with col_bh:
        util_band_high = st.number_input("Band high (%)", min_value=50, max_value=100, value=95, step=5)
    underutil_pen    = st.number_input(
        "Under-utilization penalty / min", min_value=0, max_value=1000, value=100, step=10,
        help="OR-Tools cost units added per minute a tech's route falls below the low band.",
    )
    minimize_variance = st.toggle(
        "Minimize utilization variance across technicians",
        value=True,
        help="Penalises the gap between the longest and shortest route (global span cost).",
    )
    variance_span_coeff = st.number_input(
        "Variance span coefficient",
        min_value=0, max_value=10, value=0, step=1,
        help=(
            "Cost units per second of spread between longest and shortest route. "
            "0 = disabled (safe default). "
            "Raise to 1 only when techs have many jobs and you want even load-spreading. "
            "Values above 1 may cause job drops on sparse schedules."
        ),
    )

    st.divider()

    # ── Priority Weights ──────────────────────────────────────────────────────
    st.subheader("Priority Weights")
    pw_new  = st.number_input("New Customer",      value=1000, step=100, min_value=0)
    pw_asap = st.number_input("ASAP",              value=100,  step=10,  min_value=0)
    pw_days = st.number_input("Last Service Days", value=1,    step=1,   min_value=0)

    st.divider()

    # ── Phase II Constraint Flags ─────────────────────────────────────────────
    st.subheader("Phase II Constraints")
    st.caption("Toggle checks applied during technician eligibility matching.")
    with st.expander("Eligibility checks (all ON by default)", expanded=False):
        en_location    = st.toggle("Location threshold",  value=True,
                                   help="Exclude depots >N min drive from job cluster.")
        if en_location:
            loc_thresh = st.number_input("Max drive time depot→job (min)",
                                         value=60, step=5, min_value=5)
        else:
            loc_thresh = 9999  # effectively disabled
        en_skill       = st.toggle("Skill check",       value=True)
        en_license     = st.toggle("License check",     value=True)
        en_availability = st.toggle("Availability check", value=True)
        en_equipment   = st.toggle("Equipment check",   value=True)
        en_capacity    = st.toggle("Truck capacity check", value=True)

    # ── Optimizer Constraint Flags ────────────────────────────────────────────
    st.subheader("Optimizer Constraints")
    st.caption("Toggle hard constraints in the OR-Tools VRP model.")
    with st.expander("Route constraints (all ON by default)", expanded=False):
        en_work_hours = st.toggle(
            "Work-hour limit (Time dimension hard cap)", value=True,
            help="OFF = solver can schedule beyond the tech's shift end.",
        )
        if en_work_hours:
            max_hours = st.number_input("Max hours / tech (flat fallback)",
                                        value=8, step=1, min_value=1,
                                        help="Used only when 'shift calendar' is OFF.")
        else:
            max_hours = 24  # irrelevant when disabled

        en_lunch = st.toggle("Lunch break insertion", value=True)
        if en_lunch:
            lunch_after = st.number_input("Insert lunch after (min)", value=240, step=30, min_value=60)
            lunch_dur   = st.number_input("Lunch duration (min)",     value=30,  step=5,  min_value=0)
        else:
            lunch_after = 240
            lunch_dur   = 0

    st.divider()

    # ── Geo Clustering ────────────────────────────────────────────────────────
    st.subheader("Geo Clustering")
    geo_enabled = st.toggle("k-means clustering (1 zone per tech)", value=True)

    st.divider()

    # ── Batch Settings ────────────────────────────────────────────────────────
    st.subheader("Batch Processing")
    st.caption(
        "Jobs are processed in priority-ordered batches. "
        "Keeps OSRM requests and OR-Tools problem size small regardless of job count."
    )
    batch_size = st.number_input(
        "Jobs per batch",
        min_value=10, max_value=200, value=45, step=5,
        help=(
            "Each batch = 1 OSRM request (≤ batch_size + n_techs locations) "
            "and 1 OR-Tools solve. "
            "45 is the sweet spot: fast OSRM URLs, OR-Tools solves in < 5 s per batch."
        ),
    )

    st.divider()

    # ── LLM Settings ─────────────────────────────────────────────────────────
    st.subheader("LLM Narrative")
    st.caption("Generate an AI executive summary after optimization.")
    llm_provider = st.selectbox(
        "Provider",
        options=["groq", "gemini", "ollama", "none"],
        index=0,
        help="groq / gemini read their API key from the .env file.",
    )
    if llm_provider == "groq":
        llm_api_key = os.environ.get("GROQ_API_KEY", "")
    elif llm_provider == "gemini":
        llm_api_key = os.environ.get("GOOGLE_API_KEY", "")
    else:
        llm_api_key = ""

    st.divider()

    run_btn = st.button("Run Optimizer", type="primary", use_container_width=True)


# ── Config dict (built from sidebar values) ──────────────────────────────────
cfg = {
    "priority_weights": {
        "new_customer":      int(pw_new),
        "asap":              int(pw_asap),
        "last_service_days": int(pw_days),
    },
    "objective_weights": {
        "maximize_revenue":          w_revenue,
        "minimize_drive_time":       w_drive,
        "meet_customer_preference":  w_pref,
        "maximize_tech_utilization": w_util,
        "service_call_penalty":      int(service_call_penalty),
    },
    "utilization": {
        "target_pct":                   int(target_util_pct),
        "band_low_pct":                 int(util_band_low),
        "band_high_pct":                int(util_band_high),
        "underutil_penalty_per_minute": int(underutil_pen),
        "use_shift_calendar":           use_calendar,
        "minimize_variance":            minimize_variance,
        "variance_span_coefficient":    int(variance_span_coeff),
    },
    "operational_constraints": {
        "max_hours_per_tech":         int(max_hours),
        "lunch_after_minutes":        int(lunch_after),
        "lunch_break_minutes":        int(lunch_dur),
        "location_threshold_minutes": int(loc_thresh),
        "route_date":                 str(route_date) if filter_by_date else "all",
        "filter_by_date":             filter_by_date,
        "solver_time_limit":          60,
    },
    "constraint_flags": {
        "enable_work_hours":         en_work_hours,
        "enable_capacity":           en_capacity,
        "enable_location_threshold": en_location,
        "enable_lunch_break":        en_lunch,
        "enable_skill_check":        en_skill,
        "enable_license_check":      en_license,
        "enable_equipment_check":    en_equipment,
        "enable_availability_check": en_availability,
    },
    "llm": {
        "provider": llm_provider,
        "enabled":  llm_provider != "none",
        "api_key":  llm_api_key,
    },
    "geo_clustering": {
        "enabled":      geo_enabled,
        "random_state": 42,
    },
}


# ── Data loader (handles uploads + sample fallback) ──────────────────────────
def _load_data() -> dict:
    uploads = {
        "jobs":        up_jobs,
        "technicians": up_techs,
        "trucks":      up_trucks,
        "programs":    up_progs,
    }
    tmp = tempfile.mkdtemp()
    try:
        for name in uploads:
            sample = os.path.join(DATA_DIR, f"{name}.csv")
            if os.path.exists(sample):
                shutil.copy(sample, os.path.join(tmp, f"{name}.csv"))
        for name, upload in uploads.items():
            if upload is not None:
                dest = os.path.join(tmp, f"{name}.csv")
                with open(dest, "wb") as f:
                    f.write(upload.getvalue())
        return load_all(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ── Run pipeline ──────────────────────────────────────────────────────────────
if run_btn:
    st.session_state.pop("results", None)

    try:
        with st.status("Running optimization pipeline…", expanded=True) as status:
            st.write("Loading data…")
            data    = _load_data()
            n_techs = len(data["technicians"])
            st.write(f"   {len(data['jobs'])} jobs · {n_techs} technicians loaded")

            # ── Date filter ───────────────────────────────────────────────────
            DATE_COL = "scheduled_date"
            if filter_by_date and DATE_COL in data["jobs"].columns:
                all_job_count = len(data["jobs"])
                parsed_dates  = pd.to_datetime(
                    data["jobs"][DATE_COL], errors="coerce"
                ).dt.date
                data["jobs"] = (
                    data["jobs"][parsed_dates == route_date]
                    .reset_index(drop=True)
                )
                n_filtered = len(data["jobs"])
                if n_filtered == 0:
                    st.warning(
                        f"No jobs have `scheduled_date = {route_date}`. "
                        f"({all_job_count} jobs exist in the file for other dates.) "
                        "Change the Route Date in the sidebar or toggle off date filtering."
                    )
                    status.update(label="No jobs for selected date", state="error")
                    st.stop()
                st.write(
                    f"   Date filter: **{n_filtered}** of {all_job_count} jobs "
                    f"scheduled on {route_date}"
                )
                from data_loader import _compute_job_fields
                data["jobs"] = _compute_job_fields(data["jobs"], data["programs"])
            else:
                reason = (
                    "date filter disabled — running all jobs"
                    if not filter_by_date
                    else f"no `{DATE_COL}` column in jobs.csv — running all jobs"
                )
                st.write(f"   {len(data['jobs'])} jobs loaded ({reason})")
            # ── End date filter ───────────────────────────────────────────────

            n_jobs = len(data["jobs"])
            st.write("Calling OSRM for distance matrix…")
            dist_data = build_distance_matrix(data["technicians"], data["jobs"])
            st.write(f"   {n_techs + n_jobs} × {n_techs + n_jobs} matrix ready")

            st.write("Phase I — computing priority queue…")
            priority_queue = compute_priority(data["jobs"], cfg)

            job_cluster_map = None
            if geo_enabled:
                st.write("Geo clustering (k-means)…")
                job_cluster_map = cluster_jobs_to_techs(
                    priority_queue, data["technicians"], random_state=42,
                )

            st.write("Phase II — capacity matching…")
            eligibility_df = match_resources(
                priority_queue, data["technicians"], data["trucks"],
                data["programs"], dist_data, cfg, route_date,
                job_cluster_map=job_cluster_map,
            )

            n_batches = max(1, -(-len(priority_queue) // int(batch_size)))  # ceil div
            st.write(
                f"Phase III — OR-Tools VRP optimizer "
                f"({len(priority_queue)} jobs → {n_batches} batch{'es' if n_batches > 1 else ''} of {int(batch_size)})…"
            )
            routes, unassigned = optimize_batched(
                dist_data, priority_queue, data["technicians"],
                data["trucks"], data["programs"], eligibility_df, cfg,
                batch_size=int(batch_size),
                status_cb=lambda msg: st.write(msg),
                job_cluster_map=job_cluster_map,
            )

            st.write("Building Folium route map…")
            map_path = os.path.join(OUT_DIR, "route_map.html")
            if routes:
                build_route_map(routes, data["jobs"], data["technicians"], output_path=map_path)
            else:
                build_jobs_map(data["jobs"], output_path=map_path)
            with open(map_path, "r", encoding="utf-8") as f:
                map_html = f.read()

            kpi_df  = None
            summary = None
            if routes:
                kpi_df  = build_kpi(data["jobs"], data["technicians"], routes, dist_data)
                summary = build_optimization_summary(
                    jobs=data["jobs"],
                    priority_queue=priority_queue,
                    technicians=data["technicians"],
                    routes=routes,
                    unassigned=unassigned,
                    dist_data=dist_data,
                    cfg=cfg,
                    programs=data["programs"],
                    eligibility_df=eligibility_df,
                )

            status.update(label="Optimization complete!", state="complete")

        st.session_state["results"] = {
            "data":            data,
            "priority_queue":  priority_queue,
            "eligibility_df":  eligibility_df,
            "routes":          routes,
            "unassigned":      unassigned,
            "map_html":        map_html,
            "kpi_df":          kpi_df,
            "summary":         summary,
            "job_cluster_map": job_cluster_map,
            "cfg":             cfg,
            "dist_data":       dist_data,
        }
        st.session_state.pop("llm_narrative", None)  # clear stale narrative on new run

    except Exception as exc:
        st.error(f"Pipeline failed: {exc}")
        st.exception(exc)


# ── Results ───────────────────────────────────────────────────────────────────
if "results" in st.session_state:
    r      = st.session_state["results"]
    routes = r["routes"]

    # Top-line metrics strip
    if r["kpi_df"] is not None:
        kpi = dict(zip(r["kpi_df"]["Metric"], r["kpi_df"]["Value"]))
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Revenue",        kpi.get("Total Revenue ($)", "—"))
        c2.metric("Assigned",       kpi.get("Assigned Jobs", "—"))
        c3.metric("Unassigned",     kpi.get("Unassigned Jobs", "—"))
        c4.metric("Avg Util %",     kpi.get("Avg Utilization (%)", "—"))
        c5.metric("Distance",       f"{kpi.get('Total Distance (mi)', '—')} mi")
        c6.metric("Avg Stops/Tech", kpi.get("Avg Stops / Technician", "—"))
        st.divider()

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Priority Queue",
        "Eligibility",
        "Route Map",
        "KPI Dashboard",
        "Service History",
        "Summary",
        "AI Narrative",
    ])

    # ── Tab 1: Priority Queue ─────────────────────────────────────────────────
    with tab1:
        st.subheader("Phase I — Prioritized Job Queue")
        cols = [
            "priority_rank", "job_id", "customer_name", "job_type",
            "new_customer", "asap", "last_service_days", "priority_score",
        ]
        st.dataframe(
            r["priority_queue"][cols],
            use_container_width=True,
            hide_index=True,
        )

    # ── Tab 2: Eligibility ────────────────────────────────────────────────────
    with tab2:
        st.subheader("Phase II — Eligible Technicians per Job")

        # Active / disabled constraint summary
        cflags = r["cfg"].get("constraint_flags", {})
        disabled = [
            k.replace("enable_", "").replace("_", " ").title()
            for k, v in cflags.items() if not v
        ]
        if disabled:
            st.warning(
                f"The following constraints are **disabled**: {', '.join(disabled)}. "
                "All technicians are eligible for those checks."
            )

        if r.get("job_cluster_map"):
            from collections import defaultdict
            zones = defaultdict(list)
            for jid, tid in r["job_cluster_map"].items():
                zones[tid].append(jid)
            st.info(
                f"Geo clustering active — {len(r['data']['technicians'])} zones "
                f"({', '.join(f'{t}: {len(j)} jobs' for t, j in sorted(zones.items()))})"
            )
            with st.expander("View cluster assignments"):
                cdf = pd.DataFrame([
                    {"job_id": jid, "assigned_zone_tech": tid}
                    for jid, tid in r["job_cluster_map"].items()
                ])
                st.dataframe(cdf, use_container_width=True, hide_index=True)

        display = r["eligibility_df"].copy()
        display["eligible_technicians"] = display["eligible_technicians"].apply(
            lambda x: ", ".join(x) if x else "NONE"
        )
        st.dataframe(display, use_container_width=True, hide_index=True)

    # ── Tab 3: Route Map ──────────────────────────────────────────────────────
    with tab3:
        st.subheader("Phase III — Optimized Routes")

        if routes:
            cols_r = st.columns(len(routes))
            for i, (tid, info) in enumerate(routes.items()):
                stops     = [s for s in info["route"] if s not in ("Depot", "BREAK")]
                shift_min = info.get("shift_minutes", 480)
                util_pct  = round(info["total_minutes"] / shift_min * 100, 1) if shift_min else 0
                with cols_r[i]:
                    st.metric(
                        label=f"Tech {tid}",
                        value=f"{len(stops)} stops",
                        delta=f"{info['total_minutes']} / {shift_min} min ({util_pct}%)",
                    )
        else:
            st.warning("No routes generated — showing job locations only.")

        if r["unassigned"]:
            st.error(f"Unassigned jobs ({len(r['unassigned'])}): {', '.join(r['unassigned'])}")

        with st.container():
            st.components.v1.html(r["map_html"], height=620, scrolling=False)

        with open(os.path.join(OUT_DIR, "route_map.html"), "rb") as f:
            st.download_button(
                label="Download Route Map (HTML)",
                data=f,
                file_name="route_map.html",
                mime="text/html",
            )

    # ── Tab 4: KPI Dashboard ──────────────────────────────────────────────────
    with tab4:
        st.subheader("KPI Dashboard")
        if r["kpi_df"] is not None:
            st.dataframe(r["kpi_df"], use_container_width=True, hide_index=True)

            # Per-tech utilisation table
            if routes:
                st.subheader("Per-Technician Utilization")
                util_rows = []
                for tid, info in routes.items():
                    stops      = [s for s in info["route"] if s not in ("Depot", "BREAK")]
                    shift_min  = info.get("shift_minutes", 480)
                    eff_min    = info.get("eff_shift_minutes", shift_min)
                    lunch_min  = info.get("lunch_minutes", 0)
                    target_min = info.get("target_minutes", eff_min * 0.9)
                    # Utilization vs. full shift so lunch is treated as productive time
                    util_pct   = round(info["total_minutes"] / shift_min * 100, 1) if shift_min else 0
                    band_low   = r["cfg"].get("utilization", {}).get("band_low_pct", 85)
                    band_high  = r["cfg"].get("utilization", {}).get("band_high_pct", 95)
                    status_badge = (
                        "In band" if band_low <= util_pct <= band_high
                        else ("Under" if util_pct < band_low else "Over")
                    )
                    util_rows.append({
                        "Tech": tid,
                        "Stops": len(stops),
                        "Used (min)": info["total_minutes"],
                        "Shift (min)": shift_min,
                        "Lunch (min)": lunch_min,
                        "OR-Tools Budget (min)": eff_min,
                        "Target (min)": round(target_min, 0),
                        "Util %": util_pct,
                        "Band Status": status_badge,
                    })
                st.dataframe(pd.DataFrame(util_rows), use_container_width=True, hide_index=True)
        else:
            st.warning("No routes generated — KPIs unavailable.")

    # ── Tab 5: Service History ────────────────────────────────────────────────
    with tab5:
        st.subheader("Service History — Area-Based Duration")

        jobs_df    = r["data"]["jobs"]
        programs_df = r["data"]["programs"]

        # Build rate lookup
        rate_col     = ("minutes_per_1000sqft" if "minutes_per_1000sqft" in programs_df.columns
                        else "service_time_minutes")
        mat_col      = ("material_per_1000sqft" if "material_per_1000sqft" in programs_df.columns
                        else "material_required")
        rate_map     = programs_df.set_index("program_id")[rate_col].to_dict()
        mat_rate_map = programs_df.set_index("program_id")[mat_col].to_dict()

        # Service rate reference card
        st.caption("**Service Rates (from programs.csv)**")
        rate_rows = []
        for _, prog_row in programs_df.iterrows():
            rate_rows.append({
                "Program":               prog_row["program_id"],
                "Min / 1,000 sqft":      prog_row[rate_col],
                "Material / 1,000 sqft": prog_row[mat_col],
                "Skill Required":        prog_row.get("required_skill", ""),
                "Equipment Required":    prog_row.get("required_equipment", ""),
            })
        st.dataframe(pd.DataFrame(rate_rows), use_container_width=True, hide_index=True)
        st.caption(
            "**Formula:** `duration_minutes = ceil(lawn_area_sqft / 1,000 × min/1,000sqft)`  "
            "|  `material = round(lawn_area_sqft / 1,000 × material/1,000sqft)`"
        )
        st.divider()

        # Per-job service breakdown
        svc_rows = []
        for _, job in jobs_df.iterrows():
            pid   = job["program_id"]
            area  = int(job["lawn_area_sqft"])
            rate  = rate_map.get(pid, 0)
            m_rate = mat_rate_map.get(pid, 0)
            dur   = int(job.get("duration_minutes", 0))
            mat   = int(job.get("material_required", 0))
            svc_rows.append({
                "Job ID":           job["job_id"],
                "Customer":         job["customer_name"],
                "Program":          pid,
                "Lawn Area (sqft)": area,
                f"Rate (min/1k sqft)": rate,
                "Duration (min)":   dur,
                f"Mat Rate (/1k sqft)": m_rate,
                "Material Needed":  mat,
                "Revenue ($)":      job["revenue"],
            })
        svc_df = pd.DataFrame(svc_rows)

        # Summary stats
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Avg Lawn Area",   f"{int(svc_df['Lawn Area (sqft)'].mean()):,} sqft")
        col_b.metric("Avg Duration",    f"{svc_df['Duration (min)'].mean():.0f} min")
        col_c.metric("Largest Property", f"{svc_df['Lawn Area (sqft)'].max():,} sqft")
        col_d.metric("Smallest Property", f"{svc_df['Lawn Area (sqft)'].min():,} sqft")

        st.subheader("All Jobs — Computed Service Times")
        st.dataframe(svc_df, use_container_width=True, hide_index=True)

        # Per-program breakdown
        st.subheader("By Program")
        prog_summary = (
            svc_df.groupby("Program")
            .agg(
                Jobs=("Job ID", "count"),
                Avg_Area=("Lawn Area (sqft)", "mean"),
                Avg_Duration=("Duration (min)", "mean"),
                Total_Material=("Material Needed", "sum"),
            )
            .reset_index()
            .rename(columns={
                "Avg_Area": "Avg Area (sqft)",
                "Avg_Duration": "Avg Duration (min)",
                "Total_Material": "Total Material",
            })
        )
        prog_summary["Avg Area (sqft)"]    = prog_summary["Avg Area (sqft)"].round(0).astype(int)
        prog_summary["Avg Duration (min)"] = prog_summary["Avg Duration (min)"].round(1)
        st.dataframe(prog_summary, use_container_width=True, hide_index=True)

    # ── Tab 6: Optimization Summary ───────────────────────────────────────────
    with tab6:
        st.subheader("Optimization Summary")
        if r["summary"]:
            st.code(r["summary"], language=None)
        else:
            st.warning("No summary available.")

    # ── Tab 7: AI Narrative ───────────────────────────────────────────────────
    with tab7:
        st.subheader("AI Executive Narrative")
        run_cfg = r["cfg"]
        provider_name = run_cfg.get("llm", {}).get("provider", "none")

        if provider_name == "none":
            st.info(
                "LLM provider is set to **none**. "
                "Select Groq, Gemini, or Ollama in the sidebar and re-run to enable."
            )
        elif not routes:
            st.warning("No routes to summarize.")
        else:
            st.caption(
                f"Provider: **{provider_name}** · "
                "Generates a plain-English 3–5 sentence executive summary."
            )

            # Show cached narrative if already generated this session
            if "llm_narrative" not in st.session_state:
                st.session_state["llm_narrative"] = None

            gen_btn = st.button("Generate AI Summary", type="primary")

            if gen_btn:
                with st.spinner(f"Calling {provider_name}…"):
                    api_key   = run_cfg.get("llm", {}).get("api_key", "")
                    narrative = generate_narrative(
                        routes=routes,
                        unassigned=r["unassigned"],
                        jobs=r["data"]["jobs"],
                        technicians=r["data"]["technicians"],
                        dist_data=r["dist_data"],
                        cfg=run_cfg,
                        api_key=api_key,
                        priority_queue=r.get("priority_queue"),
                        eligibility_df=r.get("eligibility_df"),
                    )
                    st.session_state["llm_narrative"] = narrative

            if st.session_state.get("llm_narrative"):
                st.markdown(st.session_state["llm_narrative"])
                st.caption(
                    "Note: AI-generated narrative based on route metrics. "
                    "Verify against the Summary tab for full detail."
                )

else:
    # ── Landing state ─────────────────────────────────────────────────────────
    st.info("Configure settings in the sidebar and click **Run Optimizer** to begin.")

    try:
        sample = _load_data()
        with st.expander("Preview built-in sample data", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                st.caption("**Jobs**")
                st.dataframe(sample["jobs"].head(5), use_container_width=True, hide_index=True)
                st.caption("**Trucks**")
                st.dataframe(sample["trucks"], use_container_width=True, hide_index=True)
            with c2:
                st.caption("**Technicians**")
                st.dataframe(
                    sample["technicians"][["tech_id", "name", "shift_start", "shift_end", "skills", "available_days"]].head(5),
                    use_container_width=True, hide_index=True,
                )
                st.caption("**Programs**")
                st.dataframe(sample["programs"], use_container_width=True, hide_index=True)
    except Exception:
        pass


# ── Service History & Map Comparison (always visible) ────────────────────────
st.divider()
st.header("Service History & Map Comparison")

_sh_df, _ll_df, _br_df = _load_sh_data()

if _sh_df is None:
    st.warning(
        "Service history files not found. "
        "Ensure `service_history.xlsx` and `customer_lat_long.xlsx` are in "
        f"`{DATA_DIR}`."
    )
else:
    sh_tab1, sh_tab2 = st.tabs(["Service History Map", "Side-by-Side Comparison"])

    # ── Tab: Service History Map ──────────────────────────────────────────────
    with sh_tab1:
        st.subheader("Service History Routes by Technician")

        _dates = available_dates(_sh_df)
        _default_sh = _dates[0] if _dates else date(2026, 6, 11)

        c_date, c_btn = st.columns([3, 1])
        with c_date:
            sh_date = st.date_input(
                "Service Date",
                value=_default_sh,
                min_value=_dates[-1] if _dates else date(2025, 1, 1),
                max_value=_dates[0]  if _dates else date(2026, 12, 31),
                key="sh_date_tab1",
                help=f"Available dates: {_dates[-1]} → {_dates[0]}"
            )
        with c_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            sh_load = st.button("Load Map", key="sh_load_tab1", use_container_width=True)

        if sh_load or st.session_state.get("sh_map_html_tab1") is not None:
            if sh_load:
                with st.spinner("Building service history map…"):
                    _html, _mdf, _summ = build_sh_map(_sh_df, _ll_df, _br_df, sh_date)
                if _html is None:
                    st.warning(f"No service records found for {sh_date}.")
                    st.session_state.pop("sh_map_html_tab1", None)
                    st.session_state.pop("sh_merged_tab1", None)
                    st.session_state.pop("sh_summ_tab1", None)
                else:
                    st.session_state["sh_map_html_tab1"] = _html
                    st.session_state["sh_merged_tab1"]   = _mdf
                    st.session_state["sh_summ_tab1"]     = _summ

            if st.session_state.get("sh_map_html_tab1"):
                summ  = st.session_state["sh_summ_tab1"]
                mdf   = st.session_state["sh_merged_tab1"]
                html  = st.session_state["sh_map_html_tab1"]

                # Metrics strip
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Date",         str(summ["date"]))
                mc2.metric("Technicians",  summ["technicians"])
                mc3.metric("Total Stops",  summ["stops"])
                mc4.metric("Revenue",      f"${summ['revenue']:,.2f}")

                st.components.v1.html(html, height=580, scrolling=False)

                # Download service history map
                st.download_button(
                    "Download Service History Map (HTML)",
                    data=html,
                    file_name=f"service_history_{summ['date']}.html",
                    mime="text/html",
                )

                st.divider()
                st.subheader("Stop Detail")

                # Filters
                fc1, fc2 = st.columns(2)
                with fc1:
                    tech_filter = st.multiselect(
                        "Filter by Technician",
                        options=summ["techs"],
                        default=summ["techs"],
                        key="sh_tech_filter",
                    )
                with fc2:
                    prog_filter = st.multiselect(
                        "Filter by Program",
                        options=sorted(mdf["PROGRAM_CODE"].unique()),
                        default=sorted(mdf["PROGRAM_CODE"].unique()),
                        key="sh_prog_filter",
                    )

                display_df = mdf[
                    mdf["TECHNICIAN"].isin(tech_filter) &
                    mdf["PROGRAM_CODE"].isin(prog_filter)
                ].copy()

                disp_cols = [
                    c for c in [
                        "CUSTOMER_NUMBER", "CUST_NAME", "STREET_ADDRESS", "CITY",
                        "TECHNICIAN", "PROGRAM_CODE", "CATEGORY",
                        "REVENUE", "SERVICE_SIZE",
                    ] if c in display_df.columns
                ]
                st.dataframe(
                    display_df[disp_cols].rename(columns={
                        "CUSTOMER_NUMBER": "Customer #",
                        "CUST_NAME":       "Name",
                        "STREET_ADDRESS":  "Address",
                        "CITY":            "City",
                        "TECHNICIAN":      "Technician",
                        "PROGRAM_CODE":    "Program",
                        "CATEGORY":        "Category",
                        "REVENUE":         "Revenue ($)",
                        "SERVICE_SIZE":    "Size (k sqft)",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )

                # Per-tech summary
                st.subheader("Per-Technician Summary")
                tech_summ = (
                    display_df.groupby("TECHNICIAN")
                    .agg(
                        Stops=("CUSTOMER_NUMBER", "count"),
                        Revenue=("REVENUE", "sum"),
                        Programs=("PROGRAM_CODE", lambda x: ", ".join(sorted(x.unique()))),
                    )
                    .reset_index()
                    .rename(columns={"TECHNICIAN": "Technician"})
                )
                tech_summ["Revenue"] = tech_summ["Revenue"].map("${:,.2f}".format)
                st.dataframe(tech_summ, use_container_width=True, hide_index=True)

    # ── Tab: Side-by-Side Comparison ─────────────────────────────────────────
    with sh_tab2:
        st.subheader("Optimizer Routes vs. Service History Routes")
        st.caption(
            "Left — OR-Tools optimized routes (last run). "
            "Right — Actual service history for the selected date."
        )

        _dates2 = available_dates(_sh_df)
        _default2 = _dates2[0] if _dates2 else date(2026, 6, 11)

        cc_date, cc_btn = st.columns([3, 1])
        with cc_date:
            sh_date2 = st.date_input(
                "Service Date",
                value=_default2,
                min_value=_dates2[-1] if _dates2 else date(2025, 1, 1),
                max_value=_dates2[0]  if _dates2 else date(2026, 12, 31),
                key="sh_date_tab2",
            )
        with cc_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            cmp_load = st.button("Load Comparison", key="cmp_load", use_container_width=True)

        if cmp_load:
            with st.spinner("Building service history map for comparison…"):
                _chtml, _cmdf, _csumm = build_sh_map(_sh_df, _ll_df, _br_df, sh_date2)
            if _chtml is None:
                st.warning(f"No service records for {sh_date2}.")
                st.session_state.pop("cmp_sh_html", None)
            else:
                st.session_state["cmp_sh_html"]  = _chtml
                st.session_state["cmp_sh_summ"]  = _csumm

        # Left map: optimizer route_map.html
        opt_map_path = os.path.join(OUT_DIR, "route_map.html")
        opt_html = None
        if os.path.exists(opt_map_path):
            with open(opt_map_path, "r", encoding="utf-8") as _f:
                opt_html = _f.read()

        cmp_sh_html = st.session_state.get("cmp_sh_html")

        if opt_html or cmp_sh_html:
            ml, mr = st.columns(2)

            with ml:
                st.markdown("### Optimizer Routes")
                if opt_html:
                    st.components.v1.html(opt_html, height=520, scrolling=False)
                    with open(opt_map_path, "rb") as _f:
                        st.download_button(
                            "Download Optimizer Map",
                            data=_f,
                            file_name="route_map.html",
                            mime="text/html",
                            key="dl_opt_cmp",
                        )
                else:
                    st.info("Run the optimizer first to see the route map here.")

            with mr:
                st.markdown(f"### Service History — {sh_date2}")
                if cmp_sh_html:
                    summ2 = st.session_state.get("cmp_sh_summ", {})
                    sc1, sc2, sc3 = st.columns(3)
                    sc1.metric("Technicians", summ2.get("technicians", "—"))
                    sc2.metric("Stops",       summ2.get("stops", "—"))
                    sc3.metric("Revenue",     f"${summ2.get('revenue', 0):,.2f}")
                    st.components.v1.html(cmp_sh_html, height=520, scrolling=False)
                    st.download_button(
                        "Download Service History Map",
                        data=cmp_sh_html,
                        file_name=f"service_history_{sh_date2}.html",
                        mime="text/html",
                        key="dl_sh_cmp",
                    )
                else:
                    st.info("Click **Load Comparison** to render the service history map.")
        else:
            st.info(
                "Click **Load Comparison** above. "
                "Also run the Optimizer (sidebar) to populate the left-hand route map."
            )


# ── Route Assistant Chatbot (always visible) ──────────────────────────────────
# st.divider()
# st.header("Route Assistant")
# st.caption(
#     "Ask anything about the optimizer output, trade-off decisions, or routing logic. "
#     "Run the optimizer first for context-aware answers."
# )

# # API key input (reads from .env; user can override inline)
# _env_groq_key = os.environ.get("GROQ_API_KEY", "")
# with st.expander("Groq API Key", expanded=not bool(_env_groq_key)):
#     _chat_api_key = st.text_input(
#         "Groq API Key",
#         value=_env_groq_key,
#         type="password",
#         placeholder="gsk_...",
#         help="Set GROQ_API_KEY in routing_poc/.env to avoid entering it here.",
#         key="chat_api_key_input",
#     )
#     _chat_model = st.selectbox(
#         "Model",
#         ["llama-3.3-70b-versatile", "llama-3.1-70b-versatile",
#          "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
#         index=0,
#         key="chat_model_select",
#     )

# # Initialise chat history
# if "chat_messages" not in st.session_state:
#     st.session_state["chat_messages"] = []

# # Clear chat button
# if st.button("Clear chat", key="clear_chat"):
#     st.session_state["chat_messages"] = []
#     st.rerun()

# # Render existing messages
# for msg in st.session_state["chat_messages"]:
#     with st.chat_message(msg["role"]):
#         st.markdown(msg["content"])

# # Chat input
# if prompt := st.chat_input(
#     "Ask about routes, unassigned jobs, trade-offs, technician utilization…"
# ):
#     # Show and store user message
#     with st.chat_message("user"):
#         st.markdown(prompt)
#     st.session_state["chat_messages"].append({"role": "user", "content": prompt})

#     # Build context from latest optimizer results (if any)
#     _ctx = build_context(st.session_state.get("results"))

#     # Stream assistant reply
#     with st.chat_message("assistant"):
#         response_text = st.write_stream(
#             stream_response(
#                 messages=st.session_state["chat_messages"][:-1]
#                 + [{"role": "user", "content": prompt}],
#                 context=_ctx,
#                 api_key=_chat_api_key or _env_groq_key,
#                 model=_chat_model,
#             )
#         )

#     st.session_state["chat_messages"].append(
#         {"role": "assistant", "content": response_text}
#     )
