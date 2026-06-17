import numpy as np
import pandas as pd
import requests

OSRM_URL   = "http://router.project-osrm.org/table/v1/driving/"
NULL_VALUE = 999_999.0   # seconds / metres — used when OSRM returns null
OSRM_BLOCK = 45          # max unique locations per REQUEST SIDE (sources OR destinations)
                          # Each request sends ≤ 2×OSRM_BLOCK coords → URL stays ~2 KB


# ── Low-level OSRM block call ─────────────────────────────────────────────────

def _osrm_block_request(locs_subset: list, src_pos: list, dst_pos: list) -> tuple:
    """
    Single OSRM table API call.

    locs_subset : list of (lat, lng) — only the coordinates needed for this block
    src_pos     : positions within locs_subset to use as sources
    dst_pos     : positions within locs_subset to use as destinations

    Returns (raw_distances, raw_durations) — each a 2-D list [src][dst].
    """
    coords  = ";".join(f"{lng},{lat}" for lat, lng in locs_subset)
    src_str = ";".join(map(str, src_pos))
    dst_str = ";".join(map(str, dst_pos))
    url = (
        f"{OSRM_URL}{coords}"
        f"?annotations=distance,duration"
        f"&sources={src_str}&destinations={dst_str}"
    )
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if data.get("code") != "Ok":
        raise RuntimeError(
            f"OSRM error: {data.get('code')} — {data.get('message', '')}"
        )
    return data["distances"], data["durations"]


# ── Full NxN matrix builder with 2-D blocking ────────────────────────────────

def _call_osrm(locations: list) -> tuple:
    """
    Build a full NxN distance/duration matrix for any number of locations.

    Strategy — 2-D block decomposition:
      • Deduplicate coordinates so identical depot positions count once.
      • Split unique locations into row-chunks and column-chunks of OSRM_BLOCK.
      • Each OSRM request covers one (row-chunk × column-chunk) pair.
        URL contains ≤ 2×OSRM_BLOCK unique coordinates → always within HTTP limits.
      • Assemble blocks into the full unique matrix, then remap to original indices.

    For 190 jobs + 16 techs = 191 unique locations (1 shared depot + 190 jobs):
      • ceil(191 / 45) = 5 chunks per axis → 25 OSRM requests (≈ 10 s).
    """
    n_orig = len(locations)

    # ── Deduplicate ───────────────────────────────────────────────────────────
    unique_locs: list = []
    loc_to_uid:  dict = {}
    orig_to_uid: list = []

    for lat, lng in locations:
        key = (round(lat, 6), round(lng, 6))
        if key not in loc_to_uid:
            loc_to_uid[key] = len(unique_locs)
            unique_locs.append((lat, lng))
        orig_to_uid.append(loc_to_uid[key])

    n_uniq = len(unique_locs)

    dist_uniq = np.full((n_uniq, n_uniq), NULL_VALUE, dtype=float)
    dur_uniq  = np.full((n_uniq, n_uniq), NULL_VALUE, dtype=float)
    np.fill_diagonal(dist_uniq, 0.0)
    np.fill_diagonal(dur_uniq,  0.0)

    # ── 2-D block requests ────────────────────────────────────────────────────
    all_uid    = list(range(n_uniq))
    row_chunks = [all_uid[i:i + OSRM_BLOCK] for i in range(0, n_uniq, OSRM_BLOCK)]
    col_chunks = [all_uid[j:j + OSRM_BLOCK] for j in range(0, n_uniq, OSRM_BLOCK)]

    for src_chunk in row_chunks:
        for dst_chunk in col_chunks:
            # Combine unique ids needed for this block (ordered, no duplicates)
            needed     = list(dict.fromkeys(src_chunk + dst_chunk))
            uid_to_pos = {uid: pos for pos, uid in enumerate(needed)}
            locs_sub   = [unique_locs[u] for u in needed]
            src_pos    = [uid_to_pos[u] for u in src_chunk]
            dst_pos    = [uid_to_pos[u] for u in dst_chunk]

            raw_d, raw_t = _osrm_block_request(locs_sub, src_pos, dst_pos)

            for i, su in enumerate(src_chunk):
                for j, du in enumerate(dst_chunk):
                    if su == du:
                        dist_uniq[su][du] = 0.0
                        dur_uniq[su][du]  = 0.0
                    else:
                        vd = raw_d[i][j]
                        vt = raw_t[i][j]
                        dist_uniq[su][du] = NULL_VALUE if vd is None else float(vd)
                        dur_uniq[su][du]  = NULL_VALUE if vt is None else float(vt)

    # ── Remap unique → original indices (vectorised) ──────────────────────────
    uid_arr   = np.array(orig_to_uid)
    dist_full = dist_uniq[np.ix_(uid_arr, uid_arr)]
    dur_full  = dur_uniq[np.ix_(uid_arr, uid_arr)]

    return dist_full, dur_full


# ── Public builder ────────────────────────────────────────────────────────────

def build_distance_matrix(technicians: pd.DataFrame, jobs: pd.DataFrame) -> dict:
    """
    Build a combined NxN OSRM matrix for all tech start locations + all job locations.

    Index layout:
      0 .. n_techs-1               : technician depot nodes
      n_techs .. n_techs+n_jobs-1  : job nodes (in jobs.iterrows order)

    Returns:
      distance_m  : np.ndarray  (metres,  shape NxN)
      duration_s  : np.ndarray  (seconds, shape NxN)
      n_techs     : int
      tech_index  : {tech_id: row_index}
      job_index   : {job_id:  row_index}   (already offset by n_techs)
    """
    techs_reset = technicians.reset_index(drop=True)
    jobs_reset  = jobs.reset_index(drop=True)

    tech_locs = list(zip(techs_reset["start_lat"], techs_reset["start_lng"]))
    job_locs  = list(zip(jobs_reset["lat"],         jobs_reset["lng"]))
    locations = tech_locs + job_locs

    n_techs = len(techs_reset)

    distance_m, duration_s = _call_osrm(locations)

    tech_index = {
        row["tech_id"]: i
        for i, (_, row) in enumerate(techs_reset.iterrows())
    }
    job_index = {
        row["job_id"]: n_techs + j
        for j, (_, row) in enumerate(jobs_reset.iterrows())
    }
    return {
        "distance_m": distance_m,
        "duration_s": duration_s,
        "n_techs":    n_techs,
        "n_jobs":     len(jobs_reset),
        "tech_index": tech_index,
        "job_index":  job_index,
    }
