import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment


def cluster_jobs_to_techs(jobs: pd.DataFrame, technicians: pd.DataFrame,
                           random_state: int = 42) -> dict:
    """
    Spatially partition jobs into k geographic zones (k = number of techs)
    using k-means on lat/lng, then assign each zone to the nearest tech depot
    via the Hungarian algorithm (minimises total centroid-to-depot distance).

    Returns: {job_id: tech_id}
    """
    n_techs = len(technicians)
    techs_reset = technicians.reset_index(drop=True)
    coords = jobs[["lat", "lng"]].values  # (n_jobs, 2)

    km = KMeans(n_clusters=n_techs, random_state=random_state, n_init=10)
    labels = km.fit_predict(coords)          # cluster label per job
    centroids = km.cluster_centers_          # (n_techs, 2)

    depot_coords = techs_reset[["start_lat", "start_lng"]].values  # (n_techs, 2)

    # Cost matrix: Euclidean distance in lat/lng space from each centroid to each depot
    cost = np.linalg.norm(
        centroids[:, None, :] - depot_coords[None, :, :], axis=2
    )  # (n_techs, n_techs)

    # Optimal 1-to-1 assignment: cluster i -> tech j
    cluster_rows, tech_cols = linear_sum_assignment(cost)
    cluster_to_tech = {
        int(cluster_rows[i]): techs_reset.iloc[tech_cols[i]]["tech_id"]
        for i in range(len(cluster_rows))
    }

    return {
        row["job_id"]: cluster_to_tech[int(labels[idx])]
        for idx, (_, row) in enumerate(jobs.iterrows())
    }


def print_clusters(job_cluster_map: dict) -> None:
    tech_jobs = defaultdict(list)
    for job_id, tech_id in job_cluster_map.items():
        tech_jobs[tech_id].append(job_id)
    print("\n=== Geographic Clusters (k-means) ===")
    for tech_id in sorted(tech_jobs):
        assigned = tech_jobs[tech_id]
        print(f"  {tech_id} ({len(assigned)} jobs): {assigned}")
    print("======================================\n")
