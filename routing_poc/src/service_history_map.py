"""
Service-history Folium map builder.

Joins service_history.xlsx with customer_lat_long.xlsx, groups stops by
technician, and renders coloured route polylines starting from the branch
depot (branch.csv) + numbered circle markers.
"""

from __future__ import annotations

from datetime import date

import folium
import pandas as pd

# Distinct hex colours for up to 15 technicians
_TECH_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#469990", "#dcbeff",
    "#9a6324", "#800000", "#aaffc3", "#000075", "#a9a9a9",
]


def load_service_data(
    sh_path: str,
    lat_lng_path: str,
    branch_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and lightly clean all three source files."""
    sh = pd.read_excel(sh_path)
    ll = pd.read_excel(lat_lng_path)
    br = pd.read_csv(branch_path)

    sh["TECHNICIAN"] = sh["TECHNICIAN"].fillna("Unknown")
    sh["SERVICE_DATE"] = pd.to_datetime(sh["SERVICE_DATE"])
    ll = ll.dropna(subset=["LATITUDE", "LONGITUDE"])
    return sh, ll, br


def available_dates(sh_df: pd.DataFrame) -> list[date]:
    return sorted(sh_df["SERVICE_DATE"].dt.date.unique(), reverse=True)


def build_sh_map(
    sh_df: pd.DataFrame,
    lat_long_df: pd.DataFrame,
    branch_df: pd.DataFrame,
    service_date: date,
) -> tuple[str | None, pd.DataFrame | None, dict | None]:
    """
    Build a Folium service-history map for one date.

    Each technician's route starts from the branch depot (branch.csv),
    matched via BRANCH_NUMBER in the service history records.

    Returns (html_string, merged_df, summary_dict) or (None, None, None).
    """
    day_df = sh_df[sh_df["SERVICE_DATE"].dt.date == service_date].copy()
    if day_df.empty:
        return None, None, None

    addr_cols = lat_long_df[
        ["CUSTOMER_NUMBER", "LATITUDE", "LONGITUDE",
         "STREET_ADDRESS", "CITY", "CUSTOMER_NAME"]
    ].rename(columns={"CUSTOMER_NAME": "CUST_NAME"})

    merged = day_df.merge(addr_cols, on="CUSTOMER_NUMBER", how="left")
    merged = merged.dropna(subset=["LATITUDE", "LONGITUDE"]).reset_index(drop=True)

    if merged.empty:
        return None, None, None

    # Build branch lookup: branch_id → (lat, lng, name)
    branch_lookup = {
        int(row["branch_id"]): (row["lat"], row["lng"], row["name"])
        for _, row in branch_df.iterrows()
    }

    center = [merged["LATITUDE"].mean(), merged["LONGITUDE"].mean()]
    m = folium.Map(location=center, zoom_start=11, tiles="CartoDB positron")

    techs = sorted(merged["TECHNICIAN"].unique())
    color_map = {t: _TECH_COLORS[i % len(_TECH_COLORS)] for i, t in enumerate(techs)}

    # Track which branch depots have been drawn (draw each depot marker once)
    drawn_depots: set[int] = set()

    for tech in techs:
        tdf = merged[merged["TECHNICIAN"] == tech].reset_index(drop=True)
        color = color_map[tech]

        # Determine this tech's branch (use the most common branch_id for the day)
        branch_id = int(tdf["BRANCH_NUMBER"].mode().iloc[0])
        depot = branch_lookup.get(branch_id)

        # Build coordinate list: depot → stop 1 → stop 2 → … → last stop
        stop_coords = list(zip(tdf["LATITUDE"], tdf["LONGITUDE"]))
        if depot:
            route_coords = [(depot[0], depot[1])] + stop_coords
        else:
            route_coords = stop_coords

        if len(route_coords) > 1:
            folium.PolyLine(
                route_coords, color=color, weight=3, opacity=0.75, tooltip=tech,
            ).add_to(m)

        # Depot marker (drawn once per unique branch, shared across all techs)
        if depot and branch_id not in drawn_depots:
            drawn_depots.add(branch_id)
            depot_popup = (
                f"<b>🏢 Depot</b><br>"
                f"{depot[2]}<br>"
                f"Branch #{branch_id}<br>"
                f"Lat: {depot[0]:.5f}, Lng: {depot[1]:.5f}"
            )
            folium.Marker(
                location=[depot[0], depot[1]],
                icon=folium.Icon(color="black", icon="home", prefix="fa"),
                popup=folium.Popup(depot_popup, max_width=220),
                tooltip=f"Depot — {depot[2]}",
            ).add_to(m)

        # Customer stop markers
        for idx, row in tdf.iterrows():
            cust = row.get("CUST_NAME") or str(row["CUSTOMER_NUMBER"])
            popup_html = (
                f"<b>Stop #{idx + 1}</b><br>"
                f"<b>{cust}</b><br>"
                f"#{row['CUSTOMER_NUMBER']}<br>"
                f"{row.get('STREET_ADDRESS','')}, {row.get('CITY','')}<br>"
                f"Tech: <b>{tech}</b><br>"
                f"Program: <b>{row['PROGRAM_CODE']}</b> ({row.get('CATEGORY','')})<br>"
                f"Revenue: <b>${row['REVENUE']:.2f}</b><br>"
                f"Size: {row.get('SERVICE_SIZE','')} k sqft"
            )
            folium.CircleMarker(
                location=[row["LATITUDE"], row["LONGITUDE"]],
                radius=7,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.85,
                popup=folium.Popup(popup_html, max_width=230),
                tooltip=f"#{idx + 1} · {cust}",
            ).add_to(m)

            folium.Marker(
                location=[row["LATITUDE"], row["LONGITUDE"]],
                icon=folium.DivIcon(
                    html=(
                        f'<div style="font-size:9px;color:white;background:{color};'
                        f'border-radius:50%;width:16px;height:16px;'
                        f'text-align:center;line-height:16px;font-weight:bold;'
                        f'margin-left:-8px;margin-top:-8px;">{idx + 1}</div>'
                    ),
                    icon_size=(16, 16),
                    icon_anchor=(8, 8),
                ),
            ).add_to(m)

    # Legend
    depot_names = {
        bid: branch_lookup[bid][2]
        for bid in drawn_depots
        if bid in branch_lookup
    }
    depot_legend = "".join(
        f'<div style="margin:3px 0">'
        f'<span style="font-size:13px;margin-right:6px;">🏢</span>'
        f'<b>Depot</b> — {name}'
        f'</div>'
        for name in depot_names.values()
    )
    tech_legend = "".join(
        f'<div style="margin:3px 0">'
        f'<span style="display:inline-block;width:13px;height:13px;border-radius:50%;'
        f'background:{color_map[t]};margin-right:6px;vertical-align:middle;"></span>'
        f'{t} <span style="color:#666">({len(merged[merged["TECHNICIAN"]==t])} stops)</span>'
        f'</div>'
        for t in techs
    )
    legend = (
        '<div style="position:fixed;bottom:30px;left:30px;z-index:9999;background:white;'
        'padding:12px 14px;border-radius:7px;border:1px solid #ccc;font-size:12px;'
        'max-height:340px;overflow-y:auto;box-shadow:0 2px 6px rgba(0,0,0,.15);">'
        f'<b>📅 {service_date}</b><br>'
        f'{depot_legend}'
        f'<hr style="margin:6px 0">'
        f'{tech_legend}'
        '</div>'
    )
    m.get_root().html.add_child(folium.Element(legend))

    summary = {
        "date":        service_date,
        "technicians": len(techs),
        "stops":       len(merged),
        "revenue":     round(merged["REVENUE"].sum(), 2),
        "programs":    merged["PROGRAM_CODE"].value_counts().to_dict(),
        "techs":       techs,
        "color_map":   color_map,
    }

    return m._repr_html_(), merged, summary
