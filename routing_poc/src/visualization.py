import folium
import pandas as pd
import json
import os

DEPOT_LAT  = 37.5407
DEPOT_LNG  = -77.4360
DEPOT_NAME = "TouchTurf Depot"

TECH_COLORS = [
    "#2563EB",  # blue
    "#16A34A",  # green
    "#7C3AED",  # purple
    "#EA580C",  # orange
    "#DC2626",  # red
    "#0891B2",  # cyan
    "#CA8A04",  # amber
    "#BE185D",  # pink
]

_PERSON_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
    'fill="white" width="16" height="16">'
    '<path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4z'
    'M12 14c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z"/>'
    '</svg>'
)


# ── Static jobs-only map ───────────────────────────────────────────────────────

def build_jobs_map(jobs: pd.DataFrame, output_path: str = "output/jobs_map.html") -> folium.Map:
    m = folium.Map(location=[DEPOT_LAT, DEPOT_LNG], zoom_start=13, tiles="OpenStreetMap")
    folium.Marker(
        location=[DEPOT_LAT, DEPOT_LNG],
        popup=folium.Popup(DEPOT_NAME, max_width=200),
        tooltip=DEPOT_NAME,
        icon=folium.Icon(color="red", icon="home", prefix="fa"),
    ).add_to(m)
    for _, job in jobs.iterrows():
        popup_html = (
            f"<b>Job:</b> {job['job_id']}<br>"
            f"<b>Customer:</b> {job['customer_name']}<br>"
            f"<b>Type:</b> {job.get('job_type', 'N/A')}<br>"
            f"<b>Revenue:</b> ${job['revenue']}"
        )
        folium.CircleMarker(
            location=[job["lat"], job["lng"]], radius=8,
            color="#2563EB", fill=True, fill_color="#3B82F6", fill_opacity=0.8,
            popup=folium.Popup(popup_html, max_width=220),
            tooltip=f"{job['job_id']} - {job['customer_name']}",
        ).add_to(m)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    m.save(output_path)
    return m


# ── Route map helpers ──────────────────────────────────────────────────────────

def _build_route_js_data(routes: dict, job_info: dict, tech_info: dict) -> dict:
    """Waypoint data for the animation panel."""
    js_data = {}
    for i, (tid, info) in enumerate(routes.items()):
        tech   = tech_info.get(tid, {})
        t_lat  = float(tech.get("start_lat", DEPOT_LAT))
        t_lng  = float(tech.get("start_lng", DEPOT_LNG))
        color  = TECH_COLORS[i % len(TECH_COLORS)]

        waypoints = []
        stop_num  = 0
        for stop in info["route"]:
            if stop == "BREAK":
                if waypoints:
                    prev = waypoints[-1]
                    waypoints.append({"lat": prev["lat"], "lng": prev["lng"],
                                      "label": "BREAK", "info": "Lunch Break - 30 min",
                                      "stop_num": "B"})
            elif stop == "Depot":
                label = "Start" if not waypoints else "Return to Depot"
                waypoints.append({"lat": t_lat, "lng": t_lng, "label": "Depot",
                                   "info": f"Tech {tid} depot ({label})",
                                   "stop_num": "D"})
            else:
                stop_num += 1
                job = job_info.get(stop, {})
                waypoints.append({
                    "lat":      float(job.get("lat", t_lat)),
                    "lng":      float(job.get("lng", t_lng)),
                    "label":    stop,
                    "info":     (f"Stop #{stop_num}: {job.get('customer_name', stop)}"
                                 f" | {job.get('job_type','')} | ${job.get('revenue',0)}"
                                 f" | {job.get('duration_minutes',0)} min service"),
                    "stop_num": stop_num,
                })

        if len(waypoints) > 1:
            js_data[tid] = {"color": color, "waypoints": waypoints,
                            "total_minutes": info["total_minutes"],
                            "tech_name": str(tech.get("name", tid))}
    return js_data


def _filter_html(feature_group_names: dict, depot_data: dict, map_var: str) -> str:
    fg_json     = json.dumps(feature_group_names)
    depot_json  = json.dumps(depot_data)

    options = '<option value="">Show All</option>\n'
    for tid, d in depot_data.items():
        options += (f'<option value="{tid}" style="color:{d["color"]}">'
                    f'{tid} - {d["name"]}</option>\n')

    return f"""
<div id="filter-panel" style="
    position:fixed;top:80px;left:12px;z-index:9999;
    background:#fff;padding:16px 18px;border-radius:10px;
    box-shadow:0 4px 16px rgba(0,0,0,.25);width:220px;
    font-family:Arial,sans-serif;font-size:13px;">

  <div style="font-weight:700;font-size:15px;margin-bottom:10px;color:#1e3a5f;">
    Filter by Technician
  </div>

  <select id="filter-select" onchange="filterByTech(this.value)"
      style="width:100%;padding:6px;border-radius:5px;
             border:1px solid #ccc;font-size:13px;margin-bottom:8px;">
    {options}
  </select>

  <div id="filter-info" style="font-size:11px;color:#6b7280;min-height:16px;"></div>
</div>

<script>
(function(){{
  var FG_NAMES = {fg_json};
  var DEPOTS   = {depot_json};
  var MAP_VAR  = "{map_var}";
  var theMap   = null;
  var depotMkrs = {{}};

  var PERSON_SVG = '{_PERSON_SVG}';

  function getMap(){{
    if(!theMap) theMap = window[MAP_VAR];
    return theMap;
  }}

  function makeDepotIcon(color, isSelected){{
    var size = isSelected ? 38 : 26;
    var half = size/2;
    var svgSize = isSelected ? 20 : 14;
    var svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
            + 'fill="white" width="'+svgSize+'" height="'+svgSize+'">'
            + '<path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 '
            + '1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z"/>'
            + '</svg>';
    var bg  = isSelected ? color : '#6b7280';
    var bdr = isSelected ? '3px solid white' : '2px solid white';
    var shd = isSelected ? '0 2px 10px rgba(0,0,0,.55)' : '0 1px 4px rgba(0,0,0,.35)';
    return L.divIcon({{
      html: '<div style="background:'+bg+';border-radius:50%;width:'+size+'px;height:'+size+'px;'
          + 'display:flex;align-items:center;justify-content:center;'
          + 'border:'+bdr+';box-shadow:'+shd+';">'+svg+'</div>',
      iconSize:   [size, size],
      iconAnchor: [half, half],
      className:  ''
    }});
  }}

  window.addEventListener('load', function(){{
    var map = getMap();

    // Create one depot marker per tech (managed by JS, not Folium)
    Object.keys(DEPOTS).forEach(function(tid){{
      var d   = DEPOTS[tid];
      var mkr = L.marker([d.lat, d.lng], {{
        icon: makeDepotIcon(d.color, false),
        zIndexOffset: 500
      }});
      mkr.bindTooltip('<b>'+tid+'</b> - '+d.name+' depot');
      mkr.bindPopup('<b>'+tid+' - '+d.name+'</b><br>Start / End location');
      mkr.addTo(map);
      depotMkrs[tid] = mkr;
    }});
  }});

  window.filterByTech = function(techId){{
    var map  = getMap();
    var info = document.getElementById('filter-info');

    Object.keys(FG_NAMES).forEach(function(tid){{
      var fg   = window[FG_NAMES[tid]];
      var mkr  = depotMkrs[tid];
      var show = (techId === '' || tid === techId);

      // Feature group (polyline + stop markers)
      if(fg){{
        if(show && !map.hasLayer(fg)) fg.addTo(map);
        if(!show && map.hasLayer(fg)) map.removeLayer(fg);
      }}

      // Depot marker — human icon only for the selected tech
      if(mkr){{
        if(show){{
          if(!map.hasLayer(mkr)) mkr.addTo(map);
          mkr.setIcon(makeDepotIcon(DEPOTS[tid].color, techId !== '' && tid === techId));
          mkr.setZIndexOffset(tid === techId ? 1000 : 500);
        }} else {{
          if(map.hasLayer(mkr)) map.removeLayer(mkr);
        }}
      }}
    }});

    // Fit map to selected tech's route
    if(techId !== ''){{
      var fg = window[FG_NAMES[techId]];
      if(fg){{
        try{{
          var b = fg.getBounds();
          if(b.isValid()) map.fitBounds(b, {{padding:[60,60]}});
        }}catch(e){{}}
      }}
      var d = DEPOTS[techId];
      info.innerHTML = '<span style="color:'+d.color+';font-weight:bold;">&#9632;</span>'
                     + ' Showing '+techId+' - '+d.name;

      // Sync animation dropdown if present
      var animSel = document.getElementById('tech-select');
      if(animSel) animSel.value = techId;
    }} else {{
      info.innerText = 'Showing all technicians';
      var animSel = document.getElementById('tech-select');
      if(animSel) animSel.value = '';
    }}
  }};
}})();
</script>
"""


def _animation_html(route_js_data: dict, map_var: str) -> str:
    routes_json = json.dumps(route_js_data)
    return f"""
<div id="anim-panel" style="
    position:fixed;top:80px;right:12px;z-index:9999;
    background:#fff;padding:16px 18px;border-radius:10px;
    box-shadow:0 4px 16px rgba(0,0,0,.25);width:240px;
    font-family:Arial,sans-serif;font-size:13px;">

  <div style="font-weight:700;font-size:15px;margin-bottom:10px;color:#1e3a5f;">
    Route Animation
  </div>

  <label style="font-size:11px;color:#555;display:block;margin-bottom:3px;">Select Technician</label>
  <select id="tech-select"
      style="width:100%;padding:5px;border-radius:5px;border:1px solid #ccc;
             margin-bottom:10px;font-size:13px;">
    <option value="">-- choose tech --</option>
  </select>

  <div style="display:flex;gap:6px;margin-bottom:10px;">
    <button id="play-btn" onclick="animPlay()"
        style="flex:1;padding:6px;background:#2563eb;color:#fff;
               border:none;border-radius:5px;cursor:pointer;font-size:12px;">
      &#9654; Play
    </button>
    <button onclick="animReset()"
        style="flex:1;padding:6px;background:#6b7280;color:#fff;
               border:none;border-radius:5px;cursor:pointer;font-size:12px;">
      &#8635; Reset
    </button>
  </div>

  <label style="font-size:11px;color:#555;">Speed</label>
  <input id="speed-slider" type="range" min="1" max="5" value="3"
      style="width:100%;margin-bottom:8px;" oninput="updateSpeed(this.value)">

  <div id="anim-progress" style="font-size:11px;color:#888;margin-bottom:3px;"></div>
  <div id="anim-info"
      style="font-size:12px;color:#374151;background:#f3f4f6;
             padding:6px 8px;border-radius:5px;min-height:36px;line-height:1.5;">
    Select a tech and press Play.
  </div>
</div>

<script>
(function(){{
  var ROUTES  = {routes_json};
  var MAP_VAR = "{map_var}";
  var theMap  = null;
  var techId  = null;
  var segIdx  = 0;
  var stepT   = 0;
  var isPaused = false;
  var animTimer = null;
  var vehMarker = null;
  var STEPS   = 60;
  var FRAME_MS = 40;
  var speedMult = 1;

  function getMap(){{
    if(!theMap) theMap = window[MAP_VAR];
    return theMap;
  }}

  window.addEventListener('load', function(){{
    var sel = document.getElementById('tech-select');
    Object.keys(ROUTES).forEach(function(tid){{
      var r = ROUTES[tid];
      var opt = document.createElement('option');
      opt.value = tid;
      opt.textContent = tid+' - '+r.tech_name+' ('+r.total_minutes+' min)';
      opt.style.color = r.color;
      sel.appendChild(opt);
    }});
  }});

  function lerp(a,b,t){{ return [a[0]+(b[0]-a[0])*t, a[1]+(b[1]-a[1])*t]; }}

  function makeVehIcon(color){{
    return L.divIcon({{
      html:'<div style="width:18px;height:18px;border-radius:50%;background:'+color
          +';border:3px solid #fff;box-shadow:0 0 8px rgba(0,0,0,.55);"></div>',
      iconSize:[18,18], iconAnchor:[9,9], className:''
    }});
  }}

  function setInfo(t){{ document.getElementById('anim-info').innerText = t; }}
  function setProgress(c,tot){{
    document.getElementById('anim-progress').innerText = 'Leg '+c+' of '+(tot-1);
  }}

  function animStep(){{
    if(isPaused) return;
    var wpts = ROUTES[techId].waypoints;
    if(segIdx >= wpts.length-1){{
      clearInterval(animTimer); animTimer=null;
      document.getElementById('play-btn').innerHTML='&#9654; Play';
      setInfo('Route complete! Total: '+ROUTES[techId].total_minutes+' min');
      setProgress(wpts.length-1, wpts.length);
      return;
    }}
    var t   = stepT/STEPS;
    var pos = lerp([wpts[segIdx].lat,wpts[segIdx].lng],
                   [wpts[segIdx+1].lat,wpts[segIdx+1].lng], t);
    if(vehMarker) vehMarker.setLatLng(pos);
    getMap().panTo(pos,{{animate:true,duration:0.05}});
    stepT += speedMult;
    if(stepT >= STEPS){{
      stepT=0; segIdx++;
      if(segIdx < wpts.length){{
        setInfo(wpts[segIdx].info);
        setProgress(segIdx, wpts.length);
        if(wpts[segIdx].label==='BREAK'){{
          clearInterval(animTimer);
          setInfo('Lunch Break (30 min)');
          setTimeout(function(){{
            segIdx++;
            if(segIdx < wpts.length){{
              setInfo(wpts[segIdx].info);
              animTimer = setInterval(animStep, FRAME_MS);
            }}
          }}, 1500);
        }}
      }}
    }}
  }}

  window.animPlay = function(){{
    var sel = document.getElementById('tech-select').value;
    if(!sel){{ alert('Please select a technician first.'); return; }}

    if(animTimer && sel===techId){{
      isPaused = !isPaused;
      document.getElementById('play-btn').innerHTML =
        isPaused ? '&#9654; Resume' : '&#9646;&#9646; Pause';
      return;
    }}

    techId=sel; segIdx=0; stepT=0; isPaused=false;
    var wpts  = ROUTES[techId].waypoints;
    var color = ROUTES[techId].color;

    if(animTimer){{ clearInterval(animTimer); animTimer=null; }}
    if(vehMarker){{ getMap().removeLayer(vehMarker); vehMarker=null; }}

    vehMarker = L.marker([wpts[0].lat,wpts[0].lng],{{
      icon:makeVehIcon(color), zIndexOffset:1000
    }}).addTo(getMap());

    getMap().panTo([wpts[0].lat,wpts[0].lng],{{animate:true,duration:0.5}});
    setInfo(wpts[0].info);
    setProgress(0, wpts.length);
    document.getElementById('play-btn').innerHTML='&#9646;&#9646; Pause';
    animTimer = setInterval(animStep, FRAME_MS);

    // Sync filter dropdown
    var fs = document.getElementById('filter-select');
    if(fs && fs.value !== sel) fs.value = sel;
  }};

  window.animReset = function(){{
    if(animTimer){{ clearInterval(animTimer); animTimer=null; }}
    isPaused=false; segIdx=0; stepT=0;
    document.getElementById('play-btn').innerHTML='&#9654; Play';
    setInfo('Select a tech and press Play.');
    document.getElementById('anim-progress').innerText='';
    if(vehMarker && techId && ROUTES[techId]){{
      var wpts = ROUTES[techId].waypoints;
      vehMarker.setLatLng([wpts[0].lat,wpts[0].lng]);
      getMap().panTo([wpts[0].lat,wpts[0].lng]);
    }}
  }};

  window.updateSpeed = function(val){{ speedMult=parseInt(val); }};
}})();
</script>
"""


# ── Main route map ─────────────────────────────────────────────────────────────

def build_route_map(routes: dict, jobs: pd.DataFrame,
                    technicians: pd.DataFrame,
                    output_path: str = "output/route_map.html") -> folium.Map:

    m = folium.Map(location=[DEPOT_LAT, DEPOT_LNG], zoom_start=13, tiles="OpenStreetMap")

    job_info  = {row["job_id"]:  row for _, row in jobs.iterrows()}
    tech_info = {row["tech_id"]: row for _, row in technicians.iterrows()}

    # Static company depot marker
    folium.Marker(
        location=[DEPOT_LAT, DEPOT_LNG],
        popup=folium.Popup(DEPOT_NAME, max_width=200),
        tooltip=DEPOT_NAME,
        icon=folium.Icon(color="red", icon="home", prefix="fa"),
    ).add_to(m)

    feature_group_names = {}
    depot_data          = {}

    for i, (tid, info) in enumerate(routes.items()):
        color  = TECH_COLORS[i % len(TECH_COLORS)]
        tech   = tech_info.get(tid, {})
        t_lat  = float(tech.get("start_lat", DEPOT_LAT))
        t_lng  = float(tech.get("start_lng", DEPOT_LNG))

        # One FeatureGroup per tech — polyline + numbered stop markers
        fg = folium.FeatureGroup(name=f"Tech {tid}", show=True)

        # Route polyline
        coords = []
        for stop in info["route"]:
            if stop == "Depot":
                coords.append((t_lat, t_lng))
            elif stop != "BREAK" and stop in job_info:
                coords.append((job_info[stop]["lat"], job_info[stop]["lng"]))

        if len(coords) >= 2:
            folium.PolyLine(
                locations=coords, color=color, weight=4, opacity=0.75,
                tooltip=f"Tech {tid} - {info['total_minutes']} min",
            ).add_to(fg)

        # Numbered stop markers
        stop_num = 0
        for stop in info["route"]:
            if stop in ("Depot", "BREAK"):
                continue
            stop_num += 1
            if stop not in job_info:
                continue
            row = job_info[stop]
            lat, lng = row["lat"], row["lng"]
            pref = str(row.get("preferred_tech_id", "")).strip()
            pref_note = f"<br><b>Pref Tech:</b> {pref}" if pref else ""
            popup_html = (
                f"<b>Stop #{stop_num}:</b> {stop}<br>"
                f"<b>Customer:</b> {row['customer_name']}<br>"
                f"<b>Type:</b> {row.get('job_type','N/A')}<br>"
                f"<b>Revenue:</b> ${row['revenue']}<br>"
                f"<b>Duration:</b> {row['duration_minutes']} min<br>"
                f"<b>Tech:</b> {tid}{pref_note}"
            )
            icon_html = (
                f'<div style="background:{color};color:#fff;border-radius:50%;'
                f'width:26px;height:26px;display:flex;align-items:center;'
                f'justify-content:center;font-weight:bold;font-size:12px;'
                f'border:2px solid #fff;box-shadow:0 1px 3px rgba(0,0,0,.4);">'
                f'{stop_num}</div>'
            )
            folium.Marker(
                location=[lat, lng],
                popup=folium.Popup(popup_html, max_width=240),
                tooltip=f"#{stop_num} {stop} - Tech {tid}",
                icon=folium.DivIcon(html=icon_html, icon_size=(26, 26), icon_anchor=(13, 13)),
            ).add_to(fg)

        fg.add_to(m)
        feature_group_names[tid] = fg.get_name()
        depot_data[tid] = {
            "lat":   t_lat,
            "lng":   t_lng,
            "color": color,
            "name":  str(tech.get("name", tid)),
        }

    # Inject filter panel (top-left) + animation panel (top-right)
    js_data  = _build_route_js_data(routes, job_info, tech_info)
    map_var  = m.get_name()
    m.get_root().html.add_child(folium.Element(_filter_html(feature_group_names, depot_data, map_var)))
    m.get_root().html.add_child(folium.Element(_animation_html(js_data, map_var)))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    m.save(output_path)
    return m
