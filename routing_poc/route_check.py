import requests

pairs = [
    ("Depot -> J016", "-77.4360,37.5407", "-77.4720,37.5880"),
    ("J016 -> Depot", "-77.4720,37.5880", "-77.4360,37.5407"),
]

for label, a, b in pairs:
    url = f"http://router.project-osrm.org/route/v1/driving/{a};{b}?overview=false"
    r   = requests.get(url, timeout=15).json()
    leg = r["routes"][0]["legs"][0]
    km  = round(leg["distance"] / 1000, 2)
    mn  = round(leg["duration"] / 60,   1)
    print(f"{label}:  {km} km  /  {mn} min")
