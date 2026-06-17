# TouchTurf Routing Engine POC

A proof-of-concept routing engine for field service scheduling.

## Features

- Phase I: Queue Priority (new customer, ASAP, job type, days since last visit)
- Phase II: Capacity Matching (skills, licenses, availability, equipment, location)
- Phase III: Route Optimization via OR-Tools + OSRM real road distances

## Stack

Python 3.11 · pandas · numpy · ortools · folium · scikit-learn · scipy · streamlit

## Run

```bash
pip install -r routing_poc/requirements.txt
streamlit run routing_poc/app.py
```
