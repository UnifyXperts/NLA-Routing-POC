# CLAUDE.md

## Project Name

TouchTurf Routing Engine POC

---

# Objective

Build a Google Colab Proof of Concept demonstrating the core routing engine described in the TouchTurf Routing Requirements document.

The goal is NOT to build ERPNext integration.

The goal is to prove that routing logic can:

* Prioritize jobs
* Match jobs to qualified technicians
* Calculate travel distances
* Generate optimized routes
* Respect constraints
* Visualize routes on a map

---

# Technology Stack

Python 3.11

Libraries:

* pandas
* numpy
* ortools
* folium
* geopy

Environment:

* Google Colab

---

# Data Files

Create CSV files.

## jobs.csv

Columns:

```text
job_id
customer_name
lat
lng
revenue
duration_minutes
new_customer
asap
last_service_days
program_id
job_type
preferred_tech_id
preferred_time_window
```

`job_type` values: `service_call` or `regular`

`preferred_tech_id` — optional, blank if no preference

`preferred_time_window` — optional, format `HH:MM-HH:MM`, blank if no preference

---

## technicians.csv

Columns:

```text
tech_id
name
start_lat
start_lng
max_hours
skills
licenses
available_days
shift_start
shift_end
```

`available_days` — comma-separated, e.g. `Mon,Tue,Wed,Thu,Fri`

`shift_start` / `shift_end` — format `HH:MM`, e.g. `07:00` / `15:00`

---

## trucks.csv

Columns:

```text
truck_id
tech_id
capacity
equipment
```

---

## programs.csv

Columns:

```text
program_id
required_skill
required_equipment
material_required
```

---

# POC Scope

The POC must implement only:

Phase I
Phase II
Phase III

from the routing requirements.

Ignore:

* Mobile App
* SMS
* ETA Notifications
* Delivery Notes
* Invoicing
* Route Confirmation Workflow

---

# PHASE I

Queue Priority

Reference:

Routing Requirements
Section 5.6

---

## Input

Jobs

---

## Rule 1

Avoid Days

POC:

Skip

No customer data available.

---

## Rule 2

New Customer

Highest Priority

---

## Rule 3

ASAP

Second Highest Priority

---

## Rule 4

Job Type

Service Calls have higher priority than non-service calls (regular jobs).

Sort order within each group:

```text
Group A — service_call jobs, sorted by days_since_last_visit descending
Group B — regular jobs, sorted by days_since_last_visit descending
```

Final queue = Group A + Group B

---

## Rule 5

Days Since Last Visit

Within each job type group, higher value means higher priority

---

## Output

Prioritized Job Queue

Example:

```text
J003
J009
J015
J010
J018
```

---

# PHASE II

Capacity Matching

Reference:

Routing Requirements
Section 6.1

---

For each job determine eligible technician-truck combinations.

---

## Check Skill

Program required skill must exist on technician skills list.

Example:

```text
Program requires: Aeration

Technician Skills: Aeration, Mosquito
```

Pass

---

## Check License

Program required license must exist on technician licenses list.

Example:

```text
Program requires: Pesticide License

Technician Licenses: Pesticide License, CDL
```

Pass

---

## Check Availability

Technician must be available on the route date.

Check 1 — Day of week must be in technician `available_days`.

Check 2 — Job `preferred_time_window` (if set) must fall within technician `shift_start` / `shift_end`.

Example:

```text
Route Date: Monday

Technician available_days: Mon,Tue,Wed,Thu,Fri
```

Pass

---

## Check Location

Technician start location must be within a reasonable proximity threshold to the job cluster.

Use OSRM drive time from technician `start_lat/start_lng` to job `lat/lng`.

Threshold:

```text
Max drive time from depot to first job: 60 minutes
```

Technicians whose depot exceeds 60 minutes to every job in the cluster are excluded.

---

## Check Equipment

Program required equipment must exist on truck.

Example:

```text
Program requires: Aerator

Truck has: Aerator
```

Pass

---

## Check Capacity

Truck capacity must be greater than or equal to material required.

Example:

```text
Material Needed = 100

Truck Capacity = 1000
```

Pass

---

## Output

Eligible Resource List

Example:

```text
J003

T01
T03
```

---

# PHASE III

Route Optimization

Reference:

Routing Requirements
Section 6

---

Goal:

Generate routes.

---

# Distance Matrix

Use OSRM (Open Source Routing Machine) for real road-network distances and drive times.

Do not use Haversine.

OSRM Public API endpoint:

```text
http://router.project-osrm.org/table/v1/driving/
```

Build a coordinate list of all locations (depot + jobs).

Make a single OSRM Table API call to retrieve the full NxN matrix in one request.

Store two matrices:

```python
distance_matrix   # meters
duration_matrix   # seconds
```

Both matrices are used downstream:

- `duration_matrix` feeds the OR-Tools optimizer (drive time objective)
- `distance_matrix` feeds the KPI dashboard (total distance metric)

---

# Optimization Objectives

All four objectives are active with fixed weights. Weights are not user-configurable.

---

## Objective Weights — Fixed

```python
OBJECTIVE_WEIGHTS = {
    "maximize_revenue":           1.0,
    "minimize_drive_time":        1.0,
    "meet_customer_preference":   1.0,
    "maximize_tech_utilization":  1.0,
}
```

Weights are hardcoded and balanced equally. Do not expose weights as a configurable option in the UI or config file.

---

## How weights interact

All four objectives are combined into a single OR-Tools cost expression:

```text
total_cost = (W1 * -revenue_term)
           + (W2 *  drive_time_term)
           + (W3 *  preference_penalty_term)
           + (W4 * -utilization_term)
```

The optimizer minimizes `total_cost`.

---

## Objective 1 — Maximize Revenue

Maximize total revenue of assigned jobs.

Revenue values come from `jobs.csv` column `revenue`.

In OR-Tools: negate revenue and add as a minimization term scaled by `OBJECTIVE_WEIGHTS["maximize_revenue"]`.

---

## Objective 2 — Minimize Drive Time

Minimize total drive time across all technician routes.

Drive time values come from `duration_matrix` (OSRM seconds).

Note: drive time only — does not include job service time. `duration_minutes` is handled separately in constraints.

Scaled by `OBJECTIVE_WEIGHTS["minimize_drive_time"]`.

---

## Objective 3 — Meet Customer Preference

Penalize routes that violate customer preferences.

Two preference types:

```text
preferred_tech_id     — job is assigned to a different technician
preferred_time_window — job is scheduled outside the preferred time window
```

For time window check, compute estimated arrival time at each job:

```text
arrival_time[j] = shift_start
                + sum(duration_matrix[prev][curr] for each leg up to j)   # drive seconds
                + sum(duration_minutes for all jobs before j)              # service time
```

If `arrival_time[j]` falls outside `preferred_time_window`, it is a preference violation.

Each violation adds a penalty term scaled by `OBJECTIVE_WEIGHTS["meet_customer_preference"]`.

---

## Objective 4 — Maximize Technician Utilization

Target technician utilization at up to 90% of available shift hours.

```text
actual_route_time = sum(duration_matrix drive seconds)     # total drive time
                  + sum(duration_minutes * 60)             # total job service time in seconds

target_time = max_hours * 3600 * 0.90                     # 90% utilization ceiling

idle_time = target_time - actual_route_time
```

Both drive time and job service time count toward utilization.

The optimizer targets filling each technician's route to 90% of their shift — not 100% — to preserve buffer for unexpected delays.

Scaled by `OBJECTIVE_WEIGHTS["maximize_tech_utilization"]`.

---

# Constraints

> **Note:** Weights and constraints are optional for the POC. The optimizer should run and produce a valid route even if some or all constraints are relaxed or omitted. Constraints are a best-effort enforcement layer, not a hard blocker.

## Work Hours

A technician's total route time cannot exceed their `max_hours`.

Total route time is calculated as:

```text
total_route_time (minutes) = sum of drive times between stops    (from duration_matrix, converted to minutes)
                           + sum of duration_minutes for each assigned job
```

Constraint enforced in OR-Tools:

```text
total_route_time <= max_hours * 60
```

Default `max_hours` = 8, so the hard limit is 480 minutes per technician.

---

## Capacity

Truck capacity cannot be exceeded.

---

## Lunch Break

Optional — not required.

Lunch break insertion is not enforced by the optimizer or post-processing.

If a lunch break is included in a future implementation, it should be inserted after the midpoint job in the sequence and counted toward `total_route_time` when checking the work hours constraint.

No OR-Tools break modeling required.

---

# Route Output

Generate route sequence.

Example:

```text
Tech T01

Depot
J003
J001
J011
Depot
```

---

# Visualization

Use Folium.

Display:

* Depot
* Stops
* Sequence Number
* Technician Routes

Output:

```text
route_map.html
```

---

# KPI Dashboard

Generate summary.

Metrics:

```text
Total Jobs
Assigned Jobs
Unassigned Jobs
Total Revenue
Total Distance
Average Stops Per Technician
```

Display as pandas dataframe.

---

# LLM Summary — Route Explanation

After the optimizer completes, call an LLM to generate a plain-English summary of the final route output.

## LLM Provider

Do NOT use the Anthropic API.

Use one of the following — confirmed at implementation time based on availability:

**Option A — AWS Bedrock (preferred if AWS credentials are available)**

```text
Service : Amazon Bedrock
Model   : amazon.titan-text-lite-v1  or  meta.llama3-8b-instruct-v1:0
Library : boto3
Auth    : AWS credentials via environment variables or IAM role
          (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION)
```

**Option B — Free/open alternative (no cloud account required)**

```text
Option B1 — Ollama (local)
  Model  : llama3 or mistral (pulled locally)
  Library: ollama Python package  or  requests to http://localhost:11434

Option B2 — Groq Cloud (free tier)
  Model  : llama3-8b-8192
  Library: groq Python SDK
  Auth   : GROQ_API_KEY environment variable

Option B3 — Google Gemini (free tier)
  Model  : gemini-1.5-flash
  Library: google-generativeai
  Auth   : GOOGLE_API_KEY environment variable
```

The integration code must be written so the provider can be swapped by changing a single constant (e.g. `LLM_PROVIDER = "bedrock"` or `"groq"` or `"ollama"`).

---

## What to include in the prompt

Send the following context to the LLM:

```text
- Total jobs scheduled vs. total jobs available
- Per-technician route: stop sequence, total drive time, total service time, utilization %
- Total revenue captured
- Number of unassigned jobs and the reason each was skipped (no eligible tech, capacity exceeded, etc.)
- Any customer preference violations (wrong tech assigned, time window missed)
```

## Expected output

The LLM should return a 3–5 sentence narrative, for example:

```text
Today's route covers 18 of 20 jobs across 3 technicians, capturing $4,200 in revenue.
Tech T01 is running at 87% utilization with 6 stops in the north cluster.
Tech T03 has a preference violation on J011 — the customer requested a 09:00–11:00 window
but estimated arrival is 11:45.
Two jobs (J007, J019) were unassigned because no technician holds the required Pesticide License.
```

## Integration point

Call the LLM after `dashboard.py` generates the KPI dataframe and before the final output is printed.

Print the LLM narrative to stdout below the KPI table.

If the LLM call fails (no credentials, network error, provider unavailable), print a fallback static summary derived directly from the KPI data — do not crash the program.

---

# Folder Structure

```text
routing_poc/

data/
    jobs.csv
    technicians.csv
    trucks.csv
    programs.csv

src/

    config.py

    data_loader.py

    phase1_priority.py

    phase2_matching.py

    distance_matrix.py

    optimizer.py

    visualization.py

    dashboard.py

main.py
```
---

# Flow diagram
flowchart TD
 
    A[Pacing<br/>All Open Orders] --> B[Routing Engine]
 
    %% Main branches
    B --> C[I. Queue Priority]
    B --> D[II. Capacity Match]
    B --> E[III. Objective Function]
 
    %% Queue Priority
    C --> P1[P1: Avoid Days<br/>Filter Out Blocked Days]
    P1 --> P1A[P1a: New Customer?<br/>Prioritize First-Time Customers]
    P1A --> P1B[P1b: ASAP Flag<br/>Urgent Jobs Pushed to Top]
    P1B --> P2[P2: Job Type<br/>2a Service Calls First<br/>2b Non-Service Calls]
    P2 --> P3[P3: Days Since Visited<br/>Sort 2a then Sort 2b]
 
    %% Capacity Matching
    D --> CM
 
    subgraph CM[Capacity Matching]
        J[Job<br/>License<br/>Skill]
        T[Technician<br/>Availability<br/>Location]
        TR[Truck<br/>Equipment<br/>Accessories<br/>Material]
 
        J --> T
        T --> TR
    end
 
    P3 --> CM
 
    %% Objective Function
    E --> OF
 
    subgraph OF[Objective Function]
        O1[Maximize Revenue]
        O2[Minimize Drive Time]
        O3[Meet Customer Preference]
        O4[Max Technician Utilization]
    end
 
    CM --> OF
 
    OF --> R[Optimized Route Schedule]

---

# Deliverables

Claude should generate:

1. CSV sample data
2. Data Loader
3. Priority Engine
4. Eligibility Engine
5. Distance Matrix Builder
6. OR-Tools Route Optimizer
7. Folium Map
8. LLM Route Summary (groq — plain-English narrative of results)
9. Main Runner Script

---

# Success Criteria

POC is successful if:

* 20 Richmond VA jobs load successfully
* Jobs are prioritized
* Eligible technicians are identified
* Routes are generated
* Map renders correctly
* KPIs are displayed
* Execution completes in under 30 seconds

```
```
