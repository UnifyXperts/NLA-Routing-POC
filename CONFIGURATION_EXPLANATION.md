# Routing Engine Configuration Summary

## Priority Weights (Phase I)

These determine which jobs are prioritized first.

| Rule              | Weight | Meaning                                 |
| ----------------- | ------ | --------------------------------------- |
| New Customer      | 1000   | Highest priority                        |
| ASAP              | 100    | Urgent jobs                             |
| Last Service Days | 1      | Older service dates get higher priority |

Example:

```text
Priority Score =
(New Customer × 1000)
+ (ASAP × 100)
+ Last Service Days
```

---

## Optimization Weights (Phase III)

These determine what the route optimizer cares about.

| Weight                  | Value | Meaning                              |
| ----------------------- | ----- | ------------------------------------ |
| Distance Weight         | 1     | Minimize travel distance             |
| Revenue Weight          | 0     | Revenue ignored                      |
| Workload Balance Weight | 100   | Strongly balance technician workload |

Current behavior:

```text
Balanced Workload > Distance > Revenue
```

---

## Operational Constraints

| Constraint               | Value | Meaning                          |
| ------------------------ | ----- | -------------------------------- |
| Max Hours Per Technician | 8     | Technician cannot exceed 8 hours |
| Lunch After Minutes      | 240   | Insert break after 4 hours       |
| Lunch Break Minutes      | 30    | Lunch duration                   |

---

## Current Routing Logic

### Priority Order

```text
New Customer
    ↓
ASAP
    ↓
Longest Time Since Last Service
```

### Route Optimization

```text
Balance Technician Workload
    ↓
Minimize Distance
    ↓
Revenue Ignored
```

### Constraints

```text
Max 8 Working Hours
30 Minute Lunch Break
Lunch After 4 Hours
```
