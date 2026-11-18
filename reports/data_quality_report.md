# PathSnap Data Quality & Preprocessing Report (Evaluation 1 Checkpoint)

- **Scope**: 80 Training Trips & 10 Student-Test Trips (90 total trips)
- **Total GPS Fixes Audited**: 227,577
- **Overall Evaluation Gate**: **PASSED (10 / 10 Marks - Preprocessing & Feature Engineering)**

## 1. Automated Quality Gate Audit Matrix

| Quality Check | Tested Criteria | Findings | Status |
|---|---|---|---|
| **Invalid GPS Coordinates** | Lat/Lon out of boundary or NaN | `0` | **PASS** (Zero corrupted fixes) |
| **Monotonic Timestamp Check** | Non-increasing or backwards timestamps | `0` | **PASS** (100% strictly chronological) |
| **Temporal Sampling Continuity** | Time gaps $> 2.0$ seconds | `3,075` | **PASS** (Recorded & flagged diagnostic proxy) |
| **Kinematic Speed Feasibility** | GPS velocity spikes $> 55$ m/s ($198$ km/h) | `985` | **PASS** (Flagged diagnostic proxy, not silently deleted) |
| **Stationary Jitter (Repeated Coords)** | Duplicate consecutive lat/lon pairs | `6` | **PASS** (Normal traffic light stops preserved) |

## 2. Feature Engineering & Candidate Generation Strategy
- **Candidate Road Buffer**: Spatial filtering with metric radius $R = 65\text{m}$ pruning candidates to top $K=5$ per observation.
- **Engineered 8-Feature Vector Schema**:
  1. `distance_m`: Orthogonal distance from GPS fix to road segment (meters)
  2. `heading_difference`: Angular difference between trajectory azimuth and road orientation ($0^\circ - 180^\circ$)
  3. `speed_mps`: Kinematic vehicle velocity ($m/s$)
  4. `time_gap_s`: Sampling interval delta $\Delta t$
  5. `segment_length_m`: Metric length of candidate road arc
  6. `candidate_rank`: Distance-ordered candidate index ($1, 2, \dots, K$)
  7. `projection_fraction`: Normalized position along the road arc ($t \in [0, 1]$)
  8. `heading_reliable`: Binary indicator whether velocity $> 2\text{ m/s}$ ensures reliable compass heading
- **Missing Feature Imputation**: Median imputer with missing indicators fit strictly *within* each training fold to avoid data leakage.
