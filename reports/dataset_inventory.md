# PathSnap Dataset Inventory & Catalog Report

- **Generated At**: 2026-11-18 14:37:22
- **Total Partitioned Trips**: 90 (80 Training Trips, 0 Student-Testing Trips)
- **Total Raw GPS Observations**: 227,577
- **Total Road Network Nodes Cataloged**: 2,024,174
- **Total Road Network Arcs (Edges) Cataloged**: 4,061,945
- **Median GPS Sampling Interval**: 1.0 second

## 1. Dataset Partition Summary

| partition | trips | gps_observations | invalid_gps | non_increasing_timestamps | gaps_over_2_seconds | suspicious_speed_gt_55mps | road_arcs | road_nodes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| student_test | 10 | 19160 | 0 | 0 | 138 | 4 | 185041 | 90689 |
| train | 80 | 208417 | 0 | 0 | 2937 | 981 | 3876904 | 1933485 |

## 2. Per-Trip Trajectory Inventory (Sample 15 Trips)

| partition | trip_id | gps_observations | median_sampling_seconds | max_gps_step_m | nodes_rows | arcs_rows | route_rows |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train | 1 | 1315 | 1.0 | 72.421260072692 | 88809 | 165790 | 525 |
| train | 2 | 7243 | 1.0 | 37.27435297766178 | 7936 | 16139 | 262 |
| train | 3 | 821 | 1.0 | 29.131621981777723 | 9138 | 18028 | 79 |
| train | 4 | 1064 | 1.0 | 139.238142002117 | 7350 | 15499 | 320 |
| train | 5 | 1045 | 1.0 | 63.741007056926 | 898 | 1579 | 105 |
| train | 6 | 1450 | 1.0 | 46.69489529269771 | 3259 | 6757 | 261 |
| train | 8 | 3093 | 1.0 | 70.17394155444876 | 4995 | 10421 | 173 |
| train | 10 | 2711 | 1.0 | 39.25079087773764 | 8502 | 15715 | 210 |
| train | 11 | 2828 | 1.0 | 50.32336097354301 | 2649 | 5393 | 565 |
| train | 12 | 1608 | 1.0 | 55.20990494706893 | 16376 | 38599 | 205 |
| train | 13 | 2580 | 1.0 | 141.54308786326538 | 155003 | 307392 | 788 |
| train | 14 | 448 | 1.0 | 91.46225725429596 | 4067 | 8288 | 164 |
| train | 16 | 1305 | 1.0 | 41.60390677904763 | 8205 | 15098 | 204 |
| train | 18 | 379 | 1.0 | 59.18472964221255 | 5980 | 11545 | 166 |
| train | 19 | 2354 | 1.0 | 55.07568203225223 | 52694 | 108517 | 545 |

*(Full 90-trip dataset catalog preserved in `results/dataset_inventory.csv`)*
