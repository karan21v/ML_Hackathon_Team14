# Team 14 | PathSnap | Darshan's Evaluation 1 contribution

**Owner:** Darshan R Mali — data loading, preprocessing, exploratory data analysis (EDA).

This is a tested *data and EDA component*, **not** a trained classifier or complete map matcher. All results must reflect real runs on your own permitted data; review, adapt and explain the code, and disclose AI assistance as required by the hackathon.

## Run on Windows / VS Code (PowerShell)

1. Clone your group's repo and add the `data_processing/` and `tests/` folders, `requirements.txt`, and `.gitignore` from this package to your personal branch.
2. Place the supplied **unmodified** `Student folder.zip` outside the repo or in an ignored location. **Don't commit the raw dataset.**
3. From the project root:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m data_processing.eda --zip "..\Student folder.zip" --trip 00000001 --out results
```

Alternatively, on Linux/macOS: `python -m venv .venv`, `.venv/bin/pip install -r requirements.txt`, `.venv/bin/python -m data_processing.eda --zip "../Student folder.zip"`.

## Outputs (generated from real training data)

| File | Use in first evaluation |
| --- | --- |
| `results/dataset_inventory.csv` | Per-trip counts, missing file types and GPS quality counts for the **80 training and 10 student-testing trips**. |
| `results/dataset_summary.csv` | Summary counts by data partition. |
| `results/cleaned_selected_trip.csv` | GPS observations + quality flags and stable `obs_idx` for a selected trip. |
| `results/trajectory_plot.png` | GPS observations, sampled roads and the reference-route arcs in a single plot. |
| `results/trajectory_map.html` | Folium interactive map with separately toggleable GPS / reference / road layers (background map tiles may require Internet). |
| `results/selected_trip_stats.json` | Raw counts for a selected trip, including invalid road segments / reference IDs. |

**The raw `.track` file has longitude, latitude, timestamp (in seconds); `.nodes` has lon/lat; `.arcs` contains 0-indexed node-ID pairs, and its row number is the road `arc_id`; `.route` is an ordered sequence of arc IDs.** The `.route` rows are *not* observation-by-observation GPS labels. `obs_idx` must stay stable across feature generation, supervised alignment, and Viterbi. Road arcs should not be reindexed when filtering bad edges.

## Handoff to Dhanush / Karan

```python
from data_processing.load_data import PathSnapDataset
from data_processing.preprocess import preprocess_track, haversine_m
with PathSnapDataset("../Student folder.zip") as ds:
    observations = preprocess_track(ds.read("train", "00000001", "track"))
    nodes = ds.read("train", "00000001", "nodes")
    arcs = ds.read("train", "00000001", "arcs")
    route = ds.read("train", "00000001", "route")
```

- **Dhanush** uses valid observation coordinates and original `obs_idx` to produce candidate rows and features. He must separately design, verify and document how `.route` segments are aligned to GPS observations. Do not simply match rows by index or use the nearest `route` arc without checking the consequences.
- **Karan** trains candidate-scoring models on *genuinely* labelled candidate features, with cross-validation grouped by trip, not randomly split GPS points. Keep `trip_id`, `obs_idx`, and `arc_id` as identifiers, not raw numerical model features.
- **Rohit** measures actual model performance and creates comparison charts after genuine experiments.
- **Evaluator-only data** is neither opened nor required by this package.

## Evaluation 1 demonstration (11:30 AM)

Show the table with trip and data quality counts, display `trajectory_plot.png`, open the interactive map, explain why GPS points are not perfectly on roads, show quality flags and the stable observation index, and give the feature-generation member a live handoff. Explain that candidate generation and labelling must complete before actual supervised model training.

**Caveats:** GPS speed flags above 55 m/s are *diagnostic*, not automatic deletion. Time gaps >2 s are recorded. Repeated identical GPS positions may mean a genuine vehicle stop. Quality checks do not silently delete observations; downstream modelling decides how to handle flagged records.
