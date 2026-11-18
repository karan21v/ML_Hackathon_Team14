# PathSnap: Real-trip integration patch (2026-11-18)

## What this changes

- **Fixes tests failing on Starlette `httpx2` requirement** by removing the unneeded `httpx` dependency from `requirements.txt` and changing tests to exercise the route functions directly. It does *not* alter your installed FastAPI/Starlette versions.
- Keeps the existing synthetic scenario UI and the working API.
- Adds optional `student_test` ZIP trip selection in the same UI. Each numeric trip is processed with **its own** `.nodes/.arcs` road network. Training has **not** been repeated.
- Computes the same **eight features** as the model-training code; chooses five nearest candidate roads per observation with the existing KD-tree candidate generator.
- Uses the saved Random Forest probability ranking and directed Viterbi map matching. If the trained model is missing, real-trip requests fail explicitly rather than silently claiming ML.
- Does not open `.route` anywhere in prediction. Official evaluator trip files are locked unless explicitly authorised.
- Real-trip dashboard preview uses the **first 120 consecutive GPS points** to keep the interactive demonstration responsive. This is plainly labelled; `max_points: 0` processes the entire trip but may be slow. It does **not** imply full-route accuracy.
- New local offline tests require no httpx or httpx2. The separate live smoke script uses the Python standard library.

## A. Apply to the demo worktree, NOT your main branch

```bash
cd ~/ML_Hackathon_Team14_demo
source ~/ML_Hackathon_Team14/.venv/bin/activate
unzip -o ~/Downloads/PathSnap_Real_Trip_Integration_Patch.zip -d .
python -m pip install -r requirements.txt
python -m pytest -q tests/test_demo_integration.py tests/test_real_trip_integration.py
```

The patch assumes the **PathSnap_Demo_Integration_Fixes.zip** was applied already, as in your last terminal output. If teammates changed these same files after your snapshot, inspect the `git diff` before proceeding.

## B. Run with actual training-compatible student-test data

On Laptop 1, with the original *Student folder.zip* in Downloads:

```bash
cd ~/ML_Hackathon_Team14_demo
source ~/ML_Hackathon_Team14/.venv/bin/activate
export PATHSNAP_MODEL_PATH="$HOME/ML_Hackathon_Team14/artifacts/handoff/deploy_model.joblib"
export PATHSNAP_REQUIRE_ML=1
export PATHSNAP_TRIPS_ZIP="$HOME/Downloads/Student folder.zip"
export PATHSNAP_TRIP_PARTITION=student_test
python scripts/run_server.py
```

Use a real file path if the archive has a different name. This starts API `http://127.0.0.1:8000` and UI `/ui`. On Laptop 2 use `http://<Laptop-1-IP>:8000/ui` on the same network.

In another Terminal:

```bash
cd ~/ML_Hackathon_Team14_demo
source ~/ML_Hackathon_Team14/.venv/bin/activate
python scripts/smoke_real_trip.py --url http://127.0.0.1:8000 --points 120
```

The dashboard dropdown should now contain **Synthetic demos** and **Real dataset** numeric trip IDs.

If `networkx` is not installed, the corrected requirements will install it if your package index has access. Verify with `python -c "import networkx; print(networkx.__version__)"`.

## C. Evaluator test only when explicitly allowed

`evaluator_test.zip` must not be used for development, training, tuning, or debugging if it is reserved for final evaluation. After authorisation, a judge or authorised operator can set:

```bash
export PATHSNAP_TRIPS_ZIP="$HOME/Downloads/evaluator_test (1).zip"
export PATHSNAP_TRIP_PARTITION=evaluator_test
export PATHSNAP_ALLOW_EVALUATOR=1
```

Restart the server. **No `.route` files are read during inference.** Do not claim HMM accuracy unless scoring has been performed using a separate, permitted post-inference comparison. The existing synthetic demos still function independently.

## D. Verify, review, and commit

```bash
cd ~/ML_Hackathon_Team14_demo
git status --short
git diff --stat
git diff --check
git add src/api/main.py src/api/real_trips.py ui/index.html requirements.txt \
  tests/test_demo_integration.py tests/test_real_trip_integration.py \
  scripts/smoke_real_trip.py APPLY_REAL_TRIPS.md
# Only after tests and the live student-test smoke check pass:
git commit -m "Integrate real GPS trip files with ML and Viterbi dashboard"
git push -u origin feature/demo-integrity
```

Create a PR into `main` **only after Rohit and Dhanush review the patch**, particularly differences from newer branch work. Do **not** commit the 225 MB `.joblib`, dataset ZIP, predictions from private evaluator data, or credentials.

## Known limits

- The Python model binary is absent from Git exports, so unit tests run with a toy predictor. You must verify the 225 MB real model by setting `PATHSNAP_MODEL_PATH` and running the live student-test smoke check.
- The candidate generator follows the project feature pipeline, but the bounded network-distance searches in this adapter approximate long/complex road paths for speed. This **does not certify official route accuracy**.
- A 120-fix preview is not the entire trip. For full-trip performance or route accuracy assessment, use `max_points=0` and a formally defined evaluation protocol after a successful preview.
- Do not mix synthetic and real road networks: each numeric trip has its own graph, loaded per request.
- This patch is built against the latest repository ZIP supplied in chat plus the previous demo patch. It does not claim to include unseen commits made after that export.
