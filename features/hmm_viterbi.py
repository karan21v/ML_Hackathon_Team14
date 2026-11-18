
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def normalize_id(value):
    """Normalize numeric node/arc IDs before comparing them."""
    if pd.isna(value):
        return None

    try:
        number = float(value)
        if number.is_integer():
            return str(int(number))
    except (TypeError, ValueError):
        pass

    return str(value)


def emission_log_probabilities(group):
    """Convert candidate probabilities into normalized log emissions."""
    scores = group["candidate_score"].to_numpy(dtype=float)
    scores = np.clip(scores, 1e-12, 1.0)

    total = scores.sum()

    if total <= 0:
        scores = np.ones(len(scores)) / len(scores)
    else:
        scores = scores / total

    return np.log(scores)


def transition_log_probability(previous, current):
    """Score transitions between consecutive candidate road segments."""

    previous_arc = normalize_id(previous["arc_id"])
    current_arc = normalize_id(current["arc_id"])

    previous_to = normalize_id(previous["to_node_id"])
    current_from = normalize_id(current["from_node_id"])

    # Staying on the same road segment is highly likely.
    if previous_arc == current_arc:
        previous_fraction = float(previous["projection_fraction"])
        current_fraction = float(current["projection_fraction"])

        # Penalize substantial backward movement along a directed arc.
        if current_fraction < previous_fraction - 0.15:
            return np.log(0.05)

        return np.log(0.90)

    # Moving from the end of one directed arc to the start of
    # another connected arc is a valid transition.
    if previous_to == current_from and previous_to is not None:
        return np.log(0.80)

    # Disconnected transitions are strongly discouraged,
    # but not made impossible.
    return np.log(0.001)


def viterbi_decode(groups):
    """Find the most likely sequence of candidate road segments."""

    emissions = [
        emission_log_probabilities(group)
        for group in groups
    ]

    # Initial observation.
    scores = emissions[0].copy()
    backpointers = []

    # Process every subsequent observation.
    for t in range(1, len(groups)):
        previous_group = groups[t - 1]
        current_group = groups[t]

        next_scores = np.full(len(current_group), -np.inf)
        pointers = np.zeros(len(current_group), dtype=int)

        for j, current_candidate in enumerate(
            current_group.to_dict("records")
        ):
            possible_scores = []

            for i, previous_candidate in enumerate(
                previous_group.to_dict("records")
            ):
                transition = transition_log_probability(
                    previous_candidate,
                    current_candidate
                )

                possible_scores.append(scores[i] + transition)

            best_previous = int(np.argmax(possible_scores))

            next_scores[j] = (
                possible_scores[best_previous] + emissions[t][j]
            )

            pointers[j] = best_previous

        scores = next_scores
        backpointers.append(pointers)

    # Backtrack from the highest-scoring final state.
    state = int(np.argmax(scores))
    selected_indices = [state]

    for pointers in reversed(backpointers):
        state = int(pointers[state])
        selected_indices.append(state)

    selected_indices.reverse()

    predictions = []

    for group, selected_index in zip(groups, selected_indices):
        predictions.append(group.iloc[selected_index])

    return pd.DataFrame(predictions), float(np.max(scores))


def score_candidates(data, bundle):
    """Generate candidate probabilities using the trained ML pipeline."""

    feature_names = bundle["feature_names"]

    missing = [
        feature for feature in feature_names
        if feature not in data.columns
    ]

    if missing:
        raise ValueError(f"Missing model features: {missing}")

    X = data[feature_names].replace(
        [np.inf, -np.inf], np.nan
    )

    # Convert to NumPy to avoid feature-name warnings.
    X = X.to_numpy(dtype=float)

    model = bundle["model"]
    probabilities = model.predict_proba(X)

    classes = list(model.classes_)

    if 1 not in classes:
        raise ValueError("Model does not contain positive class 1")

    positive_index = classes.index(1)

    data = data.copy()
    data["candidate_score"] = probabilities[:, positive_index]

    return data


def reconstruct_route(data):
    """Run Viterbi independently for each trip."""

    results = []
    total_score = 0.0

    for trip_id, trip in data.groupby("trip_id", sort=True):

        groups = []

        for _, group in trip.groupby(
            "observation_index", sort=True
        ):
            groups.append(
                group.reset_index(drop=True)
            )

        if not groups:
            continue

        predicted, score = viterbi_decode(groups)

        predicted["trip_id"] = trip_id
        predicted["viterbi_score"] = score

        results.append(predicted)
        total_score += score

    if not results:
        raise ValueError("No candidate observations available")

    return pd.concat(results, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(
        description="PathSnap HMM/Viterbi route reconstruction"
    )

    parser.add_argument(
        "--features",
        default="results/features_00000001.csv"
    )

    parser.add_argument(
        "--model",
        default="artifacts/handoff/best_model.joblib"
    )

    parser.add_argument(
        "--output",
        default="results/hmm_predictions_00000001.csv"
    )

    args = parser.parse_args()

    print("Loading model...")
    bundle = joblib.load(args.model)

    print("Loading candidate features...")
    data = pd.read_csv(args.features)

    print("Generating ML candidate probabilities...")
    scored = score_candidates(data, bundle)

    print("Running HMM/Viterbi...")
    predictions = reconstruct_route(scored)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    predictions.to_csv(output_path, index=False)

    # Compress consecutive duplicate arcs into a route sequence.
    route = predictions[
        ["trip_id", "arc_id"]
    ].copy()

    route = route.loc[
        route["arc_id"].ne(route["arc_id"].shift())
        | route["trip_id"].ne(route["trip_id"].shift())
    ].reset_index(drop=True)

    route.insert(
        1,
        "route_step",
        route.groupby("trip_id").cumcount() + 1
    )

    route_path = output_path.with_name(
        output_path.stem + "_route.csv"
    )

    route.to_csv(route_path, index=False)

    print("\n========== PATHSNAP HMM RESULTS ==========")
    print("Input candidate rows:", len(data))
    print("Predicted observations:", len(predictions))
    print("Unique route segments:", len(route))
    print("Predictions saved:", output_path)
    print("Route saved:", route_path)

    print("\nFirst 10 predicted road segments:")
    print(
        predictions[
            ["observation_index", "arc_id", "candidate_score"]
        ].head(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()
