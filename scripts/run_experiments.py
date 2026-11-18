"""
Experimental Evidence Runner for PathSnap Hackathon.
Executes dataset inventory, quality audits, model benchmarks,
generates plots, and outputs interactive Folium HTML map.
"""

import os
import sys
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

import pandas as pd
from src.evaluation.data_quality import DataQualityChecker
from src.evaluation.experiment_tracker import ExperimentTracker
from src.evaluation.plotter import BenchmarkPlotter
from src.evaluation.map_visualizer import TrajectoryMapVisualizer

def main():
    print("=" * 60)
    print("  PATHSNAP HACKATHON - EXPERIMENTAL EVIDENCE GENERATOR")
    print("=" * 60)

    data_dir = os.path.join(BASE_DIR, "data")
    edges_df = pd.read_csv(os.path.join(data_dir, "road_network", "edges.csv"))

    # 1. Dataset Inventory & Data Quality Checks
    print("\n[Step 1/4] Running Dataset Inventory & Data Quality Checks...")
    dq = DataQualityChecker(data_dir=data_dir)
    dq.run_inventory()
    dq.run_checks()

    # 2. Experiment Tracker & Log
    print("\n[Step 2/4] Initializing Shared Experiment Log & Benchmarks...")
    tracker = ExperimentTracker()
    tracker.populate_standard_benchmarks()
    tracker.generate_markdown_summary()

    # 3. Model Comparison Plots
    print("\n[Step 3/4] Generating Benchmark Plots...")
    plotter = BenchmarkPlotter()
    plotter.plot_model_comparison()
    plotter.plot_hyperparameter_tuning()
    plotter.plot_noise_robustness()

    # 4. Interactive Folium Map
    print("\n[Step 4/4] Generating Interactive Folium Map...")
    visualizer = TrajectoryMapVisualizer(edges_df=edges_df)
    visualizer.save_demo_map()

    print("\nAll Experimental Evidence Artifacts Successfully Generated!")
    print(f"Reports: {os.path.join(BASE_DIR, 'reports')}")
    print(f"Plots:   {os.path.join(BASE_DIR, 'experiments', 'plots')}")

if __name__ == "__main__":
    main()
