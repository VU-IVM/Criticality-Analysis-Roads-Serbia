"""
Step 3b - Plot the baseline accessibility results.

Loads the facility-specific outputs produced by 3a_Baseline_Accessibility_Analysis.py
(factories, agriculture, firefighters, hospitals, police) and generates all figures:
cumulative access curves, access-time maps, distribution charts, and the combined
fire/police emergency-services figures.

All plotting functions live in utils/accessibility_functions.py so that this script
and the notebooks (notebooks/3a-3e) produce identical figures.
"""

import sys
from pathlib import Path

# Make the repo root importable so the shared utils package resolves.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.accessibility_functions import (
    load_accessibility_results,
    plot_access_times_factories,
    plot_accessibility_curves_agriculture,
    plot_access_time_agriculture_map,
    plot_access_curve,
    plot_access_time_map,
    plot_accessibility_chart,
    plot_emergency_curves_combined,
    plot_emergency_map_combined,
)
from config.network_config import NetworkConfig


def main():
    """Load accessibility results and produce all baseline accessibility figures."""
    config = NetworkConfig()
    parquet_dir = config.local_accessibility_parquet
    figure_dir = config.figure_path
    show = config.show_figures

    # =============================================================================
    # 1. Factories
    # =============================================================================
    print("Loading results of factory accessibility analysis...")
    df_factory_accessibility, df_factory_sinks = load_accessibility_results(
        "factories", parquet_dir
    )
    plot_access_times_factories(
        df_factory_accessibility, df_factory_sinks, figure_dir, show
    )

    # =============================================================================
    # 2. Agricultural areas
    # =============================================================================
    print("Loading results of accessibility analysis of agricultural areas...")
    df_agriculture_accessibility, df_agriculture_sinks = load_accessibility_results(
        "agriculture", parquet_dir
    )
    plot_accessibility_curves_agriculture(
        df_agriculture_accessibility, figure_dir, show
    )
    plot_access_time_agriculture_map(
        df_agriculture_accessibility, df_agriculture_sinks, figure_dir, show
    )

    # =============================================================================
    # 3. Firefighters
    # =============================================================================
    print("Loading results of fire station accessibility analysis...")
    df_firefighter_accessibility, df_fire_stations = load_accessibility_results(
        "firefighters", parquet_dir
    )
    plot_access_curve(df_firefighter_accessibility, "firefighters", figure_dir, show)
    df_fire_plot = plot_access_time_map(
        df_firefighter_accessibility, df_fire_stations, "firefighters", figure_dir, show
    )
    plot_accessibility_chart(df_fire_plot, "firefighters", figure_dir, show)

    # =============================================================================
    # 4. Hospitals
    # =============================================================================
    print("Loading results of hospital accessibility analysis...")
    df_hospital_accessibility, df_hospitals = load_accessibility_results(
        "hospitals", parquet_dir
    )
    plot_access_curve(df_hospital_accessibility, "hospitals", figure_dir, show)
    df_hospital_plot = plot_access_time_map(
        df_hospital_accessibility, df_hospitals, "hospitals", figure_dir, show
    )
    plot_accessibility_chart(df_hospital_plot, "hospitals", figure_dir, show)

    # =============================================================================
    # 5. Police stations
    # =============================================================================
    print("Loading results of police station accessibility analysis...")
    df_police_accessibility, df_police = load_accessibility_results(
        "police", parquet_dir
    )
    plot_access_curve(df_police_accessibility, "police", figure_dir, show)
    df_police_plot = plot_access_time_map(
        df_police_accessibility, df_police, "police", figure_dir, show
    )
    plot_accessibility_chart(df_police_plot, "police", figure_dir, show)

    # =============================================================================
    # 6. Combined emergency-services figures (fire | police)
    # =============================================================================
    plot_emergency_curves_combined(
        df_firefighter_accessibility, df_police_accessibility, figure_dir, show
    )
    plot_emergency_map_combined(
        df_firefighter_accessibility,
        df_fire_stations,
        df_police_accessibility,
        df_police,
        figure_dir,
        show,
    )


if __name__ == "__main__":
    main()
