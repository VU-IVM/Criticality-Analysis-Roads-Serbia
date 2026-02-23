# Road Network Criticality & Accessibility Analysis
A collection of Python scripts and Jupyter Notebooks designed to assess the **criticality** of road network segments and evaluate **population accessibility** to emergency services.  
Originally developed for the **Republic of Serbia**, the workflow is fully transferable to other national road networks.

---

## Overview

This repository provides a comprehensive workflow to:

1. **Prepare and process road networks**  
2. **Assess the criticality** of each road segment by quantifying the impact of its disruption:  
   - Vehicle‑hours lost  
   - Person‑hours lost  
   - Additional kilometres travelled  
3. **Analyse accessibility** of population clusters to:  
   - Fire stations  
   - Hospitals  
   - Police stations  
   And the access times of industrial and agricultural areas to road borders, ports and rail terminals
4. **Evaluate hazard exposure**, including:  
   - Flooding, landslides and snowdrifts under current climatic conditions 
   - Flooding and heavy precipitation under climate change  
5. **Calculate combined climate–criticality metric** that considers the previously evaluated hazard exposure, national-scale travel disruptions and local accessibiliy

---

## Workflow Structure

The analysis is implemented through a series of Jupyter Notebooks and corresponding Python scripts.  
The workflow follows this approximate sequence:

| Step | Description | Notebooks | Scripts |
|------|-------------|-----------|---------|
| **1. Network Preparation** | Load, simplify, and preprocess the national road network | `1a_NetworkFigures.ipynb`<br>`1b_NetworkPreparation.ipynb` | `1a_NetworkFigures.py`<br>`1b_NetworkPreparation.py` |
| **2. Criticality Analysis** | Compute disruption impact of each road segment | `2_MainNetwork_CriticalityAnalysis.ipynb` | `2_MainNetwork_CriticalityAnalysis.py` |
| **3. Accessibility Analysis** | Assess travel time of population clusters to facilities |`3a_Baseline_Accesibility_Analysis-factories.ipynb`<br>`3b_Baseline_Accesibility_Analysis-farms.ipynb`<br>`3c_Baseline_Accesibility_Analysis-firefighters.ipynb`<br>`3d_Baseline_Accesibility_Analysis-hospital.ipynb`<br>`3e_Baseline_Accesibility_Analysis-policestations.ipynb` | `3a_Baseline_Accesibility_Analysis.py`<br>`3b_plot_figures.py` |
| **4. Hazard Mapping** | Generate hazard layers (baseline + climate change) | `4a_Hazard_maps.ipynb`<br>`4b_Hazard_Maps_Climate_Change.ipynb` | `4a_Hazard_maps.py`<br>`4b_Hazard_maps_climate_change.py` |
| **5. Combined Risk Analysis** | Hazard‑informed network criticality and accessibility | `5a_MainNetwork_Hazard_Criticality.ipynb`<br>`5b_Flood_Scenarios_Accessibility.ipynb`<br>`5c_CombinedClimateCriticality.ipynb` notebooks | `5a_MainNetwork_Hazard_Criticality.py`<br>`5b_Flood_Scenarios_Accessibility.py`<br>`5c_CombinedClimateCriticality.py` |

---

## Repository Structure

```plaintext
criticality-analysis/
├── notebooks/                     # Jupyter Notebooks for full analysis workflow
└── src/                           # Core Python scripts for network, hazard & accessibility modelling
    └── config/                    # Configuration folder
        └── network_config.py      # Main configuration file (paths & settings)

```
---

## Installation

```bash
git clone <this-repository>
cd <repository-folder>
# Create the environment from environment.yml
conda env create -f environment.yml
# Activate the environment
conda activate <environment-name>
```
---

## Data
The input data (hazard maps, road networks, exposure layers, etc.) is **not included** in this repository.

---
## Authors
- **Elco Koks**
- **Joël De Plaen**
- **Valentin Weiwad**
---
## License

This project is licensed under the terms of the [License](LICENSE) file.
