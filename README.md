# Neuroscience Research Analysis Repository

This repository contains analysis code and data for a neuroscience research project investigating behavioral and electrophysiological data from human subjects performing decision-making tasks.

## Repository Structure

### 📁 `behEphys/`
Contains behavioral and electrophysiology data for multiple subjects (057-070).

#### Data Organization:
- **`MNIelectrodePositions.xlsx`**: Electrode position information in MNI coordinates
- **Subject folders (`subj_057/` to `subj_070/`)**:
  - `subj_XXX_mcsdBeh_*.mat`: Behavioral data files with timestamps
  - `subj_XXX_mcsdEphys.mat`: Raw electrophysiology data
  - `subj_XXX_mcsdEphysNlx.mat`: Processed electrophysiology data (Neuralynx format)
  - `subj_XXX_mcsdLoc.mat`: Electrode location data
  - `subj_XXX_mcsdPract_*.mat`: Practice session data
  - `subj_XXX_mcsdSpikes.mat`: Spike data
  - `ephysFigs/`: Individual subject analysis figures
    - `obsVars/`: Observable variables analysis plots (PDFs)
    - `appendFigs/`: Supplementary analysis figures

### 📁 `mdl_analyses/`
Model-based analysis scripts and results for neural encoding.

#### Key Files:
- **`goal_context.py`**: Aggregates goal context encoding data across subjects
- **`goal_context_subj.py`**: Subject-level goal context analysis
- **`regression_values_mcsd.py`**: Regression analysis for neural encoding
- **`regression_values_subj_mcsd.py`**: Subject-specific regression analysis
- **`utils.py`**: Utility functions for model analyses

#### Generated Results:
- **Brain region analysis**: Pie charts showing best encoded variables for each region:
  - `A_best_encoded_variables_pie_chart.png` (Amygdala)
  - `ACC_best_encoded_variables_pie_chart.png` (Anterior Cingulate Cortex)
  - `CM_best_encoded_variables_pie_chart.png` (Centromedian)
  - `H_best_encoded_variables_pie_chart.png` (Hippocampus)
  - `IFG_best_encoded_variables_pie_chart.png` (Inferior Frontal Gyrus)
  - `OF_best_encoded_variables_pie_chart.png` (Orbitofrontal)
  - `PRV_best_encoded_variables_pie_chart.png` (Perirhinal/Parahippocampal)
  - `PT_best_encoded_variables_pie_chart.png` (Posterior Temporal)
  - `SMA_best_encoded_variables_pie_chart.png` (Supplementary Motor Area)

- **Encoding analysis**:
  - `currSubgoal_encoding_percentages.png`: Current subgoal encoding across regions
  - `goalAmt_encoding_percentages.png`: Goal amount encoding across regions
  - `behavioral_variables_correlation_heatmap.png`: Correlation matrix of behavioral variables

- **`coefficient_difference_heatmaps/`**: Statistical difference maps between conditions for each brain region
- **`heat maps/`**: Correlation heatmaps for regression coefficients by brain region
- **`neural_encoding_results/`**: Individual subject neural encoding result plots

### 📁 `decoding analyses/`
Machine learning decoding analysis of neural signals.

#### Scripts:
- **`decode_option_mcsd.py`**: Decodes choice options from neural activity
- **`decode_second_choice_mcsd.py`**: Decodes secondary choice behavior
- **`test_option_mcsd.py`**: Testing script for option decoding
- **`test_second_choice_mcsd.py`**: Testing script for second choice decoding
- **`plot_decoding_mcsd.py`**: Visualization of decoding results
- **`utils.py`**: Utility functions for decoding analyses

#### Results:
- **`graphs/`**:
  - `decoding_option.png`: Option decoding performance visualization
  - `secondaction_decoding.png`: Second action decoding results

### 📊 Analysis Scripts (Root Directory)
- **`option_ccgp_mcsd.py`**: Option analysis using canonical correlation/GP methods
- **`pca_options_mcsd.py`**: Principal component analysis of option-related neural activity
- **`dpca_options_mcsd.py`**: Demixed principal component analysis for options
- **`regression_options_mcsd.py`**: Regression analysis for option encoding
- **`regression_options_subj_mcsd.py`**: Subject-specific option regression analysis

### 📈 Data Files (Root Directory)
- **`det_hmbOforgetFixedEtaBetaExplorBias_behEphys.mat`**: Model parameters and behavioral data
- **`behavioral_variables_correlation_heatmap.png`**: Overview correlation analysis

## Research Overview

This project appears to investigate:
1. **Decision-making behavior** in a multi-choice task environment
2. **Neural encoding** of task-relevant variables across multiple brain regions
3. **Goal-directed behavior** and subgoal processing
4. **Machine learning decoding** of behavioral intentions from neural signals

## Key Brain Regions Studied
- **ACC**: Anterior Cingulate Cortex
- **A**: Amygdala  
- **CM**: Centromedian nucleus
- **H**: Hippocampus
- **IFG**: Inferior Frontal Gyrus
- **OF/OFC**: Orbitofrontal Cortex
- **PRV**: Perirhinal/Parahippocampal region
- **PT**: Posterior Temporal cortex
- **SMA**: Supplementary Motor Area

## Data Collection Period
Subject data spans from April 2023 to July 2025, with 14 subjects total (subjects 057-070).

## File Formats
- **`.mat`**: MATLAB data files containing neural and behavioral data
- **`.py`**: Python analysis scripts
- **`.png`**: Generated figure outputs
- **`.pdf`**: Individual analysis reports and figures

## Dependencies
Based on the code structure, this project likely requires:
- Python 3.x
- NumPy, SciPy, Pandas
- Matplotlib, Seaborn (for plotting)
- scikit-learn (for machine learning)
- statsmodels (for statistical analysis)
- MATLAB data file support

## Usage
Each analysis script is designed to be run independently, with utility functions provided in respective `utils.py` files. The modular structure allows for both individual subject analysis and group-level statistical comparisons.

---

*This repository represents ongoing neuroscience research into the neural mechanisms of decision-making and goal-directed behavior in humans.*
