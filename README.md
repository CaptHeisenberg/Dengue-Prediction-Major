# Dengue Outbreak Prediction Using Satellite-Derived Environmental Features and Deep Learning

## Abstract

Dengue fever remains a major public health challenge in tropical and subtropical regions, with outbreaks strongly influenced by environmental and climatic factors. This project investigates the feasibility of predicting monthly dengue cases using satellite-derived environmental features and deep learning models. Multispectral imagery from Sentinel-2 is used to capture vegetation, water bodies, and urban characteristics that influence mosquito breeding conditions. Two distinct deep learning methodologies are implemented, evaluated, and compared within the same framework. The study demonstrates how integrating remote sensing data with historical dengue trends can enhance predictive performance in a regression-based forecasting setup.

## Motivation and Problem Statement

Traditional dengue surveillance systems are often reactive, relying on reported cases after outbreaks have already occurred. Early prediction of dengue incidence can enable proactive intervention, optimized resource allocation, and improved public health planning. Since dengue transmission is closely linked to environmental conditions such as vegetation density, surface water, and urbanization, satellite imagery provides a scalable and data-driven alternative for risk assessment.

**Core Objective:** Model the relationship between satellite-derived environmental features and future dengue cases, and analyze how different deep learning architectures affect prediction accuracy.

## Dataset Description

### Sentinel-2 Satellite Data
- Multispectral satellite imagery obtained from Sentinel-2
- Multiple spectral bands capture diverse environmental characteristics
- Processed on a monthly basis to align with dengue case records

### Dengue Case Data
- Monthly dengue case counts for selected cities
- Included directly in the repository
- Each data point corresponds to a specific city and month

### Dataset Access
The processed satellite dataset is hosted on Google Drive:

**[Download Satellite Data](https://drive.google.com/drive/folders/1GtCrOjzrWqbyT1iQBMrAbCWJ7XRgcgwE)**

> **Note:** The repository already contains dengue case CSV files and all result visualizations.

## Methodology

This repository implements two independent methodologies for dengue case prediction.

### Methodology A
- Uses satellite-derived features as direct inputs to a deep learning regression model
- Focuses on learning spatial-environmental patterns correlated with dengue incidence
- Serves as a baseline deep learning approach

### Methodology B
- Extends Methodology A by incorporating temporal windowing of historical inputs
- Uses a fixed multi-month window to model delayed environmental effects on dengue outbreaks
- Designed to better capture temporal dependencies between environmental conditions and disease spread

### Key Differences

| Aspect | Methodology A | Methodology B |
|--------|---------------|---------------|
| Temporal modeling | Limited | Explicit window-based modeling |
| Input structure | Single-time features | Multi-month sequences |
| Predictive focus | Immediate correlations | Delayed environmental impact |

## Feature Engineering

Features are derived from multispectral Sentinel-2 bands and include:
- Vegetation-related indicators (useful for humidity and mosquito habitat estimation)
- Water-related spectral responses (standing water detection)
- Urban surface characteristics
- Aggregated monthly statistics to reduce noise and cloud artifacts

All feature extraction and preprocessing scripts are included in the repository.

## Model Training and Evaluation

### Learning Setup
- **Problem formulation:** Regression
- **Target variable:** Monthly dengue case count
- **Input:** Satellite-derived feature vectors (with temporal windowing for Methodology B)

### Temporal Windowing
A fixed historical window is used to predict dengue cases at a future time step. This enables the model to learn lag effects between environmental changes and dengue incidence.

### Evaluation Metrics
Models are evaluated using standard regression metrics:
- R² Score
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)

## Results and Visualizations

All experimental results are provided as plots within the repository, including:
- Predicted vs. actual dengue case curves
- Training and validation performance trends
- Comparative evaluation of Methodology A and B

These visualizations demonstrate the impact of temporal modeling and feature integration on predictive performance.

## Repository Structure

```
├── Methodology_A/
│   ├── training scripts
│   ├── feature extraction
│   └── evaluation code
│
├── Methodology_B/
│   ├── window-based modeling scripts
│   ├── training and testing pipelines
│   └── evaluation code
│
├── data/
│   ├── dengue_cases.csv
│   └── auxiliary files
│
├── plots/
│   ├── prediction_results/
│   └── evaluation_metrics/
│
├── requirements.txt
└── README.md
```

## Installation and Setup

1. **Clone the repository:**
```bash
git clone <repository-url>
cd dengue-prediction
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Download the satellite dataset** from the Google Drive link and place it in the appropriate data directory.

## How to Run the Code

### Methodology A
```bash
python Methodology_A/train.py
```

### Methodology B
```bash
python Methodology_B/train.py
```

Evaluation scripts will automatically generate prediction plots and metric summaries.

## Conclusion and Future Work

This project validates the potential of satellite imagery combined with deep learning for dengue case prediction. The comparative analysis highlights the importance of temporal modeling when dealing with environmentally driven diseases.

### Future Work
- Incorporating climate variables such as rainfall and temperature
- Extending the approach to weekly forecasting
- Exploring transformer-based spatiotemporal models
- Scaling the system to larger geographic regions

## References

1. Kuo, K.-T., Ong, S. P., & Wong, M. C., *DengueNet: Dengue Prediction using Spatiotemporal Satellite Imagery*, 2020.
2. Li, Y., & Dong, S., *Big Geospatial Data and Data-Driven Methods for Urban Dengue Risk Forecasting*, 2022.
3. ESA Sentinel-2 User Guide.
4. WHO Dengue Guidelines.

## License

This project is licensed under the Apache 2.0 License.
You are free to use, modify, and distribute this software for academic and research purposes, provided that proper credit is given to the original authors.

See the LICENSE file for more details.

## Contributors
Atharva Thombare – Project Lead; Model Development, Experimentation, and Documentation

Ayush Pathak – Model Development, Experimentation, and Documentation

Shobhit Saxena – Model Development, Experimentation, and Documentation

## Acknowledgments
The European Space Agency (ESA) for providing Sentinel-2 satellite imagery.

Google Earth Engine for enabling scalable access and preprocessing of geospatial data.

The authors of prior research on satellite-based dengue prediction for foundational insights.

Academic mentors and reviewers for their guidance and constructive feedback during project evaluation.
