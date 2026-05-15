python3 << 'EOF'
content = '''[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![GitHub stars](https://img.shields.io/github/stars/Awinosarah/acute-malnutrition-early-warning?style=for-the-badge)](https://github.com/Awinosarah/acute-malnutrition-early-warning/stargazers)

# Acute Malnutrition Early Warning System

A district-level machine learning and analytics platform for monitoring, classifying, and forecasting acute malnutrition risk in Uganda.

The system integrates nutrition surveillance data, climate indicators, forecasting models, and geospatial visualizations to support early warning and decision-making for public health programs.


# Project Overview

Acute malnutrition remains a major public health concern in vulnerable populations. Early identification of high-risk districts is critical for preparedness, resource allocation, and timely intervention.

This platform was developed as a prototype decision-support system to:

- Monitor district-level malnutrition trends
- Classify nutrition risk severity
- Forecast future malnutrition burden
- Visualize spatial and temporal patterns
- Support evidence-based planning

The application is built using Streamlit and machine learning techniques for interactive analytics and forecasting.


# Key Features

## 1. District Risk Classification

The system classifies districts into risk categories:

- Normal
- Alert
- Alarm
- Emergency

Risk classification is based on:
- Historical malnutrition burden
- Percentile thresholds
- Trend behaviour
- District-level comparisons


## 2. Forecasting Module

The forecasting module predicts short-term acute malnutrition trends using machine learning.

### Forecasting Features
- 3-month forecasting
- Random Forest Regression
- Trend visualization
- District-level projections

## 3. Machine Learning Classification

The classification component predicts district risk levels using engineered indicators and surveillance trends.

### Classification Features
- Random Forest Classification
- Risk prediction probabilities
- Performance evaluation metrics
- Feature importance analysis


## 4. Geospatial Risk Mapping

Interactive district-level maps are used to visualize nutrition risk patterns across Uganda.

### Mapping Features
- GeoJSON district boundaries
- District comparison views
- Regional distribution analysis

## 5. Trend Monitoring

The application supports longitudinal monitoring of malnutrition patterns.

### Trend Features
- Historical case trends
- Regional comparison charts
- District-level progression analysis
- Risk transition monitoring

## 6. Correlation Analysis

The platform explores relationships between nutrition outcomes and associated indicators.

### Analytics Included
- Spearman correlation analysis
- Radial/radar visualization
- Variable association monitoring
- Comparative indicator analysis

# System Architecture within the DHIS2 ecosystem

The application follows a modular workflow:
![System setup in the DHIS2 eco system](Screenshots/architecture.png)

# Machine Learning Models

## Random Forest Regressor

Used for:
- Forecasting future acute malnutrition cases

Advantages:
- Handles non-linear relationships
- Suitable for complex public health data

## Random Forest Classifier

Used for:
- Predicting district risk levels

Advantages:
- Handles mixed variable types
- Supports feature importance analysis
- Performs well with complex interactions

# Data Sources

The system integrates multiple datasets including:

- Acute malnutrition surveillance data
- Climate indicators
- District metadata
- Historical nutrition trends

# Model Evaluation

The current prototype demonstrates functional workflows for:
- Risk classification
- Forecasting
- Visualization
- Monitoring

### Important Note

This system is intended as:
- An early warning prototype
- A decision-support tool
- A surveillance enhancement platform


# Intended Users

The platform may support:
- Ministry of Health programs
- Nutrition surveillance teams
- Humanitarian organisations
- Public health analysts
- District health offices

The app combines:

- district-based acute malnutrition risk classification;
- within-district and between-district percentile risk thresholds;
- Random Forest regression for 3-month case forecasting;
- Random Forest classification for risk evaluation;
- Spearman correlation analysis using radial visualizations;
- district and region comparison charts;
- historical risk trend monitoring;
- district GeoJSON risk maps.

## App Screens

### 1. Raw Classified Data
After uploading your CSV, the app classifies each district record and displays the full dataset with within-district and between-districts risk labels.

![Raw Classified Data](Screenshots/Screenshot1.png)

### 2. Current Observed Risk
Choropleth maps showing the latest within-district and between-districts risk levels across all Uganda districts.

![Current Observed Risk](Screenshots/Screenshot2.png)


### 3. Historical Risk Trends
Stacked area chart showing how district risk levels have evolved over time, with a 6-month Emergency trend overlay.

![Historical Risk Trends](Screenshots/Screenshot3.png)


### 4. Spearman Correlation Portrait — All Districts
Radial chart showing rank-based correlations between acute malnutrition and all covariates across all districts.

![Spearman Correlation All Districts](Screenshots/Screenshot4.png)

### 5. Spearman Correlation Portrait — Karaeng District
District-level Spearman correlation portrait for Karaeng District, showing local variable associations.

![Spearman Correlation Karaeng District](Screenshots/Screenshot5.png)

### 6. Forecast and 3-Month Risk Change — Within-District
3-month forecast with predicted case counts and risk change map using within-district thresholds.

![Forecast Within-District Risk](Screenshots/Screenshot6.png)


### 7. Forecast and 3-Month Risk Change — Between-Districts
3-month forecast with predicted case counts and risk change map using between-districts thresholds.

![Forecast Between-Districts Risk](Screenshots/Screenshot7.png)


## Data Requirements

The CSV should include:

| Column | Description |
|---|---|
| `Region_District` | Region and district, for example `Acholi|Agago District` |
| `District` | District name |
| `Region` | Region name |
| `time_period` | Monthly date, for example `2020-01` |
| `Acut_Malnutrition` | Acute malnutrition case count |
| `mean_temperature` | Mean temperature |
| `rainfall` | Rainfall |
| `mean_relative_humidity` | Relative humidity |
| `average_gpp` | Vegetation productivity |
| `malaria_confirmed` | Confirmed malaria cases |
| `pneumonia_cases` | Pneumonia cases |
| `pregnant_women_with_Anaemia` | Anaemia cases among pregnant women |
| `diarrhea_acute` | Acute diarrhoea cases |
| `low_birth_weight_babies` | Low birth weight babies |
| `diarrhea_persistent` | Persistent diarrhoea cases |
| `population` | District population |

---

## GeoJSON Requirements

Upload a district GeoJSON in the sidebar. Select the GeoJSON column containing district names. The app matches those names to the CSV district names.

---

## Author

Sarah Awino
'''
with open('/Users/sarahawino/Documents/New project/acute-malnutrition-early-warning/README.md', 'w') as f:
    f.write(content)
print("Done")
EOF
