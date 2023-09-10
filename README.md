[![License](https://img.shields.io/badge/License-MIT-red.svg)](https://github.com/zhangqi0210/Crime_in_Chicago/blob/main/LICENSE)
![Python 3.x](https://img.shields.io/badge/python-3.x-blue.svg)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fzhangqi0210%2FCrime_in_Chicago&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

# Crime_in_Chicago
The project titled "Crime in Chicago" is an in-depth data analysis initiative.
## Table of Contents
1. [Introduction](#introduction)
2. [Project Members](#project-members)
3. [Objectives](#objectives)
4. [Dependencies](#dependencies)
5. [Data Preprocessing](#data-preprocessing)
6. [Analysis Techniques](#analysis-techniques)
7. [Technical Deep Dive](#technical-deep-dive)
8. [How to Run the Project](#how-to-run-the-project)
9. [License](#license)

---

## Introduction

The project titled "Crime in Chicago" is an in-depth data analysis initiative. Initially, it was designed to assess the crime landscape in Chicago for a client concerned about safety. The project has been expanded upon to include time-series analysis, inter-variable relationships, and forecasting techniques. It specifically focuses on three types of crimes: theft, battery, and criminal damage, which have the highest occurrences in Chicago.

---


## Objectives

The overarching objectives of this analysis are three-fold:

1. To determine the temporal trends in crime rates, particularly whether crime is decreasing over time.
2. To analyze the relationship between different types of crimes and their corresponding arrest rates.
3. To employ data-driven techniques to forecast the projected number of theft incidents for the year 2018.

---

## Dependencies

The project makes use of several Python libraries for data manipulation, analysis, and visualization. Make sure to install the following dependencies before running the project:

- Pandas
- Matplotlib
- Seaborn
- NumPy
- scikit-learn

You can install these packages using pip:

\```bash
pip install pandas matplotlib seaborn numpy scikit-learn
\```

---

## Data Preprocessing

Given the size of the dataset, the initial phase involved a rigorous data cleaning process. Columns that were not essential for the analysis were removed to optimize performance. This also resulted in a more concise and manageable dataset.

---

## Analysis Techniques

The project employs a variety of techniques for a rigorous analysis:

1. **Time-Series Analysis**: To understand the crime rates over a timeline.
2. **Correlation Metrics**: To explore the relationship between different types of crimes and arrest rates.
3. **Forecasting**: To predict future crime rates using machine learning techniques.

---

## Technical Deep Dive

### Data Analysis

#### Time-Series Analysis

We used advanced time-series methods like ARIMA (AutoRegressive Integrated Moving Average) to analyze crime trends over time. The time-series analysis allows us to answer questions like:

- Is crime increasing or decreasing over time?
- Are there seasonal patterns in crime rates?

The ARIMA model was fine-tuned to find the best-fit model, which was then used for forecasting future crime rates.

\```python
from statsmodels.tsa.arima_model import ARIMA

# Fit the ARIMA model
model = ARIMA(time_series_data, order=(1,1,1))
model_fit = model.fit(disp=0)

# Forecast
forecast = model_fit.forecast(steps=10)
\```

#### Correlation Metrics

To explore the relationship between different variables, we computed Pearson's correlation coefficients and visualized them using heatmaps.

\```python
import seaborn as sns

# Compute correlation matrix
correlation_matrix = df.corr()

# Generate a heatmap
sns.heatmap(correlation_matrix, annot=True)
\```

### Machine Learning for Forecasting

We employed machine learning techniques like Random Forest and XGBoost for more accurate crime rate predictions. Feature importance was evaluated to understand which variables contribute most to the crime rate.

\```python
from sklearn.ensemble import RandomForestRegressor

# Initialize and fit the model
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
\```

### Data Preprocessing

Data preprocessing was handled using Pandas. Missing values were imputed, and categorical variables were encoded.

\```python
import pandas as pd

# Drop unnecessary columns
df.drop(columns=['Unused_Column'], inplace=True)

# Impute missing values
df.fillna(method='ffill', inplace=True)

# Encode categorical variables
df = pd.get_dummies(df, columns=['Category'])
\```

---

## How to Run the Project

1. Clone the repository to your local machine.
2. Navigate to the project directory.
3. Install all the dependencies mentioned in the [Dependencies](#dependencies) section.
4. Run the Jupyter Notebook titled "Final Project 2 Notebook.ipynb".

\```bash
git clone [repository_link]
cd [project_directory]
pip install -r requirements.txt
jupyter notebook "Final Project 2 Notebook.ipynb"
\```

---
