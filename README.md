# Financial-Fraud
# Random Forest Model on Fraud detection

## Project Overview

This project delves into transaction data 


## Dataset Overview

Dataset provided (`../data/raw/PS_20174392719_1491204439457_log.csv`)

## Target Attribute

isFraud is the target attribute.

## Goal
Train a model to predict financialfraud

## Project Workflow

### Part 1: Exploratory Data Analysis
#### Load data

```python
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('../data/raw/PS_20174392719_1491204439457_log.csv')
```
#### Visualizing data

![Alt text](image-1.png)

![Alt text](image-2.png)

### Fraudulent transactions were only found in Cash-out and Transfer transaction types
![Alt text](image-4.png)

### Data from a random sample of 50000 transactions

![Alt text](image-3.png)


ex. Two examples of the many charts built in 01eda.ipynb

### Part 2: Data Transformation
- Clean and transform the dataset to answer primary questions
- Check for null values, drop columns, drop rows
- Save new DataFrame to "data/processed" folder in project 


### Part 3: Build Model
#### Load data
```python 
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats

math = pd.read_csv('../data/processed/cleaned0student-mat.csv')
```
#### Build 
- Scale data
- Split
- Model


## Conclusion and Further Steps


### Insights
These images depict the top 10 features with highest importance in the model. Indicating that these features are most responsible for prediciting.

![Alt text](image.png)


### Challenges

- How to deal with outliers

### Further steps

