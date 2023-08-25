# Data Science Salaries Project
## LSTM Prediction and Dashboard
Tools: Python, Tableau, PANDAS, SciPy, Tensorflow Keras, numpy

## Install Libraries
```#install necessary libraries 
pip install pandas scipy numpy tensorflow keras
```

## Read Data in and search for Null Values
```
#Read data in and look for Null Values
import pandas as pd
import altair as alt
import scipy.stats as stats

df = pd.read_csv('Data Science Salary 2021 to 2023.csv')

df.isnull().sum()
```
