# Data Science Salaries Project
## LSTM Prediction and Dashboard (Colab & Tableau)
Tools: Python, Tableau, PANDAS, SciPy, Tensorflow Keras, numpy

## Install Libraries
```#install necessary libraries 
pip install pandas scipy numpy tensorflow keras
```

## Read Data in and search for Null Values
```
#Read data in and look for Null Values
import pandas as pd
import scipy.stats as stats

df = pd.read_csv('Data Science Salary 2021 to 2023.csv')

df.isnull().sum()
```

## Summary Statistics
```
#Collect Summary Stats
df.describe()
```
## Calculating Average Salary based on Title
```
#Calculate Average Salary by Title

Salary_By_Title = df.groupby('job_title').agg(
    AVG_Salary = ('salary', 'mean'),
)

Salary_By_Title
```
## Reordering Average Salary By Title (Descending)
```
Salary_By_Title_sorted = Salary_By_Title.sort_values(by= 'AVG_Salary', ascending=False)

Salary_By_Title_sorted.to_csv(r'AVG_Salary_by_Title_Sorted.csv')
```
## Pearson Correlation between Work Year and Salary using 
```
correlation, p_value = stats.pearsonr(df['work_year'], df['salary_in_usd'])

# Convert the result to a DataFrame
result_df = pd.DataFrame({
    'correlation': [correlation],
    'p_value': [p_value]
})

# Save to CSV
result_df.to_csv('pearson_result.csv', index=False)

```

## Feature Engineering (Calculating Growth Rate to Improve our Prediction)
```
# Calculate the average salary for each job title and work year
avg_salary_by_year = df.groupby(['work_year', 'job_title'])['salary_in_usd'].mean().reset_index()

# Compute the yearly growth rate
avg_salary_by_year.sort_values(by=['job_title', 'work_year'], inplace=True)
avg_salary_by_year['growth_rate'] = avg_salary_by_year.groupby('job_title')['salary_in_usd'].pct_change()

# Merge the growth rate back into the original DataFrame
data_with_growth_rate_by_year = df.merge(avg_salary_by_year[['work_year', 'job_title', 'growth_rate']], on=['work_year', 'job_title'], how='left')

data_with_growth_rate_by_year.to_csv(r'data_with_growth_rate_by_year.csv')
```
