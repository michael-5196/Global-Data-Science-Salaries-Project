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
![isnull1](https://github.com/michael-5196/Data_Science_Salaries_Project/assets/131683141/345506ce-c358-4d78-9f1b-75f66aa2e9d6)

## Summary Statistics
```
#Collect Summary Stats
df.describe()
```
![summary](https://github.com/michael-5196/Data_Science_Salaries_Project/assets/131683141/00c29560-2523-420f-88b7-b3b774e6e3ac)

### work_year:
#### std: The standard deviation is 0.69 years. This means the data for 'work_year' is very tightly packed around the mean.
#### min & max: The data spans from the year 2020 to 2023.The 25th, 50th (median), and 75th percentiles indicate that most of the data is centered around 2022 and 2023.

### salary_in_usd:
std: The standard deviation is 63,022, which is considerably lower than the standard deviation of the 'salary' column. This indicates less variability in USD salaries compared to the original 'salary' column.
min & max: The range for salaries in USD is from 5,132 to 450,000, which is much narrower than the original 'salary' range.
25%, 50%, and 75%: The interquartile range is between 95,000 and 175,000, indicating that the middle 50% of salaries in USD lie within this range.
### Key Takeaways:
The dataset represents work years from 2020 to 2023 with the majority of data coming from 2022 and 2023.

The 'salary' column has a broad range with some very high values that are likely skewing the mean upwards. It's essential to examine these outliers to determine if they are errors or legitimate high salaries.

When looking at the 'salary_in_usd' column, there's less variability in the salaries. This might indicate that when salaries are normalized to USD, they are more tightly clustered, or it could suggest that the most significant outliers in the 'salary' column are not from the U.S.

Further analysis, such as plotting histograms or box plots, could offer more visual insights into the distribution of these salaries.
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
## Pearson Correlation between Work Year and Salary
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

## Feature Engineering (Calculating Growth Rate to Improve our LSTM Prediction)
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
## We did not calculate Growth Rate for 2021 (we don't have prior data), this should leave is with some nulls. Which is what we are looking for. 
```
data_with_growth_rate_by_year.isnull().sum()
```
## Dropping these null values would be unnecessary, let's impute them with the mean of the growth rate column
```
# Calculate the mean growth rate (excluding NaN values)
mean_growth_rate = data_with_growth_rate_by_year['growth_rate'].mean()

# Fill NaN values in the growth_rate column with the mean
data_with_growth_rate_by_year['growth_rate'].fillna(mean_growth_rate, inplace=True)

data_with_growth_rate_by_year.isnull().sum()
```
##
