# Data Science Salaries Project
## LSTM Prediction and Dashboard (Colab & Tableau)
Tools: Python, Tableau, PANDAS, SciPy, Tensorflow Keras, numpy

## General Questions:
How has the average salary for Data Scientists evolved from 2021 to 2023?
Which job titles have seen the largest growth in salaries from 2021 to 2023?
Which country has the highest average salary for Data Analysts?

## Which countries have the most representation in the dataset?
Are there countries where the Data Analyst role is paid significantly higher than the global average?
How do salaries vary by continent or region? For example, North America vs. Europe vs. Asia.


## Predictions:
How do the LSTM model's predictions compare to actual salaries in subsequent years?

Can the model's predictions be refined further by using additional features?

Are there specific job titles or countries that have unusually high or low salaries?

-Create a heatmap showing average salaries by country and job title.

-Box plots to show the distribution of salaries for different job titles.

-Bar graphs to showcase countries with the highest and lowest growth rates.

# Result in Tableau:
![interactive-dash](https://github.com/michael-5196/Data_Science_Salaries_Project/assets/131683141/db105e47-28c1-40ef-a170-43e943a4ca49)
Link = "https://public.tableau.com/app/profile/michael.scott3344/viz/DataScienceSalariesDashboard_16929375239230/Dashboard1"

# How we got there: 

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
## Observations: 

### work_year:
-Standard Deviation: The standard deviation is 0.69 years. This means the data for 'work_year' is very tightly packed around the mean.

-Min & Max: The data spans from the year 2020 to 2023.The 25th, 50th (median), and 75th percentiles indicate that most of the data is centered around 2022 and 2023.

### salary_in_usd:
-Standard Deviation: The standard deviation is 63,022, which is considerably lower than the standard deviation of the 'salary' column. This indicates less variability in USD salaries compared to the original 'salary' column.

-Min & Max: The range for salaries in USD is from 5,132 to 450,000, which is much narrower than the original 'salary' range.
25%, 50%, and 75%: The interquartile range is between 95,000 and 175,000, indicating that the middle 50% of salaries in USD lie within this range.

### Key Takeaways:
-The dataset represents work years from 2020 to 2023 with the majority of data coming from 2022 and 2023.

-The 'salary' column has a broad range with some very high values that are likely skewing the mean upwards. It's essential to examine these outliers to determine if they are errors or legitimate high salaries.

-When looking at the 'salary_in_usd' column, there's less variability in the salaries. This might indicate that when salaries are normalized to USD, they are more tightly clustered, or it could suggest that the most significant outliers in the 'salary' column are not from the U.S.

#### Further analysis could offer more visual insights into the distribution of these salaries.

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


## Average Salary by Title 
```
#Calculate Average Salary by Title

Salary_By_Title = df.groupby('job_title').agg(
    AVG_Salary = ('salary', 'mean'),
)

Salary_By_Title
```
![AVG_Salary_by_title](https://github.com/michael-5196/Data_Science_Salaries_Project/assets/131683141/6f7c1c6d-c1a8-43cf-88a8-f4f97b86d617)

## Sort and Save to CSV for Tableau
```
Salary_By_Title_sorted = Salary_By_Title.sort_values(by= 'AVG_Salary', ascending=False)

Salary_By_Title_sorted.to_csv(r'AVG_Salary_by_Title_Sorted.csv')
```
## Average Data Analyst Salary by Country 
```
# Filter the data for the "Data Analyst" job title
data_analyst_df = df[df['job_title'] == 'Data Analyst']

# Calculate average salary by country for Data Analysts
average_salary_da_by_country = data_analyst_df.groupby('company_location')['salary_in_usd'].mean().reset_index()

average_salary_da_by_country.to_csv(r'average_salary_da_by_country.csv')

```
## Pearson Correlation between Salary in USD and Work Year
If work_year refers to years of experience, does a slight increase in experience result in a slight increase in salary?
Are there other factors or variables in the dataset that might impact salaries? This would help in understanding if other confounding variables might be impacting the correlation.
```
correlation, p_value = stats.pearsonr(df['work_year'], df['salary_in_usd'])

# Convert the result to a DataFrame
result_df = pd.DataFrame({
    'correlation': [correlation],
    'p_value': [p_value]
})

# Save to CSV
result_df.to_csv('pearson_result.csv', index=False)
result_df
```
### There's a weak positive linear relationship between work_year and salary_in_usd. Trends Over Time:
![P_Value](https://github.com/michael-5196/Data_Science_Salaries_Project/assets/131683141/722209d7-0735-4763-bc92-be8464f0de14)
### Questions:
How does the average salary_in_usd change from one work_year to the next? The weak positive correlation suggests that salaries in USD are increasing over the years covered by the dataset. Might be good in the future to look into economic behaviors such as inflation to see the correlation. 
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
## LSTM Salary Prediction (using Data Analyst as an example)
```
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Filter data for a specific job title, e.g., 'Data Analyst'
data_analyst = data_with_growth_rate_by_year[data_with_growth_rate_by_year['job_title'] == 'Data Analyst']

# Normalize the data (salary and growth rate)
max_salary = max(data_analyst['salary_in_usd'])
min_salary = min(data_analyst['salary_in_usd'])
normalized_salary = (data_analyst['salary_in_usd'] - min_salary) / (max_salary - min_salary)
normalized_growth_rate = (data_analyst['growth_rate'] - data_analyst['growth_rate'].min()) / (data_analyst['growth_rate'].max() - data_analyst['growth_rate'].min())

# Create sequences
X, y = [], []
for i in range(len(normalized_salary)-3):
    sequence = list(zip(normalized_salary.iloc[i:i+3], normalized_growth_rate.iloc[i:i+3]))
    X.append(sequence)
    y.append(normalized_salary.iloc[i+3])

X = np.array(X)
y = np.array(y)

# Build LSTM Model
model = Sequential()
model.add(LSTM(50, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X, y, epochs=200, batch_size=1)

# Make predictions for 2024
input_sequence = list(zip(normalized_salary.iloc[-3:], normalized_growth_rate.iloc[-3:]))
input_sequence = np.array([input_sequence])
predicted_normalized_salary = model.predict(input_sequence)
predicted_salary_2024 = predicted_normalized_salary * (max_salary - min_salary) + min_salary

# Save the result
result_df = pd.DataFrame({'year': [2024], 'predicted_salary': [predicted_salary_2024[0][0]]})
result_df.to_csv(r'predicted_salaries.csv', index=False)

print("Predictions saved to 'predicted_salaries.csv'")
```
![Epochs](https://github.com/michael-5196/Data_Science_Salaries_Project/assets/131683141/5b181202-fcb1-4c01-af6b-d371374eae37)
Epochs slowed down as we approached 200 but from the screenshot we can still see that it is learning and we have not overfit anything. 

