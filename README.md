**ML_Job_Landscape_-_Skills_Demand_Analysis**

# Overview

Hello! Thank you for going through my work. This is an analysis on the job market for Machine Learning roles globally and also give a specical look in to the African job market as well. Focusing on the skills required and salary trends. I worked on this project because I've a strong desire to take on machine learning roles. I wanted to dive deep on what skills are most in demand and top paying. Find out which skills are most optimal for job opportunities for machine learning engineers / data scientists.

The data sourced from [Luke Barousse's Python Course](https://lukebarousse.com/python) which is used for this analysis, containing detailed information on job titles, salaries, locations, and essential skills. Through a series of Python scripts.


# Questions to Answer
1. What are the most demanded skills for the top 3 most popular data roles?
2. How are in-demand skills trending for Machine Learning Engineers?
3. How well do jobs and skills pay for Machine Learning Engineers and Data Scientists?
4. What is the most optimal skill to learn for Machine Learning Engineers and Data Scientists?


# Tools Used

- **Python:** allows me to analyze the data and find insights and patterns. Made use of the following Python libraries:
    - **Pandas Library:** To analyze the data. 
    - **Matplotlib Library:** To visualized the data.
    - **Seaborn Library:** Create more advanced visuals. 
- **Jupyter Notebooks:** To put down my ideas and Python scripts as well as include my analysis ofcourse.
- **Visual Studio Code:** Executing my Python scripts.
- **Git & GitHub:** For version control and sharing my Python code and analysis.


# Data Preparation and Cleanup

This section outlines the steps taken to prepare the data for analysis, ensuring accuracy and usability.


## Import & Clean Up Data

I start by importing necessary libraries and loading the dataset, followed by initial data cleaning tasks to ensure data quality.


```python
# Importing Libraries
import ast
import pandas as pd
import seaborn as sns
from datasets import load_dataset
import matplotlib.pyplot as plt  

# Loading Data
dataset = load_dataset('lukebarousse/data_jobs')
df = dataset['train'].to_pandas()

# Data Cleanup
df['job_posted_date'] = pd.to_datetime(df['job_posted_date'])
df['job_skills'] = df['job_skills'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else x)
```



## Filter ML Jobs & Also Africa

To focus my analysis on the U.S. job market, I apply filters to the dataset, narrowing down to roles based in the United States.


```python
# Step 1: extract country list and continent
def get_continent(country):
    try:
        # Use the RestCountries API to fetch country information
        response = requests.get(f'https://restcountries.com/v3.1/name/{country}')
        data = response.json()
        return data[0]['continents'][0]  
    except Exception as e:
        return 'Unknown'
# Apply the function to the job_country column
df_country['continent'] = df_country['Country'].apply(get_continent)

# Step 2: 
df_merged = df.merge(df_country, how='left',left_on='job_country',right_on='Country')
df_merged.head(3)


## Step 3: Select data frame for both ML globally as well as Africa
#Globally
df_ml = df_merged[df_merged['job_title'].isin(['Data Scientist','Machine Learning Engineer'])].copy()

#Africa
df_ml_africa = df_merged[(df_merged['job_title'].isin(['Data Scientist','Machine Learning Engineer'])) & (df_merged['continent'] == 'Africa')].copy()
```


# The Analysis

Each Jupyter notebook for this project aimed at investigating specific aspects of the data job market. Here’s how I approached each question:

## 1. What are the most demanded skills for the top 3 most popular data roles?

To find the most demanded skills for the top 3 most popular data roles. I filtered out those positions by which ones were the most popular, and got the top 5 skills for these top 3 roles. This query highlights the most popular job titles and their top skills, showing which skills I should pay attention to depending on the role I'm targeting. 


View the notebook here [Part 2: Skill Demand Analysis](Part-Two-Skill-Demand-Analysis.ipynb) for detailed info.


# Visualize Data 

Check the job skills most needed for ML jobs globally.

``` python
fig, ax = plt.subplots(len(job_titles), 1, figsize=(12, 8))

for i, job_title in enumerate(job_titles):
    df_plot = df_skills_count[df_skills_count['job_title_short']==job_title].head(5)[::-1]
    sns.barplot(data = df_plot, x='skill_count',y='job_skills',ax=ax[i],palette='dark:b_r')

plt.show()

```

### Results
<!-- 
> Data Skills Globally
![Top Skills needed for Data Roles](images\job_skills_demand_for_roles.png)


> Data Skills for Africa
![Top Skills needed for Data Roles Africa](images\job_skills_demand_for_roles_africa.png) -->



> Data Skills Globally and for Africa

<p align="center">
  <img src="images\likelihood_of_skill_requested.png" alt="Top Skills needed for Data Roles" width="45%">
  <img src="images\likelihood_of_skill_requested_africa.png" alt="Top Skills needed for Data Roles Africa" width="45%">
</p>


### Insights:

- SQL is the most in-demand skill for Data Analysts and Data Scientists, appearing in more than half of job listings for these positions. For Data Engineers, Python is the top skill, featured in 61% of global job postings and 64% of those in Africa.
- Data Engineers typically need more specialized technical expertise, such as AWS, Azure, and Spark, while Data Analysts and Data Scientists are expected to be skilled in more general data management and analysis tools like Excel and Tableau.
- Python is a highly sought-after skill for all three roles, with the highest demand among Data Scientists (66%) and Data Engineers (61%).



## 2. How are in-demand skills trending for Machine Learning Engineers?

To analyze the skill trends for Machine Learning Engineers in 2023, I focused on filtering job postings specifically for ML positions and organized the skills by the month in which the jobs were posted. This approach allowed me to identify the top five skills for Machine Learning Engineers each month, highlighting the fluctuations in skill popularity throughout the year. The results provide valuable insights into which skills gained traction and which remained consistently in demand over the course of 2023.

# Visualize Data

```python 
from matplotlib.ticker import PercentFormatter

df_plot = df_ml_percent.iloc[:, :5]
sns.lineplot(data=df_plot, dashes=False, legend='full', palette='tab10')

for i in range(5):
    plt.text(11.2, df_plot.iloc[-1, i], df_plot.columns[i], color='black')

plt.show()
```

# Results

> Trending Top Skills Globally and for Africa

<p align="center">
  <img src="images\trending_top__skills_for_mle.png" alt="Top Skills needed for Data Roles" width="45%">
  <img src="images\trending_top__skills_for_mle_africa.png" alt="Top Skills needed for Data Roles Africa" width="45%">
</p>




> Data Skills Globally

![Top Skills needed for Data Roles](images\job_skills_demand_for_roles.png)


> Data Skills for Africa
![Top Skills needed for Data Roles Africa](images\job_skills_demand_for_roles_africa.png)