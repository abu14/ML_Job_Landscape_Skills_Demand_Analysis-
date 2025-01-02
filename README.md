**ML_Job_Landscape_-_Skills_Demand_Analysis**

# Overview

Hello! Thank you for going through my work. This is an analysis on the job market for Machine Learning roles globally and also give a specical look in to the African job market as well. Focusing on the skills required and salary trends. I worked on this project because I've a strong desire to take on machine learning roles. I wanted to dive deep on what skills are most in demand and top paying. Find out which skills are most optimal for job opportunities for machine learning engineers / data scientists.

The data sourced from [Luke Barousse's Python Course](https://lukebarousse.com/python) which is used for this analysis, containing detailed information on job titles, salaries, locations, and essential skills. Through a series of Python scripts.


# Questions to Answer
1. What are the most demanded skills for the top 3 most popular data roles?
2. How are in-demand skills trending for Machine Learning Engineers?
3. How well do jobs and skills pay for Machine Learning Engineers?
4. What is the most optimal skill to learn for Machine Learning Engineers?


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


### Visualize Data 

Check the job skills most needed for ML jobs globally.

``` python
fig, ax = plt.subplots(len(job_titles), 1, figsize=(12, 8))

for i, job_title in enumerate(job_titles):
    df_plot = df_skills_count[df_skills_count['job_title_short']==job_title].head(5)[::-1]
    sns.barplot(data = df_plot, x='skill_count',y='job_skills',ax=ax[i],palette='dark:b_r')

plt.show()

```

### Results

*Data Skills Globally*

![Top Skills needed for Data Roles](images\likelihood_of_skill_requested.png)


*Data Skills for Africa*

![Top Skills needed for Data Roles Africa](images\likelihood_of_skill_requested_africa.png)




### Insights:

- SQL is the most in-demand skill for Data Analysts and Data Scientists, appearing in more than half of job listings for these positions. For Data Engineers, Python is the top skill, featured in 61% of global job postings and 64% of those in Africa.

- Data Engineers typically need more specialized technical expertise, such as AWS, Azure, and Spark, while Data Analysts and Data Scientists are expected to be skilled in more general data management and analysis tools like Excel and Tableau.

- Python is a highly sought-after skill for all three roles, with the highest demand among Data Scientists (66%) and Data Engineers (61%).



## 2. How are in-demand skills trending for Machine Learning Engineers?

To analyze the skill trends for Machine Learning Engineers in 2023, I focused on filtering job postings specifically for ML positions and organized the skills by the month in which the jobs were posted. This approach allowed me to identify the top five skills for Machine Learning Engineers each month, highlighting the fluctuations in skill popularity throughout the year. The results provide valuable insights into which skills gained traction and which remained consistently in demand over the course of 2023.

View the notebook here [Part 3: Skill Trend Analysis](Part-Three-Exploratory-Skills-Trend-Analysis.ipynb) for detailed info.

### Visualize Data

```python 
from matplotlib.ticker import PercentFormatter

df_plot = df_ml_percent.iloc[:, :5]
sns.lineplot(data=df_plot, dashes=False, legend='full', palette='tab10')

for i in range(5):
    plt.text(11.2, df_plot.iloc[-1, i], df_plot.columns[i], color='black')

plt.show()
```

### Results


*Trend for ML Skills Globally* 

![Top Skills needed for ML Roles](images\trending_top__skills_for_mle.png)


*Trend for ML Skills in Africa*

![Top Skills needed for ML Roles Africa](images\trending_top__skills_for_mle_africa.png)


### Insights:

- Python remains the top skill for Machine Learning Engineers, appearing in 70-80% of job postings, showcasing its versatility for tasks like data preprocessing, model building, deployment, and visualization.

- SQL and R maintain steady demand, with SQL at 50-55% and R at 30-40% in job postings, emphasizing the importance of data management and analysis skills.

- AWS and Tableau show variability: AWS peaks early in the year before dipping in summer, while Tableau's demand fluctuates throughout. AWS's initial demand reflects cloud computing's rise, while its dip may indicate seasonal trends or shifting priorities.



## 3. How well do jobs and skills pay for Machine Learning Engineers?

To identify the highest-paying roles, skills and looked at their median salary. But first I looked at the salary distributions of common data jobs like Data Scientist, Data Engineer, and Data Analyst, to get an idea of which jobs are paid the most. 

View the notebook here [Part 4: Salary Analysis](Part-Four-Salary-Analysis.ipynb) for detailed info.


#### Visualize Data

```python
sns.boxplot(data=df_ml_top6, x='salary_year_avg', y='job_title_short', order=job_order)

ticks_x = plt.FuncFormatter(lambda y, pos: f'${int(y/1000)}K')
plt.gca().xaxis.set_major_formatter(ticks_x)
plt.show()
```

### Results

*Data Jobs Salary Globally*

![Data Jobs Salary Distribution](images\salary_dist_data_jobs.png)


*Data Jobs Salary Africa* 

![Data Jobs Salary Distribution](images\salary_dist_data_jobs_africa.png)

*Box plot visualizing the salary distributions for the top 6 data job titles.*

### Insights:

- Salary Range: Data-related roles offer salaries ranging from under $100K to over $500K annually.

- Senior Roles Earn More: Senior Data Scientists and Senior Data Engineers typically command higher salaries than their junior counterparts.

- Analysts' Median Salary: Analysts both Data and Business Analysts tend to have lower median salaries, with most concentrated in the lower salary range.


### Highest Paid & Most Demanded Skills for Data Analysts

#### Visualize Data

```python
fig, ax = plt.subplots(2, 1,figsize=(12, 8))  

# Top 10 Highest Paid Skills for MLEs
sns.barplot(data=df_ml_top_pay, x='median', y=df_ml_top_pay.index, ax=ax[0], palette='dark:b_r')

# Top 10 Most In-Demand Skills for MLEs' 
sns.barplot(data=df_ml_skills, x='median', y=df_ml_skills.index, ax=ax[1], palette='light:b')

plt.show();

```

### Results

*Data Jobs Salary Globally* 

![Highest Paying Skills for ML](images\highest_paying_skills_mle.png)


*Data Jobs Salary Africa* 

![Highest Paying Skills for ML in Africa](images\highest_paying_skills_mle_africa.png)

*Bar charts for both global and african machine leanring job market. Which summarize the most in-demand and lucrative skills for these respective markets.*

### Insights 

- Python is the most popular tool demanded for Machine Learning Jobs. Tools like TensorFlow and Spark are high in demand. Despite them not always resulting in higher salary.

- Top Paying skills are more related niche skillsets and specializations in specific skill. For example Asana, Airtable, Ruby, and others. Although expertise in these boosts is very likely to increase higher pay, it necessary to note the no of job postings for these skills is very low. Please refer the note: [Part 4: Salary Analysis](Part-Four-Salary-Analysis.ipynb) for more.

- Despite not being among the highest-paid skills, Python remains the most in-demand skill by a significant margin. This highlights its importance as a foundational language for Machine Learning.

- Cloud computing skills (AWS, Azure) and data management tools (SQL) are also in high demand, reflecting their growing significance in the field.


## 4. What are the most optimal skills to learn for Machine Learning Engineers?


To identify the most optimal skills to learn ( the ones that are the highest paid and highest in demand) I calculated the percent of skill demand and the median salary of these skills. To easily identify which are the most optimal skills to learn. 

View the notebook here [Part 5: Optimal Skills Analysis](Part-Five-Optimal-Skills-Analysis.ipynb) for detailed info.

### Visulalize Data

```python

plt.scatter(df_ml_skills_high_demand['skill_percent'], df_ml_skills_high_demand)

# Get current axes, set limits, and format axes
ax = plt.gca()
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, pos: f'${int(y/1000)}K'))  # Example formatting y-axis

# Add labels to points and collect them in a list
texts = []
for i, txt in enumerate(df_ml_skills_high_demand.index):
    texts.append(plt.text(df_ml_skills_high_demand['skill_percent'].iloc[i], df_ml_skills_high_demand['median_salary'].iloc[i], " " + txt))

# Adjust text to avoid overlap and add arrows
adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray'))

plt.show()

```

### Results

*Most Optimal Skills for MLEs* 

![Best Paying Skills for ML](images\most_optimal_skills_for_mle.png)


*Most Optimal Skills for MLEs in Africa* 

![Best Paying Skills for ML in Africa](images\most_optimal_skills_for_mle_africa.png)


*Scatter plot for both global and african machine leanring job market. Which summarize the most optimal and best paying skills for these respective markets.*


### Insights

- Again python being the most demanded skill in job market both gloablly and in Africa as well. With a high percentage of job postings requiring it.

- Skills like PyTorch, Hadoop, & Scala are related to higher salaries. However still not too many jobs mention their need, maintaing our previous notion that niche skills do pay well but not demanded highly.

- Experience in core data science libraries the likes of NumPy, Pandas, and Scikit-learn are increadibly useful. Thus, we can highlight that fundamental knowledge and experience in manipulation of data matters.


### Visualizing Different Techonologies

Let's visualize the different technologies as well in the graph. We'll add color labels based on the technology (e.g., {databases: sql server})

```python
# Scatterplot
plt.figure(figsize=(12, 6))
sns.scatterplot(
    data=df_ml_skills_tech_high_demand,
    x='skill_percent',
    y='median_salary',
    hue='technology'
)

# Prepare texts for adjustment
texts = []  # Initialize an empty list for text objects
for i, skill_name in enumerate(df_ml_skills_high_demand.index):
    x = df_ml_skills_high_demand['skill_percent'].iloc[i]
    y = df_ml_skills_high_demand['median_salary'].iloc[i]
    texts.append(plt.text(x, y, skill_name))  

# Adjust text to avoid overlap
adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray'))

plt.show()

```

### Results

*Most Optimal Skills for MLEs by Technology* 

![Best Paying Skills for ML by Technology](images\most_optimal_skills_by_technology.png)

*A scatter plot visualizing the most optimal skills (high paying & high demand) for machine learning engineers globally with color labels for technology.*

### Insights

- The scatter plot reveals that programming skills (shown in blue) are generally clustered at higher salary levels, suggesting that expertise in programming may lead to better salary prospects in the data analytics field.

- Libraries (ML) tend to have a higher salary offering. Not to mentioon knowledge of Cloud technology seems to give a extreme boost in salary range. 





# Conclusion

Looking at the data skills market, it's pretty interesting - Python is definitely dominating as the go-to skill for data and ML jobs. But here's the cool part: niche skills like Haskell, Solidity, and Ruby on Rails are actually bringing in the highest salaries. Makes sense when you think about it - specialists are hard to find.

Cloud platforms like AWS and Azure, plus SQL, are still essential in the field. While coding skills generally mean better pay, I've noticed it's really about the depth of your expertise and what you specialize in that makes the biggest difference.

The takeaway? Python is your foundation, no doubt. But if you're looking to level up your career, picking up some specialized tools could really set you apart.
