# Film Data Analysis Project

## Introduction
This project analyzes various aspects of movies, including their runtime, ratings, number of votes, production budgets, and gross revenues. The analysis is performed using a dataset containing 4,380 movies with various attributes.

## Data Overview
The dataset contains the following columns:
- `movie_title`: Title of the movie.
- `production_date`: Release date of the movie.
- `genres`: Genres of the movie.
- `runtime_minutes`: Duration of the movie in minutes.
- `director_name`: Name of the director.
- `director_professions`: Professions of the director.
- `director_birthYear`: Birth year of the director.
- `director_deathYear`: Death year of the director (if applicable).
- `movie_averageRating`: Average rating of the movie.
- `movie_numerOfVotes`: Number of votes the movie received.
- `approval_Index`: Approval index of the movie.
- `Production budget $`: Production budget of the movie in dollars.
- `Domestic gross $`: Domestic gross revenue in dollars.
- `Worldwide gross $`: Worldwide gross revenue in dollars.

## Data Exploration
First, we load the dataset and check for any missing values or duplicates.

```python
import pandas as pd

# Load the dataset
filmData = pd.read_csv('FilmProject\\Data\\movie_statistic_dataset.csv')

# Display basic information about the dataset
print(filmData.info())
print(filmData.describe())

# Check for duplicates
duplicates = filmData.duplicated()
print("Number of duplicates:", sum(duplicates)) # Number of duplicates:0
```
![data describe](https://github.com/Bozzojr/FilmDataAnalysis_Python/assets/123130175/2ce34598-91b0-4925-9be3-316a06e7bac6)

Our data set has 4,380 non-null values for each column. There are also 0 duplicates.
At first glance, our data appears to be consistent and reliable.

For potential statistical analysis in the future, it would be helpful to know the distribution of the numerical variables. 
We'll visualize using boxplot and histograms

```python
# Set up the figure and axes for subplots
fig, axs = plt.subplots(5, 2, figsize=(15, 25))

# Runtime
axs[0, 0].boxplot(filmData['runtime_minutes'], vert=False, patch_artist=True)
axs[0, 0].set_title('Boxplot of Runtime')
axs[0, 0].set_xlabel('Runtime (Minutes)')

axs[0, 1].hist(filmData['runtime_minutes'], bins=60, color='red', alpha=0.7)
axs[0, 1].set_title('Distribution of Runtime')
axs[0, 1].set_xlabel('Runtime (Minutes)')
axs[0, 1].set_ylabel('Frequency')

# Average Rating
axs[1, 0].boxplot(filmData['movie_averageRating'], vert=False, patch_artist=True)
axs[1, 0].set_title('Boxplot of Ratings')
axs[1, 0].set_xlabel('Ratings')

axs[1, 1].hist(filmData['movie_averageRating'], bins=80, color='tomato', alpha=0.7)
axs[1, 1].set_title('Distribution of Rating')
axs[1, 1].set_xlabel('Ratings')
axs[1, 1].set_ylabel('Frequency')

# Number of Votes
axs[2, 0].boxplot(filmData['movie_numerOfVotes'], vert=False, patch_artist=True)
axs[2, 0].set_title('Boxplot of Votes')
axs[2, 0].set_xlabel('Number of Votes')

axs[2, 1].hist(filmData['movie_numerOfVotes'], bins=100, color='blue', alpha=0.7)
axs[2, 1].set_title('Distribution of Votes')
axs[2, 1].set_xlabel('Number of Votes')
axs[2, 1].set_ylabel('Frequency')

# Approval Index
axs[3, 0].boxplot(filmData['approval_Index'], vert=False, patch_artist=True)
axs[3, 0].set_title('Boxplot of Approval Index')
axs[3, 0].set_xlabel('Approval Index')

axs[3, 1].hist(filmData['approval_Index'], bins=100, color='green', alpha=0.7)
axs[3, 1].set_title('Distribution of Approval Index')
axs[3, 1].set_xlabel('Approval Index')
axs[3, 1].set_ylabel('Frequency')

# Production Budget
axs[4, 0].boxplot(filmData['Production budget $'], vert=False, patch_artist=True)
axs[4, 0].set_title('Boxplot of Production Budget')
axs[4, 0].set_xlabel('Production Budget ($)')

axs[4, 1].hist(filmData['Production budget $'], bins=100, color='purple', alpha=0.7)
axs[4, 1].set_title('Distribution of Production Budget')
axs[4, 1].set_xlabel('Production Budget ($)')
axs[4, 1].set_ylabel('Frequency')

# Adjust layout
plt.subplots_adjust(left=0.1, bottom=0.05, right=0.9, top=0.95, wspace=0.1, hspace=0.8)
plt.show()
```

![boxplot_histograms](https://github.com/Bozzojr/FilmDataAnalysis_Python/assets/123130175/95e34f61-6063-46f3-82c8-15736e97794b)

Runtime are slightly skewed right, and Votes & Production budget are clearly skewed right. Ratings and Approval Index seem to have a normal distribution. 

## Genre Analysis
Let run some analysis based on Genre
First, let's see what genres are the most popular in this data set
```python
filmData['genre_list'] = filmData['genres'].str.split(',')
exploded_genres = filmData.explode('genre_list')
genre_counts = exploded_genres['genre_list'].value_counts()
```
Wow it looks like our top genres are Drama, Comedy, Action, Adventure, and Crime

Here is some additional Genre Analysis, saved into a data frame
```python
# What Genres makes the most money on average
average_gross_by_genre = exploded_genres.groupby('genre_list')['Worldwide gross $'].mean().sort_values(ascending=False)
print(average_gross_by_genre)
# Animation wouldn't have been my first thought on top grossing genre
# But what Genre costs the most to produce?
average_cost_by_genre = exploded_genres.groupby('genre_list')['Production budget $'].mean().sort_values(ascending = False)
print(average_cost_by_genre)
# Interesting to see animation also has the highest production budget, makes sense
# Might as well make a table with all the genre data we can get
average_runtime_by_genre = exploded_genres.groupby('genre_list')['runtime_minutes'].mean().sort_values(ascending = False)
average_rating_by_genre = exploded_genres.groupby('genre_list')['movie_averageRating'].mean().sort_values(ascending = False)
average_votes_by_genre = exploded_genres.groupby('genre_list')['movie_numerOfVotes'].mean().sort_values(ascending = False)
average_approval_by_genre = exploded_genres.groupby('genre_list')['approval_Index'].mean().sort_values(ascending = False)
average_domestic_gross_by_genre = exploded_genres.groupby('genre_list')['Domestic gross $'].mean().sort_values(ascending = False)
max_gross_by_genre = exploded_genres.groupby('genre_list')['Worldwide gross $'].max().sort_values(ascending = False)
min_gross_by_genre = exploded_genres.groupby('genre_list')['Worldwide gross $'].min().sort_values(ascending = False)
max_budget_by_genre = exploded_genres.groupby('genre_list')['Production budget $'].max().sort_values(ascending = False)
min_budget_by_genre = exploded_genres.groupby('genre_list')['Production budget $'].min().sort_values(ascending = False)


genre_data = pd.DataFrame({
    'Average_Runtime': average_runtime_by_genre,
    'Average_Rating': average_rating_by_genre,
    'Average_Number_of_Votes': average_votes_by_genre,
    'Average_Approval': average_approval_by_genre,
    'Min_Product_Budget_($)': min_budget_by_genre,
    'Max_Product_Budget_($)': max_budget_by_genre,
    'Average_Product_Budget_($)': average_cost_by_genre,
    'Min_Worldwide_Gross_($)': min_gross_by_genre,
    'Max_Worldwide_Gross_($)': max_gross_by_genre,
    'Average_Worldwide_Gross_($)': average_gross_by_genre,
})
print(genre_data)
```

## Production Spend
Has production spend been increasing over time?
```python
filmData['production_date'] = pd.to_datetime(filmData['production_date'])
filmData['year'] = filmData['production_date'].dt.year
annual_budget = filmData.groupby('year')['Production budget $'].mean()

#Plot data
plt.figure(figsize=(12, 6))
plt.plot(annual_budget.index, annual_budget, marker='o', linestyle='-', color='b')
plt.title('Trend of Production Budgets Over Time')
plt.xlabel('Year')
plt.ylabel('Average Production Budget ($)')
plt.grid(True)
plt.show()
```
![ProductionBudget_Over_Time](https://github.com/Bozzojr/FilmDataAnalysis_Python/assets/123130175/7ea71f6e-416c-4de6-8444-ff7043a06bf6)

## Correlation Analysis and Heatmap
We'll start by calculating the correlations for the numerical variables in our data, and creating a heatmap from it

```python
import seaborn as sns
# Calculate the correlation matrix for numerical variables
correlation_matrix = filmData[['runtime_minutes', 'movie_averageRating', 'movie_numerOfVotes', 'approval_Index', 'Production budget $', 'Domestic gross $', 'Worldwide gross $']].corr()

# Create a heatmap using seaborn
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title('Heatmap of Correlations Among Numerical Variables')
plt.show()
```
![correlation_heatmap](https://github.com/Bozzojr/FilmDataAnalysis_Python/assets/123130175/504d07c4-2bba-4baf-83d8-eadd7f9231f0)

There is a strong correlation between Production Budget and Worldwide Revenue (r = 0.73)

We can graph this in order to visualize the correlation

```python
plt.figure(figsize=(10, 6))
plt.scatter(filmData['Production budget $'], filmData['Worldwide gross $'], alpha=0.6, label = 'Data points')
# Calculate Trend Line
z = np.polyfit(filmData['Production budget $'], filmData['Worldwide gross $'], 1) # 1 for linear
p = np.poly1d(z)
plt.plot(filmData['Production budget $'], p(filmData['Production budget $']), 'r--', label = 'Trend line')

plt.title('Production Budget vs Worldwide Gross Revenue')
plt.xlabel('Production Budget ($)')
plt.ylabel('Worldwide Gross Revenue ($)')
plt.grid(True)
plt.legend()
plt.show()
```
![ProductBudget_WWRevenue_trendline](https://github.com/Bozzojr/FilmDataAnalysis_Python/assets/123130175/604252c9-391a-4c85-b559-63d5f9ed1884)
## Statistical Testing for Genres
### ANOVA (Analysis of Variance
We can use the ANOVA to see if average worldwide gross revenue significantly differ among genres

```python
genre_groups = {}
for genre in exploded_genres['genre_list'].unique():
    genre_groups[genre] = exploded_genres[exploded_genres['genre_list'] == genre]['Worldwide gross $'].dropna()

genre_anova_result = stats.f_oneway(*genre_groups.values())
print(f"ANOVA result: F-statistic = {genre_anova_result.statistic}, p-value = {genre_anova_result.pvalue}")
```
**ANOVA result: F-statistic = 57.62354351968861, p-value = 8.335690614180068e-251**

With a high F-statistic and a low P-value, we can conclude there are significant differences in revenue among genres

We can now use Tukey's HSD (Honestly Significant Difference) test to identify which pairs of genres are significantly different

Our threshold for significant difference will be .05 or 5%, which implies that there is a 5% risk of concluding that a difference exists when there is no actual difference.

```python
genre_tukey_results = pairwise_tukeyhsd(endog=exploded_genres['Worldwide gross $'], groups = exploded_genres['genre_list'], alpha = 0.05)
print(genre_tukey_results)
```
This list shows us, with 95% confidence, which genres are significantly different from eachother in terms of worldwide gross revenue



