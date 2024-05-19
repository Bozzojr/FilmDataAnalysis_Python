import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm
import seaborn as sns

filmData = pd.read_csv('FilmProject\Data\movie_statistic_dataset.csv')

filmData.head()
filmData.describe()
filmData.info()
# 4380 entries and each column has 4380 non null values, indicating no missing values

duplicates = filmData.duplicated()
print(sum(duplicates))
#No dupicates either, data appears to be consistent and reliable

# For an additional check on the data, I am going to make a few histograms and boxplots on the numerical columns to see what the distribution looks like and examine outliers

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

# Genre Analysis
# First lets see what genres are the most popular in this data set
filmData['genre_list'] = filmData['genres'].str.split(',')
exploded_genres = filmData.explode('genre_list')
genre_counts = exploded_genres['genre_list'].value_counts()

print(genre_counts)
# Wow it looks like our top genres are Drama, Comedy, Action, Adventure, and Crime. Lets save a list of our top 10 most popular Genres
ten_most_popular_genres = genre_counts.head(10)
ten_most_popular_genres_names = ten_most_popular_genres.tolist()

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



# Has Production budget spend been increasing as time goes on, if so, by how much?
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

# Statistical Testing for Genres
# ANOVA (Analysis of Variance) Are the average worldwide gross revenues significantly different among genres

#Create a list of revenue data for each of the top genres to perform ANOVA
genre_groups = {}
for genre in exploded_genres['genre_list'].unique():
    genre_groups[genre] = exploded_genres[exploded_genres['genre_list'] == genre]['Worldwide gross $'].dropna()

genre_anova_result = stats.f_oneway(*genre_groups.values())
print(f"ANOVA result: F-statistic = {genre_anova_result.statistic}, p-value = {genre_anova_result.pvalue}")
# ANOVA result: F-statistic = 57.62354351968861, p-value = 8.335690614180068e-251

# Since our F statistical is high and p value is low (almost 0), we can conclude there is significant difference among the different groups of genres
# We can use Tukey's Honestly Significan Difference (HSD) test to identify which pairs of genres are significantly different
genre_tukey_results = pairwise_tukeyhsd(endog=exploded_genres['Worldwide gross $'], groups = exploded_genres['genre_list'], alpha = 0.05)

print(genre_tukey_results)
# This list shows us, with 95% confidence, which genres are significantly different from eachother in terms of worldwide revenue
# I'm going to convert the results to a data frame for easier analysis
genre_tukey_results_df = pd.DataFrame(data = genre_tukey_results._results_table.data[1:], columns = genre_tukey_results._results_table.data[0])



# Correlation Analysis
# Is there any correlation between production budget and revenue?
budget_revenue_correlation = filmData['Production budget $'].corr(filmData['Worldwide gross $'])
# looks like there is a strong positive correlation with r = 0.733
# Graph the strong correlation so we can see it
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


# Let's examine other variables and their correlation to eachother
rating_revenue_correlation = filmData['movie_averageRating'].corr(filmData['Worldwide gross $']) # r = 0.222
runtime_revenue_correlation = filmData['runtime_minutes'].corr(filmData['Worldwide gross $']) # r = 0.239215
votes_revenue_correlation = filmData['movie_numerOfVotes'].corr(filmData['Worldwide gross $']) # r = 0.5835471

# Calculate the correlation matrix for numerical variables
correlation_matrix = filmData[['runtime_minutes', 'movie_averageRating', 'movie_numerOfVotes', 'approval_Index', 'Production budget $', 'Domestic gross $', 'Worldwide gross $']].corr()

# Create a heatmap using seaborn
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title('Heatmap of Correlations Among Numerical Variables')
plt.show()

# Make a regression model for worldwide revenue
# One-hot encode 'genre_list'
filmData_regression = pd.get_dummies(exploded_genres, columns=['genre_list'], drop_first=True)
filmData_regression['production_date'] = pd.to_datetime(filmData_regression['production_date'])
filmData_regression['production_year'] = filmData_regression['production_date'].dt.year

genre_columns = [col for col in filmData_regression.columns if 'genre_list_' in col]

#Re-aggregate Data
filmData_regression = filmData_regression.groupby('movie_title').agg({
    'production_date': 'first',
    'genres': 'first',
    'runtime_minutes': 'first',
    'director_name': 'first',
    'director_professions': 'first',
    'director_birthYear': 'first',
    'director_deathYear': 'first',
    'movie_averageRating': 'first',
    'movie_numerOfVotes': 'first',
    'approval_Index': 'first',  
    'Production budget $': 'first',
    'Domestic gross $': 'first',
    'Worldwide gross $': 'first',
    'production_year': 'first',
    **{col: 'max' for col in genre_columns}
}).reset_index()





# Prepare predictors for x, y least squares constant
X = filmData_regression[['Production budget $', 'runtime_minutes', 'movie_averageRating', 'movie_numerOfVotes', 'production_year'] + [col for col in genre_columns]]
bool_cols = [col for col in X.columns if X[col].dtype == bool]
X[bool_cols] = X[bool_cols].astype(int) 

y = filmData_regression['Worldwide gross $']


# Add a constant for the intercept
X = sm.add_constant(X)

# Fit the model
model = sm.OLS(y, X).fit()

# View the model summary
print(model.summary())
