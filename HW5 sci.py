
"""     Filter the dataset so that it only shows only the NBA regular season data.
    Using this filtered dataset, determine the player who has played the most regular seasons.
    Calculate this player's three point accuracy for each season that they played.
    Perform a linear regression for their three point accuracy across the years played, and create a line of best fit.
    Calculate the average three point accuracy by integrating the fit line over the played seasons and dividing by the number difference in played seasons (lastest season - earliest season). 
    How does this value compared to the actual average number of 3-pointers made by this player?
    This player did not participate in the 2002-2003 and 2015-2016 seasons. Use interpolation to estimate these missing values.

Statistics can be a great predictor tool can be used in a variety of capacities.

    Calculate the statistical mean, variance, skew, and kurtosis for the Field Goals Made (FGM) column and the Field Goals Attempted (FGA) column. How do the statistics from the FGM compared to the FGA column.
    Perform a relational t-test on Field Goals Made (FGM) and Field Goals Attempted (FGA) columns. Addittionally, perform a regular t-test on the FGM and FGA columns individually.  How to these results of the individual t-test compared to the t-test run for the two related samples together?
    """
import scipy as sp
import scipy.stats as stats
import scipy.integrate as spi
from scipy.interpolate import interp1d
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator



# Load the dataset
df = pd.read_csv('players_stats_by_season_full_details.csv')

# asked github for help filtering the DataFrame to include only NBA regular season data and got this
nba_regular_season_df = df[(df['League'] == 'NBA') & (df['Stage'] == 'Regular_Season')].copy()

#I needed githubs help to make season into a single number so I could do a linear regression without a type error and it gave this which makes a lot of sense now
nba_regular_season_df.loc[:, 'Season_Start_Year'] = nba_regular_season_df['Season'].apply(lambda x: int(x.split('-')[0]))

# I got help from github with the nunqiue function and then used idxmax to get the player with the most seasons
player_season_counts = nba_regular_season_df.groupby('Player')['Season'].nunique()

#storing the player with the most seasons and the count of the most seasons
most_seasons_player = player_season_counts.idxmax()
most_seasons_count = player_season_counts.max()

# Filter the dataset to only include the player with the most seasons
player_df = nba_regular_season_df[nba_regular_season_df['Player'] == most_seasons_player].copy()

#My original method that's commented out caused a warning so I asked github for help and got the loc method below
#player_df['3P_accuracy'] = player_df['3PM'] / player_df['3PA'
player_df.loc[:, '3P_accuracy'] = 100* (player_df['3PM'] / player_df['3PA'])



# Perform a linear regression on the player's three-point accuracy by season
slope, intercept, r_value, p_value, std_err = stats.linregress(player_df['Season_Start_Year'], player_df['3P_accuracy'])

def linear_regression(x):
    return slope * x + intercept

# Generate new x values for the line of best fit
x = np.linspace(player_df['Season_Start_Year'].min(), player_df['Season_Start_Year'].max(), 100)
y = linear_regression(x)

# Plot the three-point accuracy and the line of best fit
plt.scatter(player_df['Season_Start_Year'], player_df["3P_accuracy"], color='red', label='3P Accuracy')
plt.plot(x, y, color='green', label='Line of Best Fit')
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
plt.legend()
plt.grid()
plt.xlabel('Season Start Year')
plt.ylabel('3P Accuracy (%)')
plt.title(f'Three-Point Accuracy for {most_seasons_player}')


# Calculate the average three-point accuracy by integrating the line of best fit
start_year = player_df['Season_Start_Year'].min()
end_year = player_df['Season_Start_Year'].max()
integral, _ = spi.quad(linear_regression, start_year, end_year)
average_3P_accuracy_fit = integral / (end_year - start_year)

# Calculate the actual average three-point accuracy
actual_average_3P_accuracy = player_df['3P_accuracy'].mean()

print(f"Average 3P Accuracy (integrated fit line): {average_3P_accuracy_fit}")
print(f"Actual Average 3P Accuracy: {actual_average_3P_accuracy}")

print("\n")

#    This player did not participate in the 2002-2003 and 2015-2016 seasons. Use interpolation to estimate these missing values.
# Identify the seasons with missing data
missing_seasons = [2002, 2015]

# Create an interpolation function based on the available data
interpolator = interp1d(player_df['Season_Start_Year'], player_df['3P_accuracy'], kind='linear', fill_value="extrapolate")

# Estimate the missing values using the interpolation function
estimated_values = interpolator(missing_seasons)

# Print the estimated values for the missing seasons
for season, value in zip(missing_seasons, estimated_values):
    print(f"Estimated 3P Accuracy for {season}-{season+1}: {value}")
print("\n")

# I don't know why github did it this way but I just asked it to write this code for me since I knew it would take a while to write it all and it just chose to do it this way
fgm_stats = {
    'mean': nba_regular_season_df['FGM'].mean(),
    'variance': nba_regular_season_df['FGM'].var(),
    'skew': stats.skew(nba_regular_season_df['FGM']),
    'kurtosis': stats.kurtosis(nba_regular_season_df['FGM'])
}

fga_stats = {
    'mean': nba_regular_season_df['FGA'].mean(),
    'variance': nba_regular_season_df['FGA'].var(),
    'skew': stats.skew(nba_regular_season_df['FGA']),
    'kurtosis': stats.kurtosis(nba_regular_season_df['FGA'])
}

# Store the values into a list
fgm_stats_list = [fgm_stats['mean'], fgm_stats['variance'], fgm_stats['skew'], fgm_stats['kurtosis']]
fga_stats_list = [fga_stats['mean'], fga_stats['variance'], fga_stats['skew'], fga_stats['kurtosis']]

# Print the results
for stat, fgm, fga in zip(['Mean', 'Variance', 'Skew', 'Kurtosis'], fgm_stats_list, fga_stats_list):
    print(f"FGM {stat}: {fgm}")
    print(f"FGA {stat}: {fga}")

print("\n")

# I got help from github with the t tests here
# Perform a relational t-test (paired t-test) on FGM and FGA columns
t_stat_rel, p_value_rel = stats.ttest_rel(nba_regular_season_df['FGM'], nba_regular_season_df['FGA'])

# Perform a regular t-test on the FGM column
t_stat_fgm, p_value_fgm = stats.ttest_1samp(nba_regular_season_df['FGM'], 0)

# Perform a regular t-test on the FGA column
t_stat_fga, p_value_fga = stats.ttest_1samp(nba_regular_season_df['FGA'], 0)

# Print the results of the t-tests
print(f"Paired t-test (FGM vs FGA) - t-statistic: {t_stat_rel}, p-value: {p_value_rel}")
print(f"One-sample t-test (FGM) - t-statistic: {t_stat_fgm}, p-value: {p_value_fgm}")
print(f"One-sample t-test (FGA) - t-statistic: {t_stat_fga}, p-value: {p_value_fga}")






# moved this down here so that the things above print out before the plot
plt.show()