#########################################
# %% Load data into pandas data-frame

from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# en_lpor_explorer.csv - High School Alcohol and Academic Performance
filename: str = './data/en_lpor_explorer.csv'

# try except block in case the file does not exist
try:
    # Read csv data into pandas dataframe
    data_frame = pd.read_csv(filename)
except:
    print(f'File cannot be opened: {filename}')


# en_lpor_classification.csv - High School Alcohol and Academic Performance
filename_classification: str = './data/en_lpor_classification.csv'

# try except block in case the file does not exist
try:
    # Read csv data into pandas dataframe
    data_frame_classification = pd.read_csv(filename_classification)
except:
    print(f'File cannot be opened: {filename_classification}')


print(data_frame.head(1))
print(data_frame_classification.head(1))

#########################################
# %% Histogram function


def show_distribution(data, title: str):
    # Sort data ascending
    data_sorted = data.sort_values(ascending=True)

    # Show histogram
    sns.displot(data_sorted).set(title=title)

    # Show counts of identical items
    counts_sorted = Counter(data_sorted)

    counts_string = [
        f'{counted[0]}: {counted[1]} participants - {round(100 * counted[1] / counts_sorted.total(), 2)}% of total participants' for counted in counts_sorted.items()]

    print('\n'.join(counts_string))


#########################################
# %% Show surveyed schools

# Extract school data
data_school = data_frame['School']

# Display it
title = 'Surveyed Schools and participation numbers'
print(f'\n{title}:\n')
show_distribution(data_school, title=title)


#########################################
# %% Show surveyed sexes

# Extract school data
data_sexes = data_frame['Gender']

# Display it
title = 'Gender of surveyed participants'
print(f'\n{title}:\n')
show_distribution(data_sexes, title=title)


#########################################
# %% Show surveyed ages

# Extract school data
data_ages = data_frame['Age']

# Cast to string so that seaborne does not try to use the actual number values as x axis but rather interprets those
# values as unique symbols thus center aligning bar and tick
data_ages_strings = data_ages.astype(str)

# Display it
title = 'Age distribution of surveyed participants'
print(f'\n{title}:\n')
show_distribution(data_ages_strings, title=title)


#########################################
# %% Generate all other histograms 'automatically'
columns = data_frame.columns
without = ['School', 'Gender', 'Age']

filtered_columns = columns.difference(without)

for column in filtered_columns:
    title = column
    print(f'\n{title}:\n')

    # casting all data to strings to prevent misaligned bars and ticks
    data_casted = data_frame[column].astype(str)

    show_distribution(data=data_casted, title=title)

# unfortunately the auto generated diagrams aren't as readable as some of the values aren't ordered quite right and some
# labels overlap - however for a quick overview of the data it is still useful


#########################################
# %% Find correlated values
correlation = data_frame_classification.corr()

plt.figure(figsize=(12, 8))
# "PiYG" is a good divergent color map (white in the middle and pink and green at the extremes)
# in combination with the vmin and vmax declaration this helps to put the white color in the middle at 0, pink at -1
# (negative correlating) and green at +1 positive correlation
sns.heatmap(correlation, cmap='PiYG', vmin=-1,
            vmax=1).set(title='Correlation Matrix of all Survey Data')

# show unique correlations as table but without the 'diagonal'/'self-correlation'
correlation_table = correlation[correlation < 1].unstack().transpose().sort_values(
    ascending=False).drop_duplicates()
pd.options.display.max_rows = 5000
print(correlation_table)

# show correlations of data with alcohol consumption
print(
    f'\n\nCorrelations with alcohol consumption on weekdays:\n{correlation["Alcohol_Weekdays"][correlation["Alcohol_Weekdays"] < 1].sort_values(ascending=False).drop_duplicates()}')
print(
    f'\n\nCorrelations with alcohol consumption on weekends:\n{correlation["Alcohol_Weekends"][correlation["Alcohol_Weekends"] < 1].sort_values(ascending=False).drop_duplicates()}')

# show correlations of data with grades
print(
    f'\n\nCorrelations with 1st semester grades:\n{correlation["Grade_1st_Semester"][correlation["Grade_1st_Semester"] < 1].sort_values(ascending=False).drop_duplicates()}')
print(
    f'\n\nCorrelations with 2nd semester grades:\n{correlation["Grade_2nd_Semester"][correlation["Grade_2nd_Semester"] < 1].sort_values(ascending=False).drop_duplicates()}')


#########################################
# %% bell curve function
def show_bell_curve(data, title: str | None = None, xlabel: str | None = None, axvline: int | None = None):
    # Sort data ascending
    data_sorted = data.sort_values(ascending=True)

    # Show histogram
    gfg = sns.displot(data_sorted, kind='kde')

    # set labels if given, use default if not
    if title:
        gfg.set(title=title)
    if xlabel:
        gfg.set(xlabel=xlabel)

    if axvline:
        # plot vertical line if given
        plt.axvline(x=axvline)

        # and display amount of values above and below it
        below_axvline = [v for v in data_sorted if v < axvline]
        at_above_axvline = [v for v in data_sorted if v >= axvline]
        print(
            f'Count below line {len(below_axvline)} - {round(100 * len(below_axvline) / len(data_sorted), 2)}%')
        print(
            f'Count at and above line {len(at_above_axvline)} - {round(100 * len(at_above_axvline) / len(data_sorted), 2)}%', '\n')

    # Show counts of identical items
    counts_sorted = Counter(data_sorted)

    counts_string = [
        f'{counted[0]}: {counted[1]} participants - {round(100 * counted[1] / counts_sorted.total(), 2)}% of total participants' for counted in counts_sorted.items()]

    print('\n'.join(counts_string))


#########################################
# %% Show bell curve of 1st semester grades

# Extract school data
data_grades_1st = data_frame['Grade_1st_Semester']
# Cast to int to make sure the values are numerical
data_grades_1st_int = data_grades_1st.astype(int)

# Display it
# (Under the Portuguese system, grades are given on a scale from 0 to 20, the minimum passing grade being 10.)
title = 'Normal distribution of 1st semester grades'
print(f'\n{title}:\n')
show_bell_curve(data_grades_1st_int, title=title,
                xlabel='Grades in 1st semester (minimum passing grade: 10)', axvline=10)


#########################################
# %% Show bell curve of 2nd semester grades

# Extract school data
data_grades_2nd = data_frame['Grade_2nd_Semester']
# Cast to int to make sure the values are numerical
data_grades_2nd_int = data_grades_2nd.astype(int)

# Display it
# (Under the Portuguese system, grades are given on a scale from 0 to 20, the minimum passing grade being 10.)
title = 'Normal distribution of 2nd semester grades'
print(f'\n{title}:\n')
show_bell_curve(data_grades_2nd_int, title=title,
                xlabel='Grades in 2nd semester (minimum passing grade: 10)', axvline=10)

# %%
