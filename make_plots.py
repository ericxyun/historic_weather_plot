#%%
import pandas as pd
from functools import reduce
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib
import os
import re
import datetime
import matplotlib.dates as mdates
from collections import defaultdict
from matplotlib.patches import Ellipse


def make_df():
    files = ['min_daily.txt', 'max_daily.txt', 'avg_daily.txt']
    file_path = ["data/{}".format(x) for x in files]
    data_list = [read_txt(x) for x in file_path]
    data = reduce(lambda left, right: pd.merge(left, right), data_list)
    data['year'] = data['Date'].apply(lambda x: x.year)
    data['month'] = data['Date'].apply(lambda x: x.month)
    data['day'] = data['Date'].apply(lambda x: x.day)
    return data


def read_txt(filename):
    """First two rows of the .txt files are not usable"""
    df = pd.read_csv(filename,
                     sep="\t",
                     skiprows=[0, 1],
                     parse_dates=['Date'],
                     usecols=['Date', '(degrees)'])
    df.rename(columns={'(degrees)': re.findall('(?:data\/)(.*)(?:\.txt)',
                                               filename)[0]},
              inplace=True)
    return df


def find_min_max(year):
    temp_dict = defaultdict(dict)
    df = data_year[year]
    min_df = df.loc[df['min_daily'] == df['min_daily'].min()].head(1)
    max_df = df.loc[df['max_daily'] == df['max_daily'].max()].head(1)
    date_min = min_df['Date'].iloc[0].date()
    date_max = max_df['Date'].iloc[0].date()
    temp_dict['min'][date_min] = min_df['min_daily'].iloc[0]
    temp_dict['max'][date_max] = max_df['max_daily'].iloc[0]
    return temp_dict


def min_max_year(year):
    """Prepare dataframe for plotting"""
    df = data[data['year'] == year].sort_values('Date')
    x = df['Date']
    avg_daily = df['avg_daily']
    min_daily = df['min_daily']
    max_daily = df['max_daily']

    """Set global plot settings"""
    matplotlib.rcParams['font.sans-serif'] = ['Aksidenz-Grotesk']
    matplotlib.style.use('seaborn')
    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(111,
                         autoscale_on=False, 
                         xlim=(datetime.date(year, 1, 1),
                               datetime.date(year + 1, 1, 1)),
                         ylim=(0, 140))

    """Make x-axis labels into months"""
    months = mdates.MonthLocator()
    fmt = mdates.DateFormatter('%b')
    X = plt.gca().xaxis
    X.set_major_locator(months)
    X.set_major_formatter(fmt)

    """Set the size ticks to be consistent for every plot"""
    plt.xlim(datetime.date(year, 1, 1),
             datetime.date(year + 1, 1, 1))
    plt.ylim(0, 140)

    """Labels for plot"""
    plt.ylabel('Temperature (°F)', fontdict={'weight': 'bold'})
    plt.title(year, fontdict={'fontsize': 14})
    minimum = int(round(min_daily.min()))
    maximum = int(round(max_daily.max()))
    average = int(round(avg_daily.mean()))
    plt.text(datetime.date(year, 12, 5), 128,
             "Average: {}".format(average))

    """Prepare variables to be used for plot annotations"""
    min_max = find_min_max(year)
    x_min = list(min_max['min'].keys())[0]
    y_min = min_max['min'][x_min]
    x_max = list(min_max['max'].keys())[0]
    y_max = min_max['max'][x_max]

    """
    Plot the arrow annotations
    https://matplotlib.org/3.1.0/api/_as_gen/matplotlib.pyplot.annotate.html
    https://matplotlib.org/3.1.0/api/_as_gen/matplotlib.patches.FancyArrowPatch.html#matplotlib.patches.FancyArrowPatch
    """
    el = Ellipse((2, -1), 0.5, 0.5)
    ax.add_patch(el)
    ax.annotate("Min: {}".format(minimum),
                xy=(x_min, y_min), xycoords='data',
                xytext=(10, 10), textcoords='offset points',
                size=10,
                arrowprops=dict(arrowstyle='fancy',
                                facecolor='g', ec='none',
                                patchB=el,
                                connectionstyle='angle3,angleA=90,angleB=0'))
    ax.annotate("Max: {}".format(maximum),
                xy=(x_max, y_max), xycoords='data',
                xytext=(10, 10), textcoords='offset points',
                size=10,
                arrowprops=dict(arrowstyle='fancy',
                                facecolor='r', ec='none',
                                patchB=el,
                                connectionstyle='angle3,angleA=90,angleB=0'))

    """Plot hollow circle as annotations"""
    # plt.scatter(x_min, y_min, s=150, facecolors='none', edgecolors='g')
    # plt.scatter(x_max, y_max, s=150, facecolors='none', edgecolors='r')

    """Plot the min, max, and average temperatures"""
    plt.plot(x, avg_daily)
    plt.plot(x, min_daily, color='green', linewidth=0.3, linestyle='dashed')
    plt.plot(x, max_daily, color='red', linewidth=0.3, linestyle='dashed')
    plt.savefig("plots/{}.png".format(year), 
                facecolor='white', transparent=False)
    plt.show()

if __name__ == '__main__':
    data = make_df()
    year_unique = data['year'].unique()
    num_days_list = [len(data[data['year'] == x]) for x in year_unique]
    """Try to set a condition to apply only to the last item on the list"""
    for year in year_unique[:-1]:
        if not len(data[data['year'] == year]) < num_day_list[3]:
            min_max_year(year)
    min_max_year(year_unique[-1])
