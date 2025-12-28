import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm


def plot_crashes_per_month(df):

    sns.countplot(x='mois', data=df)
    plt.title('Number of Crashes per Month')
    plt.xlabel('Month')
    plt.ylabel('Number of Crashes')
    plt.show()


def plot_crashes_by_lum(df):

    sns.countplot(x='lum', data=df)
    plt.title('Crashes by Light Conditions')
    plt.xlabel('Light Condition Code')
    plt.ylabel('Count')
    plt.show()


def plot_crashes_by_surface(df):

    sns.countplot(x='surf', data=df)
    plt.title('Crashes by Road Surface')
    plt.xlabel('Road Surface Type')
    plt.ylabel('Count')
    plt.show()


def plot_vehicles_distribution(df):

    sns.histplot(df['nb_vehicules'], bins=10, kde=False)
    plt.title('Distribution of Vehicles per Crash')
    plt.xlabel('Number of Vehicles')
    plt.ylabel('Count')
    plt.show()


def plot_accidents_heatmap(df):

    heat = df.pivot_table(index='hour', columns='weekday', values='Num_Acc', aggfunc='count')
    weekday_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

    heat = heat[weekday_order]
    sns.heatmap(heat, cmap='Reds')
    plt.title('Accidents by Hour and Weekday')
    plt.xlabel('Weekday')
    plt.ylabel('Hour of Day')
    plt.show()


def run_regression(df):

    X = df[['lum','atm','surf','vma','nb_vehicules']].dropna()
    y = df.loc[X.index, 'nb_usagers']
    X = sm.add_constant(X)
    
    model = sm.OLS(y, X).fit()
    print(model.summary())




