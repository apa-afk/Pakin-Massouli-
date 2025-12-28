import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import numpy as np


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D


#modif

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



def plot_pca_2D_3D(df, variables = ["age_conducteur","hour","vma","nb_vehicules","lum","atm","surf"], csp = True):
    
    #variables of interest
    pca_df = df[variables]
    
    if csp :
        pca_df = pd.concat(
            [
                pca_df,
                pd.get_dummies(df["csp_conducteur"], prefix="csp")
            ],
            axis=1
        )

        #taking into account or not the randomized csp
    
    
    #standardazing the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(pca_df)
    
    
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)
    
    print("Explained variance ratio:")
    print(pca.explained_variance_ratio_)
    print("Cumulative variance explained:")
    print(pca.explained_variance_ratio_.cumsum())
    
    
    
    #creating a pca dataframe for plotting
    pca_vis = pd.DataFrame(
        X_pca,
        columns=["PC1", "PC2", "PC3"]
    )
    
    
    pca_vis["csp"] = df.loc[pca_df.index, "csp_conducteur"].values
    
    #2D PCA plot
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=pca_vis,
        x="PC1",
        y="PC2",
        hue="csp",
        alpha=0.4,
        s=40
    )
    
    plt.title("PCA – 2D Projection (PC1 vs PC2)")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()
    
    #3D PCA plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    
    
    csp_codes = pd.Categorical(pca_vis["csp"]).codes
    
    scatter = ax.scatter(
        pca_vis["PC1"],
        pca_vis["PC2"],
        pca_vis["PC3"],
        c=csp_codes,
        cmap="tab10",
        alpha=0.4,
        s=35
    )
    
    ax.set_title("PCA – 3D Projection (PC1, PC2, PC3)")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)")
    
    plt.show()

    return pca, pca_df



    
def plot_correlation_circle(pca, feature_names, pc1=0, pc2=1, figsize=(8, 8)):
    
    loadings = pca.components_.T

    plt.figure(figsize=figsize)


    circle = plt.Circle((0, 0), 1, color='black', fill=False, linestyle='--')
    plt.gca().add_artist(circle)

    # Plot arrows
    for i, feature in enumerate(feature_names):
        x = loadings[i, pc1]
        y = loadings[i, pc2]

        plt.arrow(
            0, 0, x, y,
            head_width=0.03,
            head_length=0.03,
            length_includes_head=True
        )
        plt.text(
            x * 1.08,
            y * 1.08,
            feature,
            fontsize=10,
            ha='right',
            va='top'
        )


    plt.axhline(0, color='grey', linewidth=0.8)
    plt.axvline(0, color='grey', linewidth=0.8)

    plt.xlabel(f"PC{pc1+1} ({pca.explained_variance_ratio_[pc1]*100:.1f}%)")
    plt.ylabel(f"PC{pc2+1} ({pca.explained_variance_ratio_[pc2]*100:.1f}%)")

    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.title("PCA – Circle of Correlations")

    plt.gca().set_aspect('equal')
    plt.grid(False)
    plt.show()



def plot_crashes_heatmap(df) : 

    df['long'] = df['long'].str.replace(',', '.').astype(float)     #converts str coordinates to float
    df['lat']  = df['lat'].str.replace(',', '.').astype(float)
    df = df[
        (df['long'] >= -5.5) & (df['long'] <= 9.7) &
        (df['lat']  >= 41.0) & (df['lat']  <= 51.5)
    ]
    
    x, y = df['long'], df['lat']
    
    
    
    plt.scatter(x, y, s=4, c='red', alpha=0.03)


    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Occurences des accidents routiers en 2024")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()