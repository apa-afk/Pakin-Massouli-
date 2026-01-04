import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D



def plot_multiple(dfs, plotting_function, titles):

    fig, axs = plt.subplots(1, len(dfs), figsize=(14, 6))


    for i in range(len(dfs)):
        plotting_function(dfs[i], ax=axs[i], title=titles[i])
        
    plt.tight_layout()
    plt.show()

def hrmn_to_hour(val):
    try:
        return int(str(val)[:2])
    except (ValueError, TypeError):
        return pd.NA  # ou np.nan



def dynamiques_temporelles(list_df):
    nb_annees = len(list_df)
    occurences = [ len(list_df[i]) for i in range(nb_annees)] #nombre de victimes par an
    occurences_graves = [len(list_df[i].loc[list_df[i]['grav'] == 4]) for i in range(nb_annees)]
    grav_moy = [list_df[i]['grav'].mean() for i in range(nb_annees)]
    sexe_moy = [list_df[i]['sexe'].mean() for i in range(nb_annees)]
    
    return occurences, occurences_graves, grav_moy, sexe_moy



def normalize_coordinates(df, lat_col='lat', long_col='long'):

    def normalize_lat(val):
        if pd.isna(val):
            return np.nan

        s = str(val).strip().replace(',', '').replace(' ', '').replace('.', '')
        if s in ['0', '000000', '0000000'] or len(s) < 4:
            return np.nan

        return float(f'{s[:2]}.{s[2:]}')  # ex: 5051500 -> 50.51500

    def normalize_long(val):
        if pd.isna(val):
            return np.nan

        s = str(val).strip().replace(',', '').replace(' ', '').replace('.', '')
        if s in ['0', '000000', '0000000'] or len(s) < 4:
            return np.nan

        # gérer le signe
        sign = -1 if s.startswith('-') else 1
        s = s.lstrip('-')

        return sign * float(f'{s[:1]}.{s[1:]}')  # ex: 082600 -> 0.82600

    df[lat_col] = df[lat_col].apply(normalize_lat)
    df[long_col] = df[long_col].apply(normalize_long)

    return df



def to_int(df):

    for col in df.columns:
        s = df[col]

        # Try numeric conversion, keep NaN
        s_num = pd.to_numeric(s, errors='coerce')

        # Only convert if values are integer-like
        mask_int = s_num.notna() & (s_num % 1 == 0)

        # Create a copy to avoid SettingWithCopy
        s_out = s.copy()

        # Assign ints where appropriate
        s_out[mask_int] = s_num[mask_int].astype(int)

        df[col] = s_out

    return df



def plot_crashes_per_month(df, ax=None, title=None):
    sns.countplot(x='mois', data=df, ax=ax)
    if ax:
        ax.set_title(title if title else 'Number of Crashes per Month')
        ax.set_xlabel('Month')
        ax.set_ylabel('Number of Crashes')
    else:
        plt.title(title if title else 'Number of Crashes per Month')
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



def plot_accidents_heatmap_cote_a_cote(df, ax=None, title=None):
    heat = df.pivot_table(index='hour', columns='weekday', values='Num_Acc', aggfunc='count')
    weekday_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    heat = heat[weekday_order]
    
    sns.heatmap(heat, cmap='Reds', ax=ax)
    if ax:
        ax.set_title(title if title else 'Accidents by Hour and Weekday')
        ax.set_xlabel('Weekday')
        ax.set_ylabel('Hour of Day')
    else:
        plt.title(title if title else 'Accidents by Hour and Weekday')
        plt.xlabel('Weekday')
        plt.ylabel('Hour of Day')
        plt.show()



def run_regression(df, variables):

    if 'grav' not in variables:
        raise ValueError("'grav' must be included in variables.")

    predictors = [var for var in variables if var != 'grav']

    X = df[predictors].dropna()
    y = df['grav'].loc[X.index]

    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()

    print(model.summary())

    return model



def plot_pca_2D_3D(
    df,
    variables=[
        'grav', 'hour', 'lat', 'long', 'agg', 'vma', 'nbv',
        'catr', 'circ', 'lum', 'atm', 'surf', 'sexe', 'an_nais'
    ],
    csp=False
):

    # --- Data cleaning ---
    pca_df = df[variables].apply(pd.to_numeric, errors='coerce')
    pca_df = pca_df.dropna()

    # Optional CSP dummies
    if csp:
        csp_dummies = pd.get_dummies(
            df.loc[pca_df.index, "csp_conducteur"],
            prefix="csp"
        )
        pca_df = pd.concat([pca_df, csp_dummies], axis=1)

    # --- Standardization ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(pca_df)

    # --- PCA ---
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)

    print("Explained variance ratio:")
    print(pca.explained_variance_ratio_)
    print("Cumulative variance explained:")
    print(pca.explained_variance_ratio_.cumsum())

    # PCA dataframe for visualization
    pca_vis = pd.DataFrame(
        X_pca,
        columns=["PC1", "PC2", "PC3"],
        index=pca_df.index
    )

    if csp:
        pca_vis["csp"] = df.loc[pca_df.index, "csp_conducteur"]

    # --- Plotting ---
    fig = plt.figure(figsize=(16, 7))

    # 2D PCA
    ax1 = fig.add_subplot(1, 2, 1)
    if csp:
        sns.scatterplot(
            data=pca_vis,
            x="PC1", y="PC2",
            hue="csp",
            alpha=0.4,
            s=35,
            ax=ax1
        )
        ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    else:
        ax1.scatter(
            pca_vis["PC1"], pca_vis["PC2"],
            alpha=0.4, s=20
        )

    ax1.set_title("PCA – 2D Projection (PC1 vs PC2)")
    ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")

    # 3D PCA
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    if csp:
        csp_codes = pd.Categorical(pca_vis["csp"]).codes
        ax2.scatter(
            pca_vis["PC1"],
            pca_vis["PC2"],
            pca_vis["PC3"],
            c=csp_codes,
            cmap="tab10",
            alpha=0.4,
            s=30
        )
    else:
        ax2.scatter(
            pca_vis["PC1"],
            pca_vis["PC2"],
            pca_vis["PC3"],
            alpha=0.4,
            s=25
        )

    ax2.set_title("PCA – 3D Projection (PC1, PC2, PC3)")
    ax2.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax2.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax2.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)")

    plt.tight_layout()
    plt.show()

    return pca, pca_df


    
def plot_correlation_circle(
    pca,
    feature_names,
    pc1=0,
    pc2=1,
    figsize=(5, 5),
    max_features=None
):
    """
    Plot PCA correlation circle for selected components.

    Parameters
    ----------
    pca : fitted sklearn PCA
    feature_names : list of feature names used in PCA
    pc1, pc2 : int
        Principal components to plot
    max_features : int or None
        Optionally limit number of displayed features
        (useful if CSP dummies are included)
    """

    loadings = pca.components_.T

    if len(feature_names) != loadings.shape[0]:
        raise ValueError(
            "feature_names length does not match PCA components. "
            "Make sure you pass the exact columns used for PCA."
        )

    if max_features is not None:
        feature_names = feature_names[:max_features]
        loadings = loadings[:max_features, :]

    plt.figure(figsize=figsize)

    # Correlation circle
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
            length_includes_head=True,
            alpha=0.8
        )
        plt.text(
            x * 1.1,
            y * 1.1,
            feature,
            fontsize=9,
            ha='center',
            va='center'
        )

    # Axes
    plt.axhline(0, color='grey', linewidth=0.8)
    plt.axvline(0, color='grey', linewidth=0.8)

    plt.xlabel(f"PC{pc1+1} ({pca.explained_variance_ratio_[pc1]*100:.1f}%)")
    plt.ylabel(f"PC{pc2+1} ({pca.explained_variance_ratio_[pc2]*100:.1f}%)")

    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.title("PCA – Circle of Correlations")

    plt.gca().set_aspect("equal")
    plt.grid(False)
    plt.tight_layout()
    plt.show()



def pca_analysis(pca, pca_df, threshold=0.2):
    """
    Analyze a fitted PCA and return correlations, cos², and significant variables.

    Parameters
    ----------
    pca : sklearn PCA object
        Fitted PCA.
    pca_df : pandas DataFrame
        Original dataset used for PCA.
    threshold : float, default=0.2
        Threshold for sum of cos² in first 2 PCs to consider a variable significant.

    Returns
    -------
    corr_df : pandas DataFrame
        Correlation of each variable with each principal component.
    cos2_df : pandas DataFrame
        Squared correlations (quality of representation) of each variable.
    significant_vars : list
        List of variables whose sum of cos² across first 2 PCs >= threshold.
    """
    
    # Ensure X matches the PCA input
    X = pca_df.values
    
    # Loadings
    loadings = pca.components_.T  # shape: (n_features, n_PCs)
    
    # Compute correlations: cor[i,j] = loading[i,j] * sqrt(explained_variance[j])
    correlations = loadings * np.sqrt(pca.explained_variance_)
    
    # Convert to DataFrame
    corr_df = pd.DataFrame(
        correlations,
        index=pca_df.columns,
        columns=[f"PC{i+1}" for i in range(loadings.shape[1])]
    )
    
    # Compute cos² (quality of representation)
    cos2_df = corr_df**2
    
    # Sum of cos² across first 2 PCs
    cos2_df['sum_PC1_PC2'] = cos2_df.iloc[:, [0, 1]].sum(axis=1)
    
    # Display results
    print("Correlation of variables with PCs:")
    print(corr_df.round(3))
    print("\nCos² (quality of representation):")
    print(cos2_df.round(3))
    
    # Identify significant variables
    significant_vars = cos2_df[cos2_df['sum_PC1_PC2'] >= threshold].index.tolist()
    
    print(f"\nSignificant variables (sum cos² >= {threshold} for PC1 and PC2):")
    print(significant_vars)
    
    return corr_df, cos2_df, significant_vars



def plot_crashes_heatmap_cote_a_cote(df, ax=None, title = None):
    # Filtrer sur les coordonnées (France métropolitaine)
    df = df[
        (df['long'] > -5) & (df['long'] <= 9.7) &
        (df['lat']  >= 41.0) & (df['lat']  <= 51.5)
    ]
    
    x, y = df['long'], df['lat']
    
    if ax is None:
        plt.figure(figsize=(8,8))
        ax = plt.gca()
    
    ax.scatter(x, y, s=4, c='red', alpha=0.03)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.set_xlim(-5.5, 9.7)
    ax.set_ylim(41.0, 51.5)
