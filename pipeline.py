"""
Spotify Top Songs (2010-2019) — Popularity Prediction
Techolas Technologies Internship — Machine Learning Project

Predicts a song's Spotify popularity score (0-100) from its audio features
(tempo, energy, danceability, loudness, liveness, valence, duration,
acousticness, speechiness) plus release year and genre.
"""

import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set_style("whitegrid")
OUT = "outputs"
RESULTS = {}

# ----------------------------------------------------------------------
# 1. DATA UNDERSTANDING
# ----------------------------------------------------------------------
df = pd.read_csv("data/top10s.csv")
df = df.rename(columns={
    "Unnamed: 0": "id", "top genre": "genre", "bpm": "tempo", "nrgy": "energy",
    "dnce": "danceability", "dB": "loudness", "live": "liveness", "val": "valence",
    "dur": "duration", "acous": "acousticness", "spch": "speechiness", "pop": "popularity"
})
df = df.drop(columns=["id"])

RESULTS["shape"] = df.shape
RESULTS["dtypes"] = df.dtypes.astype(str).to_dict()
RESULTS["describe"] = df.describe().round(2).to_dict()
RESULTS["n_unique_artists"] = df["artist"].nunique()
RESULTS["n_unique_genres"] = df["genre"].nunique()
RESULTS["n_duplicated_titles"] = int(df["title"].duplicated().sum())
RESULTS["missing_values"] = int(df.isnull().sum().sum())

# ----------------------------------------------------------------------
# 2. EDA VISUALIZATIONS
# ----------------------------------------------------------------------
numeric_cols = ["tempo", "energy", "danceability", "loudness", "liveness",
                 "valence", "duration", "acousticness", "speechiness", "popularity", "year"]

# 2a. Correlation heatmap
plt.figure(figsize=(10, 8))
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
plt.title("Correlation Heatmap of Audio Features")
plt.tight_layout()
plt.savefig(f"{OUT}/01_heatmap.png", dpi=130)
plt.close()
RESULTS["corr_with_popularity"] = corr["popularity"].drop("popularity").sort_values(ascending=False).round(3).to_dict()

# 2b. Boxplots of key features
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
box_feats = ["tempo", "energy", "danceability", "loudness", "liveness", "valence", "acousticness", "speechiness"]
for ax, col in zip(axes.flat, box_feats):
    sns.boxplot(y=df[col], ax=ax, color="#1DB954")
    ax.set_title(col)
plt.suptitle("Boxplots of Audio Features (Outlier Check)")
plt.tight_layout()
plt.savefig(f"{OUT}/02_boxplots.png", dpi=130)
plt.close()

# 2c. Pairplot (sampled features to keep it readable)
pp = sns.pairplot(df[["popularity", "energy", "danceability", "acousticness", "valence"]], diag_kind="kde",
                   plot_kws={"alpha": 0.4, "s": 15, "color": "#1DB954"})
pp.fig.suptitle("Pairplot: Popularity vs Key Audio Features", y=1.02)
pp.savefig(f"{OUT}/03_pairplot.png", dpi=130)
plt.close()

# 2d. Count plot of top genres and songs per year
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
top_genres = df["genre"].value_counts().head(10)
sns.barplot(x=top_genres.values, y=top_genres.index, ax=axes[0], palette="viridis")
axes[0].set_title("Top 10 Genres by Song Count")
axes[0].set_xlabel("Count")
sns.countplot(x="year", data=df, ax=axes[1], palette="crest")
axes[1].set_title("Songs per Year (2010–2019)")
plt.tight_layout()
plt.savefig(f"{OUT}/04_countplots.png", dpi=130)
plt.close()

# 2e. Popularity distribution
plt.figure(figsize=(7, 5))
sns.histplot(df["popularity"], bins=25, kde=True, color="#1DB954")
plt.title("Distribution of Popularity Score")
plt.tight_layout()
plt.savefig(f"{OUT}/05_popularity_dist.png", dpi=130)
plt.close()

# ----------------------------------------------------------------------
# 3. DATA PREPROCESSING
# ----------------------------------------------------------------------
work = df.copy()

# Outlier handling: a handful of "silent" tracks (e.g. bpm/energy/loudness = 0/-60)
# are data artifacts (interludes), not genuine songs -> drop them.
before = len(work)
work = work[(work["popularity"] > 0) & (work["loudness"] > -30)].reset_index(drop=True)
RESULTS["rows_dropped_outliers"] = before - len(work)

# Genre has 50+ unique values with a long tail -> group rare genres into "other"
genre_counts = work["genre"].value_counts()
common_genres = genre_counts[genre_counts >= 8].index
work["genre_grouped"] = work["genre"].where(work["genre"].isin(common_genres), "other")
RESULTS["n_genre_groups"] = work["genre_grouped"].nunique()

# One-hot encode grouped genre
work = pd.get_dummies(work, columns=["genre_grouped"], prefix="genre")

feature_cols = ["tempo", "energy", "danceability", "loudness", "liveness", "valence",
                 "duration", "acousticness", "speechiness", "year"] + \
                [c for c in work.columns if c.startswith("genre_")]

X = work[feature_cols]
y = work["popularity"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
num_cols = ["tempo", "energy", "danceability", "loudness", "liveness", "valence",
            "duration", "acousticness", "speechiness", "year"]
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

RESULTS["train_size"] = len(X_train)
RESULTS["test_size"] = len(X_test)
RESULTS["n_features"] = X.shape[1]

# ----------------------------------------------------------------------
# 4. MODEL BUILDING + HYPERPARAMETER TUNING (GridSearchCV, 5-fold CV)
# ----------------------------------------------------------------------
cv = KFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "Linear Regression": (LinearRegression(), {}),
    "Ridge": (Ridge(random_state=42), {"alpha": [0.1, 1.0, 10.0, 50.0]}),
    "Lasso": (Lasso(random_state=42, max_iter=5000), {"alpha": [0.01, 0.1, 1.0, 5.0]}),
    "KNN": (KNeighborsRegressor(), {"n_neighbors": [3, 5, 7, 9, 11], "weights": ["uniform", "distance"]}),
    "SVR": (SVR(), {"C": [1, 10, 50], "kernel": ["rbf", "linear"], "epsilon": [0.5, 1.0]}),
    "Decision Tree": (DecisionTreeRegressor(random_state=42),
                       {"max_depth": [3, 5, 7, 10, None], "min_samples_leaf": [1, 3, 5]}),
    "Random Forest": (RandomForestRegressor(random_state=42),
                       {"n_estimators": [100, 200], "max_depth": [5, 10, None], "min_samples_leaf": [1, 3]}),
    "AdaBoost": (AdaBoostRegressor(random_state=42), {"n_estimators": [50, 100, 150], "learning_rate": [0.05, 0.1, 0.5, 1.0]}),
    "GradientBoosting": (GradientBoostingRegressor(random_state=42),
                          {"n_estimators": [100, 200], "learning_rate": [0.01, 0.05, 0.1], "max_depth": [2, 3, 4]}),
}

model_results = {}
fitted_models = {}
for name, (estimator, grid) in models.items():
    if grid:
        gs = GridSearchCV(estimator, grid, cv=cv, scoring="neg_mean_squared_error", n_jobs=-1)
        gs.fit(X_train_scaled, y_train)
        best_model = gs.best_estimator_
        best_params = gs.best_params_
    else:
        best_model = estimator.fit(X_train_scaled, y_train)
        best_params = {}

    preds = best_model.predict(X_test_scaled)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    model_results[name] = {"best_params": best_params, "RMSE": round(rmse, 3),
                            "MAE": round(mae, 3), "R2": round(r2, 3)}
    fitted_models[name] = best_model
    print(f"{name:20s}  RMSE={rmse:6.3f}  MAE={mae:6.3f}  R2={r2:6.3f}  params={best_params}")

RESULTS["model_results"] = model_results

results_df = pd.DataFrame(model_results).T.sort_values("RMSE")
best_model_name = results_df.index[0]
RESULTS["best_model"] = best_model_name

# ----------------------------------------------------------------------
# 5. MODEL EVALUATION VISUALS
# ----------------------------------------------------------------------
# 5a. Comparison bar chart
plt.figure(figsize=(10, 5))
plot_df = results_df[["RMSE", "MAE"]].astype(float)
plot_df.plot(kind="bar", ax=plt.gca(), color=["#1DB954", "#191414"])
plt.title("Model Comparison — RMSE & MAE (lower is better)")
plt.ylabel("Error")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
plt.savefig(f"{OUT}/06_model_comparison.png", dpi=130)
plt.close()

# 5b. R2 comparison
plt.figure(figsize=(9, 5))
r2_series = results_df["R2"].astype(float).sort_values()
colors = ["#d62728" if v < 0 else "#1DB954" for v in r2_series.values]
r2_series.plot(kind="barh", color=colors)
plt.title("Model Comparison — R\u00b2 Score (higher is better)")
plt.tight_layout()
plt.savefig(f"{OUT}/07_r2_comparison.png", dpi=130)
plt.close()

# 5c. Predicted vs Actual for best model
best_model = fitted_models[best_model_name]
best_preds = best_model.predict(X_test_scaled)
plt.figure(figsize=(6.5, 6))
plt.scatter(y_test, best_preds, alpha=0.5, color="#1DB954")
lims = [min(y_test.min(), best_preds.min()), max(y_test.max(), best_preds.max())]
plt.plot(lims, lims, "k--", lw=1)
plt.xlabel("Actual Popularity")
plt.ylabel("Predicted Popularity")
plt.title(f"Predicted vs Actual — {best_model_name}")
plt.tight_layout()
plt.savefig(f"{OUT}/08_pred_vs_actual.png", dpi=130)
plt.close()

# 5d. Feature importance (if tree-based)
if hasattr(best_model, "feature_importances_"):
    imp = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False).head(12)
    plt.figure(figsize=(8, 6))
    imp.plot(kind="barh", color="#1DB954")
    plt.gca().invert_yaxis()
    plt.title(f"Top Feature Importances — {best_model_name}")
    plt.tight_layout()
    plt.savefig(f"{OUT}/09_feature_importance.png", dpi=130)
    plt.close()
    RESULTS["top_features"] = imp.round(3).to_dict()

with open(f"{OUT}/results.json", "w") as f:
    json.dump(RESULTS, f, indent=2, default=str)

print("\nBest model:", best_model_name)
print(results_df)
print("\nDone. Results saved to outputs/results.json")
