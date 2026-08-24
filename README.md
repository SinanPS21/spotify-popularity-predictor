# Spotify Song Popularity Prediction

Machine learning project predicting a song's
Spotify popularity score (0–100) from its audio features, using the
["Top Spotify Songs from 2010–2019, by Year"](https://www.kaggle.com/datasets/leonardopena/top-spotify-songs-from-20102019-by-year) dataset (603 tracks).

## What's here

| File | Description |
|---|---|
| `Spotify_Popularity_Prediction.ipynb` | Full notebook: EDA, preprocessing, 9 tuned regression models, evaluation |
| `Spotify_Popularity_Prediction_Report.docx` | Formal write-up following the Techolas ML project report template |
| `pipeline.py` | Standalone script version of the full pipeline (used to generate the notebook's real outputs) |
| `data/top10s.csv` | Dataset (603 songs, 2010–2019) |
| `outputs/` | Generated charts and `results.json` (raw metrics) |

## Problem

Can a song's audio characteristics (tempo, energy, danceability, loudness,
liveness, valence, duration, acousticness, speechiness) plus release year and
genre predict how popular it is on Spotify?

## Approach

1. **EDA** — correlation heatmap, boxplots (outlier check), pairplot, count
   plots, popularity distribution.
2. **Preprocessing** — outlier removal, rare-genre grouping + one-hot
   encoding, `StandardScaler`, 80/20 train/test split.
3. **Modeling** — 9 regressors tuned with `GridSearchCV` (5-fold CV): Linear
   Regression, Ridge, Lasso, KNN, SVR, Decision Tree, Random Forest, AdaBoost,
   Gradient Boosting.
4. **Evaluation** — RMSE, MAE, R² on a held-out test set.

## Key result

Audio features alone are **weak predictors** of popularity for songs that
already charted in a year-end Top 10 (best R² ≈ 0.07) — a finding consistent
with prior published work on similar data (e.g. UBC's DSCI 522 project). This
suggests non-audio factors (artist fame, playlist placement, marketing,
release timing) matter far more than the acoustic signature of the song
itself.

## Running it

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
python pipeline.py          # regenerates outputs/*.png and results.json
jupyter notebook Spotify_Popularity_Prediction.ipynb
```

## Reference

This project's framing was adapted from Aman Kharwal's
[*Spotify Recommendation System with Machine Learning*](https://thecleverprogrammer.com/2021/03/03/spotify-recommendation-system-with-machine-learning/)
and built to satisfy the Techolas Technologies ML project report structure
(problem statement, EDA, preprocessing, hyperparameter tuning, model
building/evaluation, conclusion).
