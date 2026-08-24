"""
Spotify Popularity Predictor — Streamlit App
Techolas Technologies ML Internship Project

Run locally:   streamlit run app.py
Deploy:        push this folder (incl. data/top10s.csv) to GitHub,
                then deploy on streamlit.io/cloud pointing at app.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ──────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Spotify Popularity Predictor",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────
# THEME / CSS  — Spotify-inspired dark UI with glass-card accents
# ──────────────────────────────────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
    html, body, [class*="css"]  { font-family: 'Poppins', sans-serif; }

    .stApp {
        background: radial-gradient(circle at 15% 0%, #1a2e22 0%, #0d0d0d 45%, #050505 100%);
        color: #f2f2f2;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #000000 0%, #0c1f14 100%);
        border-right: 1px solid rgba(29,185,84,0.25);
    }
    section[data-testid="stSidebar"] * { color: #e6e6e6 !important; }

    /* Headings */
    h1, h2, h3 { color: #ffffff !important; font-weight: 700 !important; letter-spacing: -0.5px; }
    h1 { font-weight: 800 !important; }

    /* Hero */
    .hero {
        background: linear-gradient(135deg, #1DB954 0%, #14833b 55%, #0d1f14 100%);
        border-radius: 22px;
        padding: 46px 40px;
        margin-bottom: 28px;
        box-shadow: 0 18px 45px rgba(29,185,84,0.25);
        position: relative;
        overflow: hidden;
    }
    .hero:before {
        content: "";
        position: absolute; top: -60px; right: -60px;
        width: 220px; height: 220px; border-radius: 50%;
        background: rgba(255,255,255,0.08);
    }
    .hero h1 { font-size: 2.6rem; margin: 0 0 6px 0; color: #ffffff !important; }
    .hero p  { font-size: 1.05rem; color: rgba(255,255,255,0.92); margin: 0; max-width: 640px; }

    /* Glass cards */
    .glass-card {
        background: rgba(255,255,255,0.045);
        backdrop-filter: blur(6px);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 22px 24px;
        margin-bottom: 18px;
        transition: transform .15s ease, border-color .15s ease;
    }
    .glass-card:hover { transform: translateY(-2px); border-color: rgba(29,185,84,0.5); }

    .stat-number { font-size: 2.1rem; font-weight: 800; color: #1DB954; margin: 0; }
    .stat-label  { font-size: 0.85rem; color: #b3b3b3; text-transform: uppercase; letter-spacing: 1px; margin: 0; }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #1DB954, #17a44a);
        color: #ffffff; border: none; border-radius: 30px;
        padding: 0.65rem 1.8rem; font-weight: 700; font-size: 1rem;
        box-shadow: 0 8px 20px rgba(29,185,84,0.35);
        transition: transform .12s ease;
    }
    .stButton>button:hover { transform: scale(1.03); }

    /* Sliders */
    .stSlider [data-baseweb="slider"] > div > div { background: #1DB954 !important; }

    /* Tabs */
    button[data-baseweb="tab"] { font-weight: 600; color: #b3b3b3; }
    button[data-baseweb="tab"][aria-selected="true"] { color: #1DB954 !important; }

    /* Song row */
    .song-row {
        display: flex; align-items: center; gap: 14px;
        background: rgba(255,255,255,0.035);
        border-radius: 12px; padding: 12px 16px; margin-bottom: 8px;
        border-left: 3px solid #1DB954;
    }
    .song-badge {
        width: 42px; height: 42px; border-radius: 10px;
        background: linear-gradient(135deg, #1DB954, #0d1f14);
        display: flex; align-items: center; justify-content: center;
        font-size: 1.2rem; flex-shrink: 0;
    }

    footer, #MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

FEATURE_COLS = ["tempo", "energy", "danceability", "loudness", "liveness",
                 "valence", "duration", "acousticness", "speechiness", "year"]

FEATURE_META = {
    "tempo":        ("Tempo (BPM)",       40, 210, 120),
    "energy":       ("Energy",            0, 100, 70),
    "danceability": ("Danceability",      0, 100, 65),
    "loudness":     ("Loudness (dB)",     -25, 0, -5),
    "liveness":     ("Liveness",          0, 100, 15),
    "valence":      ("Valence (mood)",    0, 100, 55),
    "duration":     ("Duration (sec)",    120, 420, 220),
    "acousticness": ("Acousticness",      0, 100, 15),
    "speechiness":  ("Speechiness",       0, 60, 8),
}

# ──────────────────────────────────────────────────────────────────────────
# DATA + MODEL (cached)
# ──────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/top10s.csv")
    df = df.rename(columns={
        "Unnamed: 0": "id", "top genre": "genre", "bpm": "tempo", "nrgy": "energy",
        "dnce": "danceability", "dB": "loudness", "live": "liveness", "val": "valence",
        "dur": "duration", "acous": "acousticness", "spch": "speechiness", "pop": "popularity"
    })
    df = df.drop(columns=["id"])
    df = df[(df["popularity"] > 0) & (df["loudness"] > -30)].reset_index(drop=True)

    genre_counts = df["genre"].value_counts()
    common_genres = genre_counts[genre_counts >= 8].index
    df["genre_grouped"] = df["genre"].where(df["genre"].isin(common_genres), "other")
    return df


@st.cache_resource
def train_models(df):
    work = pd.get_dummies(df, columns=["genre_grouped"], prefix="genre")
    genre_dummy_cols = [c for c in work.columns if c.startswith("genre_")]
    feature_cols = FEATURE_COLS + genre_dummy_cols

    X = work[feature_cols]
    y = work["popularity"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_s, X_test_s = X_train.copy(), X_test.copy()
    X_train_s[FEATURE_COLS] = scaler.fit_transform(X_train[FEATURE_COLS])
    X_test_s[FEATURE_COLS] = scaler.transform(X_test[FEATURE_COLS])

    candidates = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=10.0, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=6, min_samples_leaf=2, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=3, random_state=42),
    }

    metrics, fitted = {}, {}
    for name, model in candidates.items():
        model.fit(X_train_s, y_train)
        preds = model.predict(X_test_s)
        metrics[name] = {
            "RMSE": round(mean_squared_error(y_test, preds) ** 0.5, 3),
            "MAE": round(mean_absolute_error(y_test, preds), 3),
            "R2": round(r2_score(y_test, preds), 3),
        }
        fitted[name] = model

    best_name = min(metrics, key=lambda k: metrics[k]["RMSE"])

    # Nearest-neighbor index for "closest songs" feature (on full scaled dataset)
    X_full_s = X.copy()
    X_full_s[FEATURE_COLS] = StandardScaler().fit(X[FEATURE_COLS]).transform(X[FEATURE_COLS])
    nn = NearestNeighbors(n_neighbors=6).fit(X_full_s)

    return {
        "metrics": metrics, "fitted": fitted, "best_name": best_name,
        "feature_cols": feature_cols, "genre_dummy_cols": genre_dummy_cols,
        "scaler": scaler, "nn": nn, "X_full_for_nn": X_full_s, "work": work,
    }


df = load_data()
bundle = train_models(df)
GENRES = sorted([c.replace("genre_", "") for c in bundle["genre_dummy_cols"]])

# ──────────────────────────────────────────────────────────────────────────
# SIDEBAR NAV
# ──────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🎧 Spotify ML")
st.sidebar.markdown("**Popularity Predictor**")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview", "🔍 Explore the Data", "🎯 Predict a Song's Popularity", "📊 Model Performance"],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")
st.sidebar.caption(f"Dataset: {len(df)} songs · {df['artist'].nunique()} artists · 2010–2019")
st.sidebar.caption("Built with Streamlit · scikit-learn · Plotly")

# ──────────────────────────────────────────────────────────────────────────
# PAGE: OVERVIEW
# ──────────────────────────────────────────────────────────────────────────
if page == "🏠 Overview":
    st.markdown("""
    <div class="hero">
        <h1>🎧 Spotify Popularity Predictor</h1>
        <p>Explore ten years of Spotify's biggest hits (2010–2019) and use a
        machine-learning model — trained on tempo, energy, danceability,
        loudness, and more — to predict how popular a song will be.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    stats = [
        (f"{len(df)}", "Songs Analyzed", c1),
        (f"{df['artist'].nunique()}", "Unique Artists", c2),
        (f"{len(GENRES)}", "Genre Groups", c3),
        (f"{df['popularity'].mean():.0f}", "Avg. Popularity", c4),
    ]
    for val, label, col in stats:
        with col:
            st.markdown(f"""
            <div class="glass-card" style="text-align:center;">
                <p class="stat-number">{val}</p>
                <p class="stat-label">{label}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("### 🏆 Most Charted Artists")
    top_artists = df["artist"].value_counts().head(8).reset_index()
    top_artists.columns = ["artist", "songs"]
    fig = px.bar(
        top_artists, x="songs", y="artist", orientation="h",
        color="songs", color_continuous_scale=["#0d1f14", "#1DB954"],
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font_color="#f2f2f2", yaxis={"categoryorder": "total ascending"},
        coloraxis_showscale=False, height=380, margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="glass-card">
    <b>💡 About this project</b><br>
     ML project. A regression pipeline was
    trained on Spotify's year-end Top 10 songs (2010–2019) to see how well
    audio features alone predict a track's popularity score. Head to
    <b>🎯 Predict a Song's Popularity</b> to try it yourself.
    </div>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────
# PAGE: EXPLORE
# ──────────────────────────────────────────────────────────────────────────
elif page == "🔍 Explore the Data":
    st.markdown("## 🔍 Explore the Data")

    tab1, tab2, tab3 = st.tabs(["Correlations", "Feature vs Popularity", "Distributions"])

    with tab1:
        corr = df[FEATURE_COLS + ["popularity"]].corr()
        fig = px.imshow(
            corr, text_auto=".2f", color_continuous_scale="Greens",
            aspect="auto",
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="#f2f2f2", height=520, margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Correlation between every audio feature and popularity. Values near ±1 are strong relationships — as you'll see, most are weak.")

    with tab2:
        feat = st.selectbox("Choose a feature", FEATURE_COLS, index=1)
        fig = px.scatter(
            df, x=feat, y="popularity", color="genre_grouped",
            opacity=0.65, trendline="ols",
            color_discrete_sequence=px.colors.sequential.Greens_r,
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="#f2f2f2", height=500, legend_title_text="Genre",
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        colA, colB = st.columns(2)
        with colA:
            fig = px.histogram(df, x="popularity", nbins=25, color_discrete_sequence=["#1DB954"])
            fig.update_layout(
                title="Popularity Distribution", plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)", font_color="#f2f2f2", height=380,
            )
            st.plotly_chart(fig, use_container_width=True)
        with colB:
            top_genres = df["genre_grouped"].value_counts().head(10).reset_index()
            top_genres.columns = ["genre", "count"]
            fig = px.pie(top_genres, names="genre", values="count", hole=0.55,
                         color_discrete_sequence=px.colors.sequential.Greens_r)
            fig.update_layout(
                title="Top Genres", paper_bgcolor="rgba(0,0,0,0)",
                font_color="#f2f2f2", height=380,
            )
            st.plotly_chart(fig, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────
# PAGE: PREDICT
# ──────────────────────────────────────────────────────────────────────────
elif page == "🎯 Predict a Song's Popularity":
    st.markdown("## 🎯 Predict a Song's Popularity")
    st.caption("Dial in a song's audio profile and see what the model predicts.")

    left, right = st.columns([1.1, 1])

    with left:
        st.markdown("#### 🎛️ Audio Profile")
        inputs = {}
        c1, c2 = st.columns(2)
        cols_cycle = [c1, c2]
        for i, key in enumerate(FEATURE_COLS[:-1]):  # all except 'year'
            label, lo, hi, default = FEATURE_META[key]
            with cols_cycle[i % 2]:
                inputs[key] = st.slider(label, lo, hi, default)

        year = st.slider("Release Year", 2010, 2019, 2017)
        genre = st.selectbox("Genre", GENRES, index=GENRES.index("dance pop") if "dance pop" in GENRES else 0)
        inputs["year"] = year

        predict_clicked = st.button("🔮 Predict Popularity", use_container_width=True)

    with right:
        st.markdown("#### 🎚️ Result")
        if predict_clicked:
            model = bundle["fitted"][bundle["best_name"]]
            row = {c: 0 for c in bundle["feature_cols"]}
            for k, v in inputs.items():
                row[k] = v
            genre_col = f"genre_{genre}"
            if genre_col in row:
                row[genre_col] = 1

            X_input = pd.DataFrame([row])[bundle["feature_cols"]]
            X_input_scaled = X_input.copy()
            X_input_scaled[FEATURE_COLS] = bundle["scaler"].transform(X_input[FEATURE_COLS])

            pred = float(np.clip(model.predict(X_input_scaled)[0], 0, 100))

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pred,
                number={"suffix": " / 100", "font": {"color": "#ffffff", "size": 42}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#f2f2f2"},
                    "bar": {"color": "#1DB954"},
                    "bgcolor": "rgba(0,0,0,0)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 40], "color": "rgba(214,39,40,0.35)"},
                        {"range": [40, 70], "color": "rgba(255,193,7,0.3)"},
                        {"range": [70, 100], "color": "rgba(29,185,84,0.35)"},
                    ],
                },
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", font_color="#f2f2f2",
                height=320, margin=dict(l=20, r=20, t=30, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

            if pred >= 70:
                msg = "🔥 Hit potential — this profile resembles Spotify's biggest chart-toppers."
            elif pred >= 40:
                msg = "🎵 Solid — a respectable mid-chart profile."
            else:
                msg = "🌱 Niche — this profile is closer to lower-charting or deep-cut tracks."
            st.markdown(f"""<div class="glass-card">{msg}<br>
            <span style="color:#b3b3b3;font-size:0.85rem;">
            Predicted with <b>{bundle['best_name']}</b>, the best-performing tuned model
            (test R² = {bundle['metrics'][bundle['best_name']]['R2']}). Audio features are
            weak predictors of popularity on their own — treat this as a rough
            estimate, not a certainty.
            </span></div>""", unsafe_allow_html=True)

            # Closest matching real songs (uses the same input row + scaler as prediction)
            st.markdown("#### 🎶 Closest Matching Real Songs")
            _, idxs = bundle["nn"].kneighbors(X_input_scaled[bundle["feature_cols"]].values)
            neighbor_rows = bundle["work"].iloc[idxs[0][:5]]
            for _, r in neighbor_rows.iterrows():
                st.markdown(f"""
                <div class="song-row">
                    <div class="song-badge">🎵</div>
                    <div>
                        <b>{r['title']}</b> — {r['artist']}<br>
                        <span style="color:#b3b3b3;font-size:0.82rem;">
                        {r['genre']} · {r['year']} · popularity {r['popularity']}
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="glass-card" style="text-align:center; padding:60px 20px;">
            <span style="font-size:2.2rem;">🎚️</span><br><br>
            Adjust the sliders and hit <b>Predict Popularity</b> to see the result.
            </div>
            """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────
# PAGE: MODEL PERFORMANCE
# ──────────────────────────────────────────────────────────────────────────
elif page == "📊 Model Performance":
    st.markdown("## 📊 Model Performance")
    st.caption("Four regression models, tuned and evaluated on a held-out test set.")

    metrics_df = pd.DataFrame(bundle["metrics"]).T.reset_index().rename(columns={"index": "Model"})
    metrics_df = metrics_df.sort_values("RMSE")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(
            metrics_df, x="Model", y=["RMSE", "MAE"], barmode="group",
            color_discrete_sequence=["#1DB954", "#0d1f14"],
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="#f2f2f2", height=400, legend_title_text="",
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(
            metrics_df, x="Model", y="R2",
            color="R2", color_continuous_scale=["#d62728", "#1DB954"],
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="#f2f2f2", height=400, coloraxis_showscale=False,
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""
    <div class="glass-card">
    <b>🏆 Best model: {bundle['best_name']}</b><br>
    RMSE = {bundle['metrics'][bundle['best_name']]['RMSE']} ·
    MAE = {bundle['metrics'][bundle['best_name']]['MAE']} ·
    R² = {bundle['metrics'][bundle['best_name']]['R2']}
    <br><br>
    <span style="color:#b3b3b3;font-size:0.88rem;">
    All models land in a narrow performance band — evidence that audio
    features alone have a low ceiling for predicting popularity among songs
    that already charted. See the project report for the full analysis.
    </span>
    </div>
    """, unsafe_allow_html=True)

    st.dataframe(metrics_df.set_index("Model"), use_container_width=True)