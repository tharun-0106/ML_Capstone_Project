"""Unified Gradio Application for Machine Learning Capstone (Review 1 & Review 2).

Showcasing all three project tracks:
  1. Regression Track: 10 algorithms predicting continuous Cocoa Ratings
  2. Classification Track: 10 algorithms (Part A + B) predicting Mobile Price Tiers
  3. Clustering Track: Unsupervised K-Means & Hierarchical grouping on 2,000 digits
  4. Executive Hub & Benchmark Lab: Metrics, visualizations, and viva story arc
"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]  # app/ and workspace root

import gradio as gr
import joblib
import numpy as np
import pandas as pd

from src import clustering as clu
from src import config as C
from src import data_loader as dl
from src import regression as reg_mod
from ui import components as ui

# ------------------------------------------------------------------ Load Artifacts
if not (C.MODELS_DIR / "best_classifier.joblib").exists() or \
   not (C.MODELS_DIR / "clustering.joblib").exists():
    raise SystemExit("Models not found - run `python train_pipeline.py` first.")

if not (C.MODELS_DIR / "best_regressor.joblib").exists():
    reg_mod.train_and_evaluate_all()

CLF = joblib.load(C.MODELS_DIR / "best_classifier.joblib")
CLU = joblib.load(C.MODELS_DIR / "clustering.joblib")
REG = joblib.load(C.MODELS_DIR / "best_regressor.joblib")
RESULTS = json.loads((C.REPORTS_DIR / "results.json").read_text())

DF_CLF = dl.load_classification()
DF_REG = dl.load_regression()
X_RAW, Y_TRUE, _ = dl.load_mfeat()
PIX = np.load(dl.CACHE_PATH, allow_pickle=True)["pix"]

CLF_FEATURES = CLF["features"]
BAND_LABELS = [C.PRICE_LABELS[i] for i in range(4)]
CSS = (HERE / "ui" / "style.css").read_text(encoding="utf-8")


# ------------------------------------------------------------------ Track 1: Regression Logic
REG_FEATURES = REG["feature_names"]
LOCATIONS = ["U.S.A.", "France", "Canada", "U.K.", "Italy", "Ecuador", "Madagascar", "Germany", "Belgium", "Switzerland", "Other"]
ORIGINS = ["Venezuela", "Ecuador", "Peru", "Madagascar", "Dominican Republic", "Brazil", "Nicaragua", "Bolivia", "Belize", "Other"]
BEAN_TYPES = ["Criollo", "Trinitario", "Forastero", "Blend", "Unknown", "Other"]

REG_PRESETS = {
    "Artisanal Madagascar (72% Single-Origin)": {
        "cocoa": 72.0, "year": 2016, "maker": "Madagascar", "origin": "Madagascar", "bean": "Criollo"
    },
    "Dark Chocolate Ecuador (85%)": {
        "cocoa": 85.0, "year": 2015, "maker": "U.S.A.", "origin": "Ecuador", "bean": "Forastero"
    },
    "Classic Swiss Blend (70%)": {
        "cocoa": 70.0, "year": 2016, "maker": "Switzerland", "origin": "Venezuela", "bean": "Trinitario"
    },
    "Mass Market Confectionery (55%)": {
        "cocoa": 55.0, "year": 2008, "maker": "Other", "origin": "Other", "bean": "Unknown"
    },
}


def predict_cocoa_rating(cocoa_pct, review_year, maker_loc, bean_orig, bean_type):
    """Predict chocolate bar rating from features using trained Ridge regressor."""
    row = {f: 0.0 for f in REG_FEATURES}
    if "cocoa percent" in row:
        row["cocoa percent"] = float(cocoa_pct)
    if "review date" in row:
        row["review date"] = float(review_year)
    if "ref" in row:
        row["ref"] = float(DF_REG["ref"].median())

    origin_match = 1.0 if (maker_loc == bean_orig and maker_loc != "Other") else 0.0
    if "origin_matches_location" in row:
        row["origin_matches_location"] = origin_match

    # Set one-hot columns
    maker_col = f"company location_{maker_loc}"
    if maker_col in row:
        row[maker_col] = 1.0

    orig_col = f"broad bean origin_{bean_orig}"
    if orig_col in row:
        row[orig_col] = 1.0

    bean_col = f"bean type_{bean_type}"
    if bean_col in row:
        row[bean_col] = 1.0

    # Composite target encoded feature fallback
    if "target_encoded_feature" in row:
        match_ratings = DF_REG[(DF_REG["broad bean origin"] == bean_orig) & (DF_REG["cocoa percent"] == cocoa_pct)]
        if len(match_ratings) > 0:
            row["target_encoded_feature"] = float(match_ratings["rating"].mean())
        else:
            row["target_encoded_feature"] = float(DF_REG["rating"].median())

    df_sample = pd.DataFrame([row])[REG_FEATURES]
    X_scaled = REG["scaler"].transform(df_sample)
    pred_val = float(REG["pipeline"].named_steps["regressor"].predict(X_scaled)[0])
    pred_val = max(1.0, min(5.0, pred_val))  # clip to legitimate rating bounds

    return ui.chocolate_rating_card(pred_val, cocoa_pct, maker_loc, bean_orig)


def apply_reg_preset(name):
    p = REG_PRESETS[name]
    return p["cocoa"], p["year"], p["maker"], p["origin"], p["bean"]


def random_reg_sample():
    sample = DF_REG.sample(1).iloc[0]
    m = sample["company location"] if sample["company location"] in LOCATIONS else "Other"
    o = sample["broad bean origin"] if sample["broad bean origin"] in ORIGINS else "Other"
    b = sample["bean type"] if sample["bean type"] in BEAN_TYPES else "Other"
    return float(sample["cocoa percent"]), int(sample["review date"]), m, o, b


# ------------------------------------------------------------------ Track 2: Classification Logic
BINARY = {"blue": "Bluetooth", "dual_sim": "Dual SIM", "four_g": "4G", "three_g": "3G",
          "touch_screen": "Touch screen", "wifi": "Wi-Fi"}

GROUPS = {
    "Memory and Speed": [
        ("ram", "RAM (MB)", 1), ("int_memory", "Storage (GB)", 1),
        ("clock_speed", "CPU Clock (GHz)", 0.1), ("n_cores", "CPU Cores", 1)
    ],
    "Display Specifications": [
        ("px_height", "Resolution Height (px)", 1), ("px_width", "Resolution Width (px)", 1),
        ("sc_h", "Screen Height (cm)", 1), ("sc_w", "Screen Width (cm)", 1)
    ],
    "Battery & Dimensions": [
        ("battery_power", "Battery Capacity (mAh)", 1), ("talk_time", "Talk Time (hours)", 1),
        ("mobile_wt", "Weight (g)", 1), ("m_dep", "Thickness (cm)", 0.1)
    ],
    "Camera System": [
        ("pc", "Main Camera (MP)", 1), ("fc", "Front Camera (MP)", 1)
    ],
}
NUMERIC = [f for g in GROUPS.values() for f, _, _ in g]

CLF_PRESETS = {
    "Budget Phone": dict(ram=700, int_memory=8, clock_speed=1.0, n_cores=4, px_height=400, px_width=800,
                         sc_h=10, sc_w=4, battery_power=900, talk_time=8, mobile_wt=170, m_dep=0.8,
                         pc=5, fc=2, conn=["Dual SIM", "3G"]),
    "Mid-Range": dict(ram=1800, int_memory=32, clock_speed=1.8, n_cores=6, px_height=800, px_width=1300,
                      sc_h=13, sc_w=6, battery_power=1300, talk_time=12, mobile_wt=150, m_dep=0.5,
                      pc=12, fc=5, conn=["Bluetooth", "Dual SIM", "4G", "3G", "Touch screen", "Wi-Fi"]),
    "Upper Mid-Range": dict(ram=2400, int_memory=48, clock_speed=2.2, n_cores=8, px_height=900,
                            px_width=1400, sc_h=15, sc_w=7, battery_power=1400, talk_time=14,
                            mobile_wt=140, m_dep=0.4, pc=16, fc=8,
                            conn=["Bluetooth", "Dual SIM", "4G", "3G", "Touch screen", "Wi-Fi"]),
    "Flagship Monster": dict(ram=3800, int_memory=64, clock_speed=2.8, n_cores=8, px_height=1700, px_width=1900,
                             sc_h=17, sc_w=8, battery_power=1900, talk_time=18, mobile_wt=130, m_dep=0.3,
                             pc=20, fc=16, conn=["Bluetooth", "Dual SIM", "4G", "3G", "Touch screen", "Wi-Fi"]),
}


def predict_price(*vals):
    nums, conn = vals[:-1], vals[-1] or []
    row = dict(zip(NUMERIC, map(float, nums)))
    row.update({f: float(lbl in conn) for f, lbl in BINARY.items()})
    proba = CLF["model"].predict_proba(pd.DataFrame([row])[CLF_FEATURES])[0]
    return ui.price_result(proba, BAND_LABELS)


def apply_clf_preset(name):
    p = CLF_PRESETS[name]
    return [float(p[f]) for f in NUMERIC] + [p["conn"], f"Loaded preset: **{name}**."]


def random_phone():
    r = DF_CLF.sample(1).iloc[0]
    conn = [lbl for f, lbl in BINARY.items() if r[f] == 1]
    note = f"Loaded real phone from test dataset. Ground truth band: **{C.PRICE_LABELS[int(r[C.TARGET_COL])]}**."
    return [float(r[f]) for f in NUMERIC] + [conn, note]


# ------------------------------------------------------------------ Track 3: Clustering Logic
ALGOS = {"K-Means": None, "Ward": "ward", "Complete": "complete", "Average": "average", "Single": "single"}
ALGO_NOTES = {
    "K-Means": "Partitions 2,000 digits into 10 Voronoi clusters by minimizing intra-cluster inertia (sum of squared Euclidean distances).",
    "Ward": "Hierarchical agglomeration minimizing the total within-cluster variance at each sequential merge step. Highest ARI.",
    "Complete": "Hierarchical agglomeration merging clusters based on maximum pairwise distance (compact sphere clusters).",
    "Average": "Hierarchical agglomeration merging clusters based on average pairwise distance across all member elements.",
    "Single": "Hierarchical agglomeration merging closest individual samples. Prone to the well-documented 'chaining' anomaly.",
}


def labels_for(algo):
    lk = ALGOS[algo]
    return CLU["kmeans"].labels_ if lk is None else CLU["agg_labels"][lk]


def metrics_for(algo):
    name = "K-Means" if ALGOS[algo] is None else f"Agglomerative ({ALGOS[algo]})"
    return CLU["metrics"].set_index("Algorithm").loc[name]


MEAN_IMGS = {a: [PIX[labels_for(a) == c].mean(0).reshape(16, 15) for c in np.unique(labels_for(a))]
             for a in ALGOS}
MAPPINGS = {a: clu.map_clusters_to_digits(Y_TRUE, labels_for(a))[0] for a in ALGOS}


def cluster_page(algo, view, color, highlight=None):
    coords = CLU["tsne_xy"] if view == "t-SNE" and CLU["tsne_xy"] is not None else CLU["pca_xy"]
    method = "t-SNE" if coords is CLU["tsne_xy"] else "PCA"
    if color == "True Digits":
        lab, title = Y_TRUE, f"{method} Projection — Coloured by Ground Truth Digits (Post-Hoc Validation)"
    else:
        lab, title = labels_for(algo), f"{method} Projection — {algo} Discovered Clusters (Unsupervised)"
    fig = ui.cluster_scatter(coords, lab, Y_TRUE, method, title, highlight)
    side = ui.metric_tiles(metrics_for(algo)) + f'<p style="color:var(--text-muted);font-size:0.9rem;margin-top:10px;">{ALGO_NOTES[algo]}</p>'
    return fig, side, ui.cluster_gallery(MEAN_IMGS[algo], MAPPINGS[algo])


def inspect_digit(idx, algo, view, color):
    idx = int(idx)
    km_c = int(CLU["kmeans"].labels_[idx])
    lk = CLU["best_linkage"]
    ag_c = int(CLU["agg_labels"][lk][idx])
    card = ui.inspect_card(
        PIX[idx].reshape(16, 15), idx,
        km_c, int(MAPPINGS["K-Means"].loc[km_c, "Majority digit"]),
        ag_c, int(MAPPINGS["Ward"].loc[ag_c, "Majority digit"]),
        lk, int(Y_TRUE[idx])
    )
    fig, side, gal = cluster_page(algo, view, color, idx)
    return card, fig


def assign_new_digits(file, n):
    if file is not None:
        path = str(file)
        X_new = np.loadtxt(path, ndmin=2, delimiter="," if path.endswith(".csv") else None)
        src = "uploaded file"
    else:
        rng = np.random.default_rng()
        X_new = X_RAW[rng.choice(len(X_RAW), int(n), replace=False)]
        src = f"{int(n)} random samples from dataset"

    if X_new.shape[1] != X_RAW.shape[1]:
        raise gr.Error(f"Each row requires {X_RAW.shape[1]} features (all 6 mfeat feature sets), but received {X_new.shape[1]}.")

    Xs = CLU["scaler"].transform(X_new)
    if "embed_projector" in CLU:
        from sklearn.preprocessing import normalize
        Zs = normalize(CLU["embed_projector"].predict(Xs))
    else:
        Zs = Xs
    lk = CLU["best_linkage"]
    km = CLU["kmeans"].predict(Zs)
    ag = clu.nearest_centroid(Zs, CLU["agg_centroids"][lk])

    out = pd.DataFrame({
        "Sample #": np.arange(1, len(Xs) + 1),
        "K-Means Group": km,
        "Likely Digit (K-Means)": [int(MAPPINGS["K-Means"].loc[c, "Majority digit"]) for c in km],
        f"Agglomerative ({lk}) Group": ag,
        "Likely Digit (Agglomerative)": [int(MAPPINGS["Ward"].loc[c, "Majority digit"]) for c in ag],
    })

    xy = CLU["pca"].transform(Zs)
    fig = ui.cluster_scatter(CLU["pca_xy"], CLU["kmeans"].labels_, Y_TRUE, "PCA", "Batch Digit Assignment Projection")
    import plotly.graph_objects as go
    fig.add_trace(go.Scatter(
        x=xy[:, 0], y=xy[:, 1], mode="markers+text", name="New Digit",
        text=[f"#{i}" for i in range(1, len(xy) + 1)], textposition="top center",
        marker=dict(size=14, color="rgba(0,0,0,0)", line=dict(width=3, color="#EF4444"))
    ))
    return out, fig, f"Successfully clustered {len(out)} samples from {src}."


def example_file():
    path = C.REPORTS_DIR / "sample_mfeat_rows.csv"
    np.savetxt(path, X_RAW[np.random.default_rng(42).choice(2000, 5, replace=False)], delimiter=",", fmt="%.6g")
    return str(path)


def fig_path(name):
    p = C.FIG_DIR / name
    return str(p) if p.exists() else None


# ------------------------------------------------------------------ Gradio App Layout
THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Plus Jakarta Sans"), gr.themes.GoogleFont("Inter"), "sans-serif"],
).set(
    body_background_fill="#F8FAFC",
    body_text_color="#0F172A",
    body_text_color_subdued="#475569",
    block_background_fill="#FFFFFF",
    block_border_width="1px",
    block_border_color="#CBD5E1",
    block_label_text_color="#0F172A",
    block_title_text_color="#0F172A",
    button_primary_background_fill="#4F46E5",
    button_primary_background_fill_hover="#4338CA",
    button_primary_text_color="#FFFFFF",
)

with gr.Blocks(title="Machine Learning Multi-Track Intelligence Studio") as demo:
    # Header Hero
    gr.HTML(ui.executive_hero())

    with gr.Tabs(elem_classes="tabs") as main_tabs:
        # ==========================================================
        # TAB 1: EXECUTIVE HUB
        # ==========================================================
        with gr.Tab("🏠 Executive Dashboard", id="hub"):
            with gr.Row():
                with gr.Column(scale=2, elem_classes="saas-card"):
                    gr.HTML('<div class="section-title">Machine Learning Architecture & Methodology</div>'
                            '<p class="section-sub">End-to-End Workflow: Problem Formulation → Data Audit → Preprocessing & Leakage Prevention → Model Training & Evaluation → Hyperparameter Tuning → Deployment.</p>')
                    gr.HTML(ui.viva_cheat_sheet())
                with gr.Column(scale=1, elem_classes="saas-card"):
                    gr.HTML('<div class="section-title">Champion Model Scorecard</div>')
                    gr.Markdown(f"""
- **Track 1 (Regression):**
  - **Best Model:** `{REG['model_name']}`
  - **Test R²:** `{REG['comparison'][0]['R2']:.4f}` | **RMSE:** `{REG['comparison'][0]['RMSE']:.4f}`
  - **5-Fold CV R²:** `{REG['top2_cv'][REG['model_name']]['mean_cv_r2']:.4f} ± {REG['top2_cv'][REG['model_name']]['std_cv_r2']:.4f}`
- **Track 2 (Classification):**
  - **Best Model:** `{CLF['name']}`
  - **Test Accuracy:** `{CLF['metrics']['Accuracy']*100:.1f}%` | **Weighted F1:** `{CLF['metrics']['F1 (weighted)']:.4f}`
  - **5-Fold CV F1:** `{RESULTS['classification']['tuning'][0]['CV F1 (5-fold)']:.4f}`
- **Track 3 (Clustering):**
  - **Optimal Clustering:** `K-Means (k=10)` & `Ward Linkage`
  - **Adjusted Rand Index (ARI):** `{float(metrics_for('K-Means')['ARI']):.3f}`
  - **Silhouette Score:** `{float(metrics_for('K-Means')['Silhouette']):.3f}`
""")
                    gr.HTML('<div class="section-title" style="margin-top:16px;">Quick Presets & Testing</div>')
                    gr.Markdown("Click into any of the dedicated track tabs above to test real-time predictions, explore 2D manifold scatter plots, or view the consolidated benchmark matrix.")

        # ==========================================================
        # TAB 2: REGRESSION TRACK (10 ALGORITHMS)
        # ==========================================================
        with gr.Tab("🍫 Cocoa Rating Regressor (10 Models)", id="regression"):
            with gr.Row():
                with gr.Column(scale=3):
                    gr.HTML('<div class="section-title">Live Chocolate Bar Rating Predictor</div>'
                            '<p class="section-sub">Adjust bean attributes to predict the continuous rating (1.00 - 5.00) using the tuned Ridge Regressor.</p>')

                    with gr.Row(elem_classes="preset-row"):
                        reg_preset_btns = [gr.Button(n, elem_classes="chip-btn", size="sm", scale=0) for n in REG_PRESETS]
                        rand_reg_btn = gr.Button("🎲 Random Sample from Dataset", elem_classes="chip-btn", size="sm", scale=0)

                    with gr.Row():
                        with gr.Column(elem_classes="saas-card"):
                            reg_cocoa = gr.Slider(50.0, 100.0, value=70.0, step=0.5, label="Cocoa Percentage (%)")
                            reg_year = gr.Slider(2006, 2017, value=2016, step=1, label="Review Year")
                        with gr.Column(elem_classes="saas-card"):
                            reg_maker = gr.Dropdown(LOCATIONS, value="Madagascar", label="Maker Location (Country)")
                            reg_origin = gr.Dropdown(ORIGINS, value="Madagascar", label="Broad Bean Origin")
                            reg_bean = gr.Dropdown(BEAN_TYPES, value="Criollo", label="Bean Variety / Type")

                with gr.Column(scale=2):
                    reg_result = gr.HTML(container=False, padding=False)
                    gr.Markdown("*Proxy Feature:* When Maker Country equals Bean Origin, `origin_matches_location = 1` simulates single-origin bean-to-bar craftsmanship.")

            # Events for regression live prediction
            reg_inputs = [reg_cocoa, reg_year, reg_maker, reg_origin, reg_bean]
            for comp in reg_inputs:
                comp.change(predict_cocoa_rating, reg_inputs, reg_result, show_progress="hidden")

            for b in reg_preset_btns:
                b.click(apply_reg_preset, gr.State(b.value), reg_inputs).then(predict_cocoa_rating, reg_inputs, reg_result)
            rand_reg_btn.click(random_reg_sample, None, reg_inputs).then(predict_cocoa_rating, reg_inputs, reg_result)

            with gr.Accordion("📈 Regression Analysis: All 10 Models, Fit Plots & Tuning", open=True):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Plot(ui.regression_comparison_bar(REG["comparison"]))
                    with gr.Column(scale=1):
                        gr.Plot(ui.regression_fit_plot(REG["test_actual"], REG["test_pred"]))
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Plot(ui.regression_residuals_plot(REG["test_pred"], REG["residuals"]))
                    with gr.Column(scale=1):
                        gr.Dataframe(
                            pd.DataFrame(REG["comparison"]),
                            interactive=False,
                            label="Regression Performance Summary (R², RMSE, MAE) Ranked by R²"
                        )
                        gr.Dataframe(
                            pd.DataFrame(REG["tuning"]),
                            interactive=False,
                            label="Hyperparameter Tuning (Random Forest & Gradient Boosting)"
                        )

        # ==========================================================
        # TAB 3: CLASSIFICATION TRACK (10 ALGORITHMS)
        # ==========================================================
        with gr.Tab("📱 Mobile Price Classifier (10 Models)", id="classification"):
            with gr.Row():
                with gr.Column(scale=3):
                    gr.HTML('<div class="section-title">Live Mobile Price Tier Predictor</div>'
                            '<p class="section-sub">Adjust 14 hardware specifications to predict phone price band with class probabilities.</p>')

                    with gr.Row(elem_classes="preset-row"):
                        clf_preset_btns = [gr.Button(n, elem_classes="chip-btn", size="sm", scale=0) for n in CLF_PRESETS]
                        rand_clf_btn = gr.Button("🎲 Random Phone from Dataset", elem_classes="chip-btn", size="sm", scale=0)

                    clf_inputs = {}
                    group_items = list(GROUPS.items())
                    with gr.Row():
                        for gname, fields in group_items[:2]:
                            with gr.Column(elem_classes="saas-card"):
                                gr.HTML(f'<div class="section-title" style="font-size:1.05rem;">{gname}</div>')
                                for f, label, step in fields:
                                    clf_inputs[f] = gr.Slider(
                                        float(DF_CLF[f].min()), float(DF_CLF[f].max()),
                                        value=float(DF_CLF[f].median()), step=step, label=label
                                    )
                    with gr.Row():
                        for gname, fields in group_items[2:]:
                            with gr.Column(elem_classes="saas-card"):
                                gr.HTML(f'<div class="section-title" style="font-size:1.05rem;">{gname}</div>')
                                for f, label, step in fields:
                                    clf_inputs[f] = gr.Slider(
                                        float(DF_CLF[f].min()), float(DF_CLF[f].max()),
                                        value=float(DF_CLF[f].median()), step=step, label=label
                                    )

                    with gr.Column(elem_classes="saas-card"):
                        gr.HTML('<div class="section-title" style="font-size:1.05rem;">Connectivity & Features</div>')
                        clf_conn = gr.CheckboxGroup(
                            list(BINARY.values()),
                            value=["Dual SIM", "4G", "3G", "Touch screen"],
                            show_label=False
                        )

                with gr.Column(scale=2):
                    clf_result = gr.HTML(container=False, padding=False)
                    clf_note = gr.Markdown()

            clf_all_inputs = [clf_inputs[f] for f in NUMERIC] + [clf_conn]
            for comp in clf_all_inputs:
                comp.change(predict_price, clf_all_inputs, clf_result, show_progress="hidden")

            for b in clf_preset_btns:
                b.click(apply_clf_preset, gr.State(b.value), clf_all_inputs + [clf_note]).then(predict_price, clf_all_inputs, clf_result)
            rand_clf_btn.click(random_phone, None, clf_all_inputs + [clf_note]).then(predict_price, clf_all_inputs, clf_result)

            with gr.Accordion("📊 Classification Benchmarks & Tuning (Part A vs Part B)", open=False):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Plot(ui.classification_model_bars(RESULTS["classification"]["final"]))
                    with gr.Column(scale=1):
                        gr.Dataframe(
                            pd.DataFrame(RESULTS["classification"]["final"]),
                            interactive=False,
                            label="Consolidated 10-Algorithm Comparison (Test Set)"
                        )
                gr.Dataframe(
                    pd.DataFrame(RESULTS["classification"]["tuning"]),
                    interactive=False,
                    label="Hyperparameter Tuning Comparison (Before vs After 5-Fold GridSearchCV)"
                )
                with gr.Row():
                    if fig_path("clf_best_confusion.png"):
                        gr.Image(fig_path("clf_best_confusion.png"), label="Best Model Confusion Matrix", show_label=True)
                    if fig_path("clf_best_importance.png"):
                        gr.Image(fig_path("clf_best_importance.png"), label="Feature Importances (RAM Dominance)", show_label=True)

        # ==========================================================
        # TAB 4: CLUSTERING TRACK (K-MEANS & HIERARCHICAL)
        # ==========================================================
        with gr.Tab("🔢 Digit Clustering Studio", id="clustering"):
            with gr.Row():
                clu_algo = gr.Radio(list(ALGOS), value="K-Means", label="Clustering Algorithm", scale=3)
                clu_view = gr.Radio(["PCA", "t-SNE"], value="t-SNE", label="2D Manifold Projection", scale=1)
                clu_color = gr.Radio(["Discovered Groups", "True Digits"], value="Discovered Groups", label="Colour Scheme", scale=2)

            with gr.Row():
                with gr.Column(scale=3, elem_classes="saas-card"):
                    clu_scatter = gr.Plot(show_label=False)
                with gr.Column(scale=2):
                    clu_metrics_html = gr.HTML(container=False, padding=False)

            with gr.Column(elem_classes="saas-card"):
                gr.HTML('<div class="section-title">Cluster Centroid Pixel Reconstructions</div>')
                clu_gallery_html = gr.HTML(container=False, padding=False)

            with gr.Tabs():
                with gr.Tab("🔍 Inspect Individual Sample"):
                    with gr.Row():
                        digit_slider = gr.Slider(0, 1999, value=1250, step=1, label="Select Any Digit Index (0 – 1999)", scale=4)
                        surprise_btn = gr.Button("🎲 Surprise Me", elem_classes="chip-btn", scale=1)
                    digit_inspect_out = gr.HTML(container=False, padding=False)

                with gr.Tab("📥 Cluster New Samples"):
                    gr.Markdown("Upload custom 649-dimensional mfeat feature vectors, or test random samples to predict their cluster assignment and projected 2D location.")
                    with gr.Row():
                        clu_upload = gr.File(label="Upload CSV/TXT (649 columns)", file_types=[".csv", ".txt"])
                        with gr.Column():
                            clu_n_random = gr.Slider(1, 20, value=5, step=1, label="Or generate this many random samples")
                            clu_download_btn = gr.Button("Download Sample File", elem_classes="chip-btn")
                            clu_download_file = gr.File(label="Sample Format", visible=True)
                    clu_assign_btn = gr.Button("Cluster New Samples", variant="primary")
                    clu_assign_status = gr.Markdown()
                    with gr.Row():
                        clu_assign_table = gr.Dataframe(interactive=False)
                        clu_assign_plot = gr.Plot(show_label=False)

                with gr.Tab("📐 Selection of k & Dendrograms"):
                    cl_res = RESULTS["clustering"]
                    gr.Markdown(f"**Elbow & Dendrogram Analysis:** Automatic knee detector found k={cl_res['k_auto_knee']}, silhouette peaked at k={cl_res['k_best_silhouette']}. Domain knowledge of 10 digit classes and marginal inertia drop justify final **k = {cl_res['k_used']}**.")
                    with gr.Row():
                        if fig_path("clu_elbow.png"):
                            gr.Image(fig_path("clu_elbow.png"), label="Inertia vs. k (Elbow Method)")
                        if fig_path("clu_dendrogram_ward.png"):
                            gr.Image(fig_path("clu_dendrogram_ward.png"), label="Hierarchical Dendrogram (Ward Linkage)")

            # Clustering events
            for comp in (clu_algo, clu_view, clu_color):
                comp.change(cluster_page, [clu_algo, clu_view, clu_color], [clu_scatter, clu_metrics_html, clu_gallery_html])

            digit_slider.release(inspect_digit, [digit_slider, clu_algo, clu_view, clu_color], [digit_inspect_out, clu_scatter])
            surprise_btn.click(lambda: int(np.random.randint(0, 2000)), None, digit_slider).then(
                inspect_digit, [digit_slider, clu_algo, clu_view, clu_color], [digit_inspect_out, clu_scatter]
            )

            clu_assign_btn.click(assign_new_digits, [clu_upload, clu_n_random], [clu_assign_table, clu_assign_plot, clu_assign_status])
            clu_download_btn.click(example_file, None, clu_download_file)

        # ==========================================================
        # TAB 5: BENCHMARK LAB
        # ==========================================================
        with gr.Tab("📊 Model Benchmark Lab", id="benchmarks"):
            gr.HTML('<div class="section-title">Multi-Track Model Performance Matrix</div>'
                    '<p class="section-sub">Consolidated benchmark comparison across all 20 evaluated models and clustering linkages.</p>')
            with gr.Row():
                with gr.Column(scale=1, elem_classes="saas-card"):
                    gr.HTML('<div class="section-title" style="font-size:1.1rem;">Track 1: 10 Regression Models</div>')
                    gr.Dataframe(pd.DataFrame(REG["comparison"]), interactive=False)
                with gr.Column(scale=1, elem_classes="saas-card"):
                    gr.HTML('<div class="section-title" style="font-size:1.1rem;">Track 2: 10 Classification Models (Tuned)</div>')
                    gr.Dataframe(pd.DataFrame(RESULTS["classification"]["final"]), interactive=False)
            with gr.Column(elem_classes="saas-card"):
                gr.HTML('<div class="section-title" style="font-size:1.1rem;">Track 3: Clustering Linkages & K-Means (k=10)</div>')
                gr.Dataframe(pd.DataFrame(RESULTS["clustering"]["metrics"]), interactive=False)

    # Initial page loads
    demo.load(predict_cocoa_rating, reg_inputs, reg_result)
    demo.load(predict_price, clf_all_inputs, clf_result)
    demo.load(cluster_page, [clu_algo, clu_view, clu_color], [clu_scatter, clu_metrics_html, clu_gallery_html])
    demo.load(inspect_digit, [digit_slider, clu_algo, clu_view, clu_color], [digit_inspect_out, clu_scatter])

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        theme=THEME,
        css=CSS,
        js="() => { document.documentElement.classList.remove('dark'); document.body.classList.remove('dark'); }"
    )
