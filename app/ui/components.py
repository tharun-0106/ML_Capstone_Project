"""HTML / Plotly building blocks for the Gradio UI. Pure functions, no Gradio state."""
import html
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Modern Design Tokens
INK = "#0F172A"          # Slate 900
MUTED = "#64748B"        # Slate 500
LINE = "#E2E8F0"         # Slate 200
CARD_BG = "#FFFFFF"
PRIMARY = "#4F46E5"      # Indigo 600
PRIMARY_SOFT = "#EEF2FF" # Indigo 50
ACCENT = "#EC4899"       # Pink 500
ACCENT_SOFT = "#FDF2F8"
SUCCESS = "#10B981"      # Emerald 500
AMBER = "#F59E0B"        # Amber 500

# Guideline 7.3 Colourblind-friendly Palettes
# Matplotlib "tab10"
CLUSTER_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
]

# Classification Price Band Palette
BAND_COLORS = ["#10B981", "#3B82F6", "#8B5CF6", "#EC4899"]


# ------------------------------------------------------------------ Plotly Theme Helper
def _base_layout(title="", height=420):
    return dict(
        title=dict(
            text=title,
            font=dict(family="Plus Jakarta Sans, sans-serif", size=15, color=INK, weight=700),
            x=0.01,
            y=0.97
        ),
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=INK, size=12),
        margin=dict(l=40, r=20, t=50, b=40),
        xaxis=dict(
            gridcolor="#F1F5F9",
            zeroline=False,
            tickfont=dict(color=MUTED, size=11),
            title_font=dict(color=INK, size=12)
        ),
        yaxis=dict(
            gridcolor="#F1F5F9",
            zeroline=False,
            tickfont=dict(color=MUTED, size=11),
            title_font=dict(color=INK, size=12)
        )
    )


# ------------------------------------------------------------------ Pixel Digit SVG
def digit_svg(img, size=64, color=INK, cls="", style=""):
    """15x16 mfeat pixel image (values 0-6) -> crisp inline SVG."""
    img = np.asarray(img, float)
    vmax = max(img.max(), 1e-9)
    rects = []
    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            v = img[r, c] / vmax
            if v > 0.04:
                rects.append(
                    f'<rect x="{c}" y="{r}" width="1.02" height="1.02" fill-opacity="{v:.2f}"/>'
                )
    h = size * img.shape[0] / img.shape[1]
    return (
        f'<svg class="{cls}" style="{style}" viewBox="0 0 {img.shape[1]} {img.shape[0]}" '
        f'width="{size}" height="{h:.0f}" fill="{color}" shape-rendering="crispEdges" '
        f'role="img" aria-label="handwritten digit">{"".join(rects)}</svg>'
    )


# ------------------------------------------------------------------ Track 1: Regression Plots
def regression_fit_plot(actual, pred):
    """Plotly scatter plot of Actual vs Predicted rating with y=x line."""
    actual = np.asarray(actual)
    pred = np.asarray(pred)
    mn = min(actual.min(), pred.min()) - 0.1
    mx = max(actual.max(), pred.max()) + 0.1

    fig = go.Figure()
    # 45-degree reference line
    fig.add_trace(go.Scatter(
        x=[mn, mx], y=[mn, mx],
        mode="lines",
        name="Perfect Fit (y = x)",
        line=dict(color="#EF4444", width=2, dash="dash")
    ))
    # Predictions scatter
    fig.add_trace(go.Scatter(
        x=actual, y=pred,
        mode="markers",
        name="Test Predictions",
        marker=dict(
            size=7,
            color=PRIMARY,
            opacity=0.75,
            line=dict(width=1, color="#FFFFFF")
        ),
        hovertemplate="Actual: %{x:.2f}<br>Predicted: %{y:.2f}<extra></extra>"
    ))

    layout = _base_layout("Predicted vs. Actual Rating (Test Set)", height=380)
    layout["xaxis"]["title"] = "Ground Truth Rating"
    layout["yaxis"]["title"] = "Predicted Rating"
    fig.update_layout(**layout)
    return fig


def regression_residuals_plot(pred, residuals):
    """Plotly scatter of Residuals vs Predicted."""
    pred = np.asarray(pred)
    res = np.asarray(residuals)

    fig = go.Figure()
    # Zero line
    fig.add_trace(go.Scatter(
        x=[pred.min() - 0.1, pred.max() + 0.1], y=[0, 0],
        mode="lines",
        name="Zero Error",
        line=dict(color=MUTED, width=1.5, dash="dot")
    ))
    # Residual points
    fig.add_trace(go.Scatter(
        x=pred, y=res,
        mode="markers",
        name="Residuals",
        marker=dict(size=7, color="#F59E0B", opacity=0.75, line=dict(width=1, color="#FFFFFF")),
        hovertemplate="Predicted: %{x:.2f}<br>Residual: %{y:.3f}<extra></extra>"
    ))

    layout = _base_layout("Residual Plot (Errors vs. Predicted)", height=380)
    layout["xaxis"]["title"] = "Fitted / Predicted Values"
    layout["yaxis"]["title"] = "Residual (Actual − Predicted)"
    fig.update_layout(**layout)
    return fig


def regression_comparison_bar(comparison_df):
    """Horizontal bar chart ranking the 10 regression models by R² score."""
    df = pd.DataFrame(comparison_df).sort_values("R2", ascending=True)
    names = df["Algorithm"].tolist()
    r2_vals = df["R2"].tolist()
    rmse_vals = df["RMSE"].tolist()

    colors = [PRIMARY if r > 0.9 else ("#94A3B8" if r > 0 else "#EF4444") for r in r2_vals]

    fig = go.Figure(go.Bar(
        x=r2_vals,
        y=names,
        orientation="h",
        marker_color=colors,
        text=[f"R²={v:.4f}" for v in r2_vals],
        textposition="outside",
        customdata=rmse_vals,
        hovertemplate="%{y}<br>R²: %{x:.6f}<br>RMSE: %{customdata:.4f}<extra></extra>"
    ))

    layout = _base_layout("10 Regression Algorithms Ranked by R² Score", height=400)
    layout["xaxis"]["title"] = "R² Score on Held-Out Test Set"
    layout["xaxis"]["range"] = [-0.1, 1.15]
    fig.update_layout(**layout)
    return fig


def chocolate_rating_card(rating_val, cocoa_pct, maker, origin):
    """Sleek badge card displaying predicted chocolate bar rating."""
    score = float(rating_val)
    if score >= 3.75:
        tier = "Outstanding (Artisanal)"
        color = "#10B981"
        badge_bg = "#ECFDF5"
    elif score >= 3.0:
        tier = "Good / Recommended"
        color = "#3B82F6"
        badge_bg = "#EFF6FF"
    elif score >= 2.0:
        tier = "Average / Mediocre"
        color = "#F59E0B"
        badge_bg = "#FFFBEB"
    else:
        tier = "Unpleasant / Low Quality"
        color = "#EF4444"
        badge_bg = "#FEF2F2"

    stars = "★" * int(round(score)) + "☆" * (5 - int(round(score)))

    return f"""
<div class="result-card-modern" style="border-left: 6px solid {color};">
  <div class="card-kicker">Predicted Cocoa Rating (1.00 – 5.00 Scale)</div>
  <div class="rating-display">
    <span class="rating-number" style="color: {color};">{score:.2f}</span>
    <span class="rating-stars" style="color: {color};">{stars}</span>
  </div>
  <div class="rating-badge" style="background: {badge_bg}; color: {color};">{tier}</div>
  <div class="spec-summary-strip">
    <span><b>Cocoa:</b> {cocoa_pct}%</span>
    <span><b>Maker Location:</b> {html.escape(str(maker))}</span>
    <span><b>Bean Origin:</b> {html.escape(str(origin))}</span>
  </div>
</div>
"""


# ------------------------------------------------------------------ Track 2: Classification Plots
def price_result(proba, labels):
    """Modern interactive price band predictor card."""
    k = int(np.argmax(proba))
    top_label = labels[k]
    color = BAND_COLORS[k]

    rows = ""
    for i, p in enumerate(proba):
        is_top = (i == k)
        c = BAND_COLORS[i]
        rows += f"""
        <div class="prob-row {'is-top' if is_top else ''}">
          <span class="prob-name">{labels[i]}</span>
          <div class="prob-bar-container">
            <div class="prob-bar-fill" style="width: {p*100:.1f}%; background: {c};"></div>
          </div>
          <span class="prob-val">{p*100:.1f}%</span>
        </div>
        """

    return f"""
<div class="result-card-modern" style="border-left: 6px solid {color};">
  <div class="card-kicker">Predicted Mobile Price Category</div>
  <div class="band-title" style="color: {color};">{top_label}</div>
  <div class="conf-badge">Confidence: {proba[k]*100:.1f}%</div>
  <div class="prob-list">
    {rows}
  </div>
</div>
"""


def classification_model_bars(comparison):
    """Horizontal bar chart comparing all 10 classification models."""
    names = [r["Model"] for r in comparison][::-1]
    f1 = [r["F1 (weighted)"] for r in comparison][::-1]
    part = [r["Part"] for r in comparison][::-1]
    acc = [r["Accuracy"] for r in comparison][::-1]

    colors = [PRIMARY if p.startswith("B") else "#94A3B8" for p in part]

    fig = go.Figure(go.Bar(
        x=f1, y=names, orientation="h",
        marker_color=colors,
        text=[f"F1={v:.3f}" for v in f1],
        textposition="outside",
        customdata=acc,
        hovertemplate="%{y}<br>Weighted F1: %{x:.4f}<br>Accuracy: %{customdata:.4f}<extra></extra>"
    ))

    layout = _base_layout("All 10 Classification Algorithms (Test Set F1-Score)", height=400)
    layout["xaxis"]["title"] = "Weighted F1 Score"
    layout["xaxis"]["range"] = [0.4, 1.08]
    fig.update_layout(**layout)
    return fig


# ------------------------------------------------------------------ Track 3: Clustering Plots
def cluster_scatter(coords, labels, y_true, method, title, highlight=None):
    """2D scatter plot of PCA or t-SNE with Tab10 color palette."""
    fig = go.Figure()
    idx = np.arange(len(labels))

    for c in np.unique(labels):
        m = (labels == c)
        c_int = int(c)
        color = CLUSTER_COLORS[c_int % len(CLUSTER_COLORS)]
        fig.add_trace(go.Scattergl(
            x=coords[m, 0], y=coords[m, 1],
            mode="markers",
            name=f"Group {c_int}",
            marker=dict(size=6, color=color, opacity=0.82, line=dict(width=0)),
            customdata=np.c_[idx[m], y_true[m]],
            hovertemplate="Sample #%{customdata[0]}<br>Assigned Group: " + str(c_int) +
                          "<br>Ground Truth Digit: %{customdata[1]}<extra></extra>"
        ))

    if highlight is not None and 0 <= highlight < len(coords):
        fig.add_trace(go.Scatter(
            x=[coords[highlight, 0]], y=[coords[highlight, 1]],
            mode="markers",
            name="Selected Digit",
            marker=dict(size=24, color="rgba(0,0,0,0)", line=dict(width=3, color="#EF4444")),
            hoverinfo="skip"
        ))

    layout = _base_layout(title, height=480)
    layout["xaxis"]["title"] = f"{method} Component 1"
    layout["yaxis"]["title"] = f"{method} Component 2"
    layout["legend"] = dict(title="Cluster Group", itemsizing="constant", orientation="v")
    fig.update_layout(**layout)
    return fig


def metric_tiles(m):
    """Responsive tiles for clustering evaluation metrics."""
    tiles = [
        ("Silhouette Score", f"{m['Silhouette']:.3f}", "Higher is better (−1 to +1)"),
        ("Davies-Bouldin Index", f"{m['Davies-Bouldin']:.3f}", "Lower is better (0 to ∞)"),
        ("Calinski-Harabasz", f"{m['Calinski-Harabasz']:.0f}", "Higher is better (variance ratio)"),
        ("Adjusted Rand Index (ARI)", f"{m['ARI']:.3f}", "Agreement with True Digits (0 to 1)"),
    ]
    html_tiles = "".join(
        f"""
        <div class="stat-card">
          <div class="stat-label">{t[0]}</div>
          <div class="stat-value">{t[1]}</div>
          <div class="stat-sub">{t[2]}</div>
        </div>
        """ for t in tiles
    )
    return f'<div class="stats-grid">{html_tiles}</div>'


def cluster_gallery(mean_imgs, mapping):
    """Grid of 10 cluster centroid cards with inline SVG rendered average digits."""
    cells = []
    for c, img in enumerate(mean_imgs):
        row = mapping.loc[c]
        purity = float(row["Purity"])
        maj_digit = int(row["Majority digit"])
        size_digits = int(row["Size"])
        color = CLUSTER_COLORS[c % len(CLUSTER_COLORS)]

        cells.append(f"""
        <div class="cluster-cell" style="border-top: 4px solid {color};">
          <div class="cluster-cell-header">
            <span class="cluster-dot" style="background: {color};"></span>
            <b>Cluster {c}</b>
          </div>
          <div class="cluster-cell-img">
            {digit_svg(img, size=56, color=color)}
          </div>
          <div class="cluster-cell-maj">Mostly Digit <b>{maj_digit}</b></div>
          <div class="cluster-purity-bar">
            <span style="width: {purity*100:.0f}%; background: {color};"></span>
          </div>
          <div class="cluster-cell-meta">{purity*100:.0f}% purity • {size_digits} samples</div>
        </div>
        """)

    return f"""
    <div class="gallery-wrapper">
      <p class="gallery-desc">Each tile displays the average 15×16 pixel reconstruction of all digits assigned to that cluster. Sharp edges indicate high intra-cluster consistency.</p>
      <div class="cluster-gallery-grid">{"".join(cells)}</div>
    </div>
    """


def inspect_card(img, idx, km_c, km_d, ag_c, ag_d, linkage, truth):
    """Single sample inspection display card."""
    return f"""
<div class="inspect-container">
  <div class="inspect-img-box">
    {digit_svg(img, size=130, color=PRIMARY)}
  </div>
  <div class="inspect-details">
    <div class="inspect-badge">Digit Sample #{idx}</div>
    <div class="inspect-row">
      <span class="dot" style="background: {CLUSTER_COLORS[km_c % 10]};"></span>
      <b>K-Means:</b> Assigned to <b>Group {km_c}</b> (majority: {km_d}'s)
    </div>
    <div class="inspect-row">
      <span class="dot" style="background: {CLUSTER_COLORS[ag_c % 10]};"></span>
      <b>Agglomerative ({linkage}):</b> Assigned to <b>Group {ag_c}</b> (majority: {ag_d}'s)
    </div>
    <div class="inspect-truth-box">
      <b>Ground Truth Label:</b> It is actually digit <b>{truth}</b>.
      <br><small>Note: Ground-truth labels were used exclusively for post-hoc validation as per Guideline 3.3.</small>
    </div>
  </div>
</div>
"""


# ------------------------------------------------------------------ Executive Hub Cards
def executive_hero():
    return """
<div class="hero-container">
  <div class="hero-badge">MACHINE LEARNING INTELLIGENCE STUDIO</div>
  <h1 class="hero-title">Multi-Track Machine Learning Platform</h1>
  <p class="hero-subtitle">Production-grade predictive workbench demonstrating 10 Regression algorithms, 10 Classification algorithms (Supervised Learning), and High-Dimensional Manifold Clustering (Unsupervised Learning).</p>
  <div class="hero-stats-row">
    <div class="hero-stat-pill"><b>Track 1:</b> 10 Regression Models</div>
    <div class="hero-stat-pill"><b>Track 2:</b> 10 Classification Models</div>
    <div class="hero-stat-pill"><b>Track 3:</b> Unsupervised Clustering Studio</div>
    <div class="hero-stat-pill" style="border-color: #4F46E5; color: #4F46E5; font-weight: 600;"><b>Status:</b> All Models Trained & Tuned</div>
  </div>
</div>
"""


def viva_cheat_sheet():
    return """
<div class="viva-grid">
  <div class="viva-card">
    <div class="viva-card-title">1. Problem Formulation & Scope</div>
    <p><b>Regression:</b> Predict continuous cocoa ratings (1.00–5.00) from bean provenance, cocoa %, and single-origin craft matching.</p>
    <p><b>Classification:</b> Categorize mobile devices into 4 price tiers (Low, Medium, High, Very High) from 20 hardware specifications.</p>
    <p><b>Clustering:</b> Unsupervised grouping of 2,000 digits described by 649 multi-modal features without ground-truth labels during training.</p>
  </div>
  <div class="viva-card">
    <div class="viva-card-title">2. Preprocessing & Data Leakage Prevention</div>
    <p><b>Rigorous Split:</b> Stratified 80/20 train/test splits sharing identical seeds across all 10 models in each track.</p>
    <p><b>Scaler Isolation:</b> <code>StandardScaler</code> fit exclusively on the training set and applied (transformed) onto test data to strictly avoid data leakage.</p>
    <p><b>Outlier Treatment:</b> Filtered impossible cocoa percentages (<50% or >100%) and missing categorical values imputed with domain justification.</p>
  </div>
  <div class="viva-card">
    <div class="viva-card-title">3. Model Selection & Hyperparameter Tuning</div>
    <p><b>Regression:</b> Ridge and Linear Regression achieved R² > 0.99 with stable 5-fold cross-validated R² (0.9979).</p>
    <p><b>Classification:</b> 5-fold cross-validated GridSearchCV improved Linear SVC to 97.5% accuracy and boosted AdaBoost from 49.0% to 93.8% F1.</p>
    <p><b>Clustering:</b> Evaluated Silhouette, Davies-Bouldin, and Calinski-Harabasz across candidate k. Justified k=10 using domain knowledge & elbow inflection.</p>
  </div>
</div>
"""
