"""Loading and preprocessing for all project datasets (Regression, Classification, Clustering)."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as C


# ---------------------------------------------------------------- regression
def load_regression():
    """Load and clean the Flavors of Cacao chocolate rating dataset."""
    df = pd.read_csv(C.REGRESSION_CSV)
    df.columns = (df.columns.str.replace(r"[\s\xa0]+", " ", regex=True)
                            .str.strip()
                            .str.lower())

    # Clean Cocoa Percent (convert e.g. '70%' to 70.0)
    df["cocoa percent"] = pd.to_numeric(
        df["cocoa percent"].astype(str).str.rstrip("%").str.strip(),
        errors="coerce"
    )

    # Impute missing categorical values
    df["bean type"] = df["bean type"].fillna("Unknown")
    df["broad bean origin"] = df["broad bean origin"].fillna("Unknown")

    # Drop duplicate records
    dup_subset = ["company (maker-if known)", "specific bean origin or bar name", "ref"]
    df = df[~df.duplicated(subset=dup_subset, keep="first")].copy()

    # Outlier filter (50% to 100% cocoa)
    df = df[(df["cocoa percent"] >= 50) & (df["cocoa percent"] <= 100)].copy()

    # Engineered feature: proxy for hyper-local / bean-to-bar single-origin producer
    df["origin_matches_location"] = (df["company location"] == df["broad bean origin"]).astype(int)

    return df


def split_regression(df=None):
    """Clean, encode and split the regression dataset matching notebook methodology."""
    df = load_regression() if df is None else df.copy()

    # Group rare categories (<10 occurrences)
    def group_rare(data, col, threshold=10):
        val_counts = data[col].value_counts()
        rare_cats = val_counts[val_counts < threshold].index
        return data[col].replace(rare_cats, "Other")

    for col in ["company location", "bean type", "broad bean origin"]:
        df[col] = group_rare(df, col, 10)

    # Stratification bin
    rating_binned = pd.qcut(df["rating"], q=3, labels=["Low", "Medium", "High"])

    categorical_cols = ["company location", "bean type", "broad bean origin"]
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Target-encoded composite feature as done in regression notebook
    df["combined_id"] = (df["company (maker-if known)"].astype(str) + "_" +
                         df["specific bean origin or bar name"].astype(str) + "_" +
                         df["cocoa percent"].astype(str))
    mapping = df.groupby("combined_id")["rating"].mean()
    df_encoded["target_encoded_feature"] = df["combined_id"].map(mapping)

    # Select only numeric & boolean columns and convert to float
    df_ml = df_encoded.select_dtypes(include=[np.number, "bool", "float64", "int64"]).astype(float).copy()

    X = df_ml.drop(columns=["rating", "rating_class", "Rating"], errors="ignore")
    y = df["rating"].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=C.TEST_SIZE, stratify=rating_binned, random_state=C.RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test, X.columns.tolist()


# ---------------------------------------------------------------- classification
def load_classification():
    """Return the Mobile Price dataframe (2000 x 21)."""
    return pd.read_csv(C.CLASSIFICATION_CSV)


def split_classification(df=None):
    """Stratified 80/20 split - identical seed to Review 1 so all 10 models share it."""
    df = load_classification() if df is None else df
    X = df.drop(columns=[C.TARGET_COL])
    y = df[C.TARGET_COL]
    return train_test_split(X, y, test_size=C.TEST_SIZE, stratify=y,
                            random_state=C.RANDOM_STATE)


# ---------------------------------------------------------------- clustering
CACHE_PATH = C.MFEAT_DIR / "mfeat_cache.npz"


def load_mfeat(sets=None):
    """Load and horizontally stack the chosen mfeat feature files with caching.

    Returns
    -------
    X : ndarray (2000, n_features)
    y : ndarray (2000,) digit labels 0-9
    feature_names : list[str]
    """
    sets = sets or C.MFEAT_SETS

    # Fast-path from cache if default sets are requested
    if sets == C.MFEAT_SETS and CACHE_PATH.exists():
        data = np.load(CACHE_PATH, allow_pickle=True)
        return data["X"], data["y"], list(data["feature_names"])

    blocks, names = [], []
    for s in sets:
        arr = np.loadtxt(C.MFEAT_DIR / f"mfeat-{s}")
        blocks.append(arr)
        names += [f"{s}_{i}" for i in range(arr.shape[1])]
    X = np.hstack(blocks)
    y = np.repeat(np.arange(10), 200)

    # Save cache if default sets
    if sets == C.MFEAT_SETS:
        pix = np.loadtxt(C.MFEAT_DIR / "mfeat-pix")
        np.savez_compressed(CACHE_PATH, X=X, y=y, feature_names=names, pix=pix)

    return X, y, names


def pixel_image(row_idx, pix=None):
    """Rebuild the 15 x 16 digit image from mfeat-pix (values 0-6)."""
    if pix is None:
        if CACHE_PATH.exists():
            data = np.load(CACHE_PATH, allow_pickle=True)
            if "pix" in data:
                pix = data["pix"]
        if pix is None:
            pix = np.loadtxt(C.MFEAT_DIR / "mfeat-pix")
    return pix[row_idx].reshape(16, 15)
