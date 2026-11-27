"""End-to-end ML project: state unemployment forecasting and COVID shock clustering."""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, silhouette_score
from sklearn.cluster import KMeans, AgglomerativeClustering

ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "Unemployment in India.csv"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

for folder in (PROCESSED, RESULTS, FIGURES):
    folder.mkdir(parents=True, exist_ok=True)


def normalise_columns(df):
    df = df.copy()
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
    return df


def load_and_aggregate(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Download 'Unemployment in India.csv' from the Kaggle dataset in README.md "
            "and place it in data/raw/."
        )

    df = normalise_columns(pd.read_csv(path))
    required = [
        "Region", "Date", "Estimated Unemployment Rate (%)",
        "Estimated Employed", "Estimated Labour Participation Rate (%)"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}. Found: {list(df.columns)}")

    df["Date"] = pd.to_datetime(df["Date"].astype(str).str.strip(), dayfirst=True, errors="coerce")
    for col in required[2:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Region", "Date", required[2], required[3]])
    df["Region"] = df["Region"].astype(str).str.strip()
    df = df[(df[required[2]] >= 0) & (df[required[2]] < 100) & (df[required[3]] > 0)]
    df["Month"] = df["Date"].dt.to_period("M").dt.to_timestamp()

    # Derive labour force = employed / (1 - unemployment_rate).
    # Aggregate rural/urban rates using the derived labour force as the weight.
    u = df[required[2]] / 100.0
    df["_labour_force"] = df[required[3]] / (1.0 - u)
    df["_unemployed"] = df["_labour_force"] - df[required[3]]

    group = df.groupby(["Region", "Month"], as_index=False).agg(
        employed=("Estimated Employed", "sum"),
        labour_force=("_labour_force", "sum"),
        unemployed=("_unemployed", "sum"),
        labour_participation=("Estimated Labour Participation Rate (%)", "mean"),
        source_rows=("Region", "size"),
    )
    group["unemployment_rate"] = 100 * group["unemployed"] / group["labour_force"]
    group = group.sort_values(["Region", "Month"]).reset_index(drop=True)
    return group


def build_forecasting_data(monthly):
    frames = []
    for region, g in monthly.groupby("Region"):
        g = g.sort_values("Month").copy()
        g["lag_1"] = g["unemployment_rate"].shift(1)
        g["lag_2"] = g["unemployment_rate"].shift(2)
        g["target"] = g["unemployment_rate"].shift(-1)
        g["target_month"] = g["Month"].shift(-1)
        # Require true consecutive calendar months for both lags and target.
        month_num = g["Month"].dt.year * 12 + g["Month"].dt.month
        prev_num = month_num.shift(1)
        prev2_num = month_num.shift(2)
        next_num = month_num.shift(-1)
        g.loc[(month_num - prev_num) != 1, ["lag_1", "lag_2"]] = np.nan
        g.loc[(month_num - prev2_num) != 2, "lag_2"] = np.nan
        g.loc[(next_num - month_num) != 1, "target"] = np.nan
        frames.append(g)
    data = pd.concat(frames, ignore_index=True)
    data = data.dropna(subset=["lag_1", "lag_2", "target", "labour_participation"])
    data["month_num"] = data["Month"].dt.month
    data["year_num"] = data["Month"].dt.year
    return data


def evaluate_regression(data):
    feature_cols = [
        "unemployment_rate", "lag_1", "lag_2", "employed",
        "labour_participation", "Region", "month_num"
    ]
    numeric = ["unemployment_rate", "lag_1", "lag_2", "employed",
               "labour_participation", "month_num"]
    categorical = ["Region"]

    unique_target_months = sorted(data["target_month"].dropna().unique())
    if len(unique_target_months) < 4:
        raise ValueError("Not enough consecutive monthly observations for a chronological train/test split.")
    test_months = set(unique_target_months[-3:])
    train = data[~data["target_month"].isin(test_months)].copy()
    test = data[data["target_month"].isin(test_months)].copy()
    if train.empty or test.empty:
        raise ValueError("Chronological split produced an empty train or test set.")

    X_train, y_train = train[feature_cols], train["target"]
    X_test, y_test = test[feature_cols], test["target"]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("cat", categorical_pipe, categorical)
    ])

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "k-NN Regression": KNeighborsRegressor(n_neighbors=min(5, max(1, len(train))), weights="distance")
    }

    rows = []
    predictions = []
    baseline = test["unemployment_rate"].to_numpy()
    rows.append({
        "Model": "Persistence Baseline",
        "RMSE": mean_squared_error(y_test, baseline) ** 0.5,
        "MAE": mean_absolute_error(y_test, baseline),
        "R2": r2_score(y_test, baseline) if len(y_test) > 1 else np.nan,
        "Train samples": len(train),
        "Test samples": len(test),
        "Test target months": ", ".join(pd.to_datetime(sorted(test_months)).strftime("%Y-%m"))
    })
    predictions.append(pd.DataFrame({
        "Region": test["Region"].values,
        "Target month": test["target_month"].values,
        "Observed": y_test.values,
        "Predicted": baseline,
        "Model": "Persistence Baseline"
    }))

    for name, estimator in models.items():
        pipe = Pipeline([("preprocess", preprocessor), ("model", estimator)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        rows.append({
            "Model": name,
            "RMSE": mean_squared_error(y_test, pred) ** 0.5,
            "MAE": mean_absolute_error(y_test, pred),
            "R2": r2_score(y_test, pred) if len(y_test) > 1 else np.nan,
            "Train samples": len(train),
            "Test samples": len(test),
            "Test target months": ", ".join(pd.to_datetime(sorted(test_months)).strftime("%Y-%m"))
        })
        predictions.append(pd.DataFrame({
            "Region": test["Region"].values,
            "Target month": test["target_month"].values,
            "Observed": y_test.values,
            "Predicted": pred,
            "Model": name
        }))

    metrics = pd.DataFrame(rows).sort_values("RMSE")
    pred_df = pd.concat(predictions, ignore_index=True)
    metrics.to_csv(RESULTS / "regression_metrics.csv", index=False)
    pred_df.to_csv(RESULTS / "test_predictions.csv", index=False)

    best_name = metrics.loc[metrics["Model"] != "Persistence Baseline"].iloc[0]["Model"]
    best_pred = pred_df[pred_df["Model"] == best_name]
    plt.figure(figsize=(7, 6))
    plt.scatter(best_pred["Observed"], best_pred["Predicted"], alpha=0.75)
    lo = min(best_pred["Observed"].min(), best_pred["Predicted"].min())
    hi = max(best_pred["Observed"].max(), best_pred["Predicted"].max())
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("Observed next-month unemployment rate (%)")
    plt.ylabel("Predicted next-month unemployment rate (%)")
    plt.title(f"Observed vs Predicted — {best_name}")
    plt.tight_layout()
    plt.savefig(FIGURES / "observed_vs_predicted.png", dpi=180)
    plt.close()
    return metrics


def build_trajectories(monthly):
    rows = []
    for region, g in monthly.groupby("Region"):
        g = g.sort_values("Month").copy()
        pre = g[(g["Month"] >= "2019-05-01") & (g["Month"] <= "2020-02-29")]
        shock = g[(g["Month"] >= "2020-03-01") & (g["Month"] <= "2020-05-31")]
        june = g[g["Month"] == "2020-06-01"]
        if pre.empty or shock.empty:
            continue
        baseline = pre["unemployment_rate"].mean()
        peak = shock["unemployment_rate"].max()
        june_rate = june["unemployment_rate"].iloc[0] if not june.empty else np.nan
        rows.append({
            "Region": region,
            "pre_shock_mean": baseline,
            "shock_peak": peak,
            "shock_increase": peak - baseline,
            "june_2020_rate": june_rate,
            "peak_to_june_change": june_rate - peak if pd.notna(june_rate) else np.nan
        })
    traj = pd.DataFrame(rows).dropna()
    return traj


def cluster_states(monthly):
    traj = build_trajectories(monthly)
    if len(traj) < 4:
        warnings.warn("Too few complete state trajectories for meaningful clustering.")
        return None, None

    feature_cols = ["pre_shock_mean", "shock_peak", "shock_increase",
                    "june_2020_rate", "peak_to_june_change"]
    X = StandardScaler().fit_transform(traj[feature_cols])

    candidates = []
    for k in range(2, min(6, len(traj) - 1) + 1):
        labels = KMeans(n_clusters=k, random_state=42, n_init=20).fit_predict(X)
        score = silhouette_score(X, labels)
        candidates.append((k, score, labels))
    best_k, best_score, best_labels = max(candidates, key=lambda x: x[1])

    traj["kmeans_cluster"] = best_labels
    traj["hierarchical_cluster"] = AgglomerativeClustering(n_clusters=best_k, linkage="ward").fit_predict(X)

    traj.to_csv(RESULTS / "state_cluster_assignments.csv", index=False)
    summary = traj.groupby("kmeans_cluster")[feature_cols].mean().reset_index()
    summary["states"] = traj.groupby("kmeans_cluster")["Region"].apply(lambda s: ", ".join(sorted(s))).values
    summary["silhouette_score"] = best_score
    summary.to_csv(RESULTS / "cluster_summary.csv", index=False)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(traj["pre_shock_mean"], traj["shock_increase"],
                          c=traj["kmeans_cluster"], cmap="viridis", s=65)
    for _, row in traj.iterrows():
        plt.annotate(row["Region"], (row["pre_shock_mean"], row["shock_increase"]),
                     fontsize=6, xytext=(3, 2), textcoords="offset points")
    plt.xlabel("Pre-shock average unemployment rate (%)")
    plt.ylabel("Increase from pre-shock average to shock peak (percentage points)")
    plt.title(f"State shock/recovery clusters — K-Means (k={best_k}, silhouette={best_score:.3f})")
    plt.tight_layout()
    plt.savefig(FIGURES / "state_shock_recovery_clusters.png", dpi=180)
    plt.close()
    return traj, summary


def main():
    monthly = load_and_aggregate(RAW_FILE)
    monthly.to_csv(PROCESSED / "state_monthly_unemployment.csv", index=False)

    plt.figure(figsize=(9, 5))
    national = monthly.groupby("Month")["unemployment_rate"].mean()
    plt.plot(national.index, national.values, marker="o")
    plt.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2020-06-30"),
                alpha=0.15, label="COVID shock window")
    plt.xlabel("Month")
    plt.ylabel("Unemployment rate (%) — unweighted mean across regions")
    plt.title("Average State/Region Unemployment Trend")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "national_unemployment_trend.png", dpi=180)
    plt.close()

    metrics = evaluate_regression(build_forecasting_data(monthly))
    clusters, summary = cluster_states(monthly)

    print("\n=== Regression metrics ===")
    print(metrics.to_string(index=False))
    if summary is not None:
        print("\n=== K-Means cluster summary ===")
        print(summary.to_string(index=False))
    print("\nOutputs saved in results/, data/processed/ and figures/.")


if __name__ == "__main__":
    main()
