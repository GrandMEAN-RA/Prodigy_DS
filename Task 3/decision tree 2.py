# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 20:56:32 2025

@author: EBUNOLUWASIMI
"""

""" 
decision_tree_analysis.py

Assumptions:
- Input CSV has target column 'y' with values 'yes'/'no' (adjust TARGET_COL if needed)
- Categorical columns will be one-hot encoded where necessary.
- Duration column name assumed to be 'duration'. Adjust if different.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

# -------------------------
# Config
# -------------------------
DATA_PATH = r"C:\Users\EBUNOLUWASIMI\Dropbox\Portfolio\Internships\prodigy\task 3\Data\bank additional\bank-additional.csv"   # <-- change to your file path
TARGET_COL = "y"         # <-- change if your target column name differs
ID_COLS = []             # e.g. ['customer_id'] if present
RANDOM_STATE = 42
TEST_SIZE = 0.2

OUTPUT_DIR = r"C:\Users\EBUNOLUWASIMI\Dropbox\Portfolio\Internships\prodigy\task 3\output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# Load data
# -------------------------
df = pd.read_csv(DATA_PATH,sep=';')
print("Loaded data shape:", df.shape)

# Basic cleanup if present
if ID_COLS:
    df = df.drop(columns=[c for c in ID_COLS if c in df.columns])

# Target encode: 'yes'/'no' -> 1/0
le = LabelEncoder()
if df[TARGET_COL].dtype == object:
    df[TARGET_COL] = le.fit_transform(df[TARGET_COL].astype(str))
else:
    # assume already numeric 0/1
    pass

# -------------------------
# Feature list
# -------------------------
all_features = [c for c in df.columns if c != TARGET_COL]
print("Candidate features:", all_features)

# Identify categorical columns (object or category dtype)
cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
# Remove target from cat_cols if present
cat_cols = [c for c in cat_cols if c != TARGET_COL]
num_cols = [c for c in all_features if c not in cat_cols]

print("Numerical columns:", num_cols)
print("Categorical columns:", cat_cols)

# -------------------------
# Preprocessing pipeline
# -------------------------
# We will one-hot encode categorical columns and pass numeric as-is
preprocessor = ColumnTransformer(
    transformers = [
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ],
    remainder = "passthrough"  # numeric columns get passed through
)

# -------------------------
# Helper: build, fit, evaluate
# -------------------------
def fit_and_evaluate(X_train, X_test, y_train, y_test, prefix="with_duration"):
    """
    Builds a baseline decision tree, applies ccp_alpha pruning path + grid search,
    evaluates and saves plots + metrics.
    """
    # baseline tree
    clf = Pipeline([
        ("pre", preprocessor),
        ("dt", DecisionTreeClassifier(random_state=RANDOM_STATE))
    ])

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }
    print(f"Baseline metrics ({prefix}):", metrics)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Confusion matrix ({prefix})")
    plt.colorbar()
    ticks = [0,1]
    plt.xticks(ticks, ["No","Yes"])
    plt.yticks(ticks, ["No","Yes"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i,j], ha="center", va="center", color="white" if cm[i,j]>cm.max()/2 else "black")
    plt.savefig(os.path.join(OUTPUT_DIR, f"confusion_{prefix}.png"), bbox_inches='tight')
    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, lw=2)
    plt.plot([0,1],[0,1], linestyle="--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC curve ({prefix}) - AUC: {metrics['roc_auc']:.3f}")
    plt.savefig(os.path.join(OUTPUT_DIR, f"roc_{prefix}.png"), bbox_inches='tight')
    plt.close()

    # Classification report text
    creport = classification_report(y_test, y_pred, target_names=["No","Yes"])
    with open(os.path.join(OUTPUT_DIR, f"classification_report_{prefix}.txt"), "w") as f:
        f.write(creport)

    # -------------------------
    # Cost-complexity pruning using the training data only
    # -------------------------
    # We must use the transformer separately: get transformed X_train for pruning path
    X_train_trans = preprocessor.fit_transform(X_train)
    # scikit-learn returns a numpy array; handle columns accordingly
    dt_for_path = DecisionTreeClassifier(random_state=RANDOM_STATE)
    dt_for_path.fit(X_train_trans, y_train)
    path = dt_for_path.cost_complexity_pruning_path(X_train_trans, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities

    # Plot alpha vs impurity
    plt.figure(figsize=(6,4))
    plt.plot(ccp_alphas, impurities, marker="o", drawstyle="steps-post")
    plt.xlabel("ccp_alpha")
    plt.ylabel("impurity (total leaf impurity)")
    plt.title(f"CCP Alpha vs Impurity ({prefix})")
    plt.savefig(os.path.join(OUTPUT_DIR, f"ccp_path_{prefix}.png"), bbox_inches='tight')
    plt.close()

    # Try several alphas with cross-validation to pick one
    alphas_to_try = np.unique(np.linspace(0, ccp_alphas.max(), num=25))
    param_grid = {
        "dt__ccp_alpha": alphas_to_try,
        "dt__max_depth": [None, 4, 6, 8, 12],
        "dt__min_samples_leaf": [1, 5, 10, 20]
    }

    gs = GridSearchCV(
        Pipeline([("pre", preprocessor), ("dt", DecisionTreeClassifier(random_state=RANDOM_STATE))]),
        param_grid=param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        verbose=1
    )
    gs.fit(X_train, y_train)
    print("Best params (pruned):", gs.best_params_)
    best = gs.best_estimator_

    # Evaluate best estimator on test set
    y_pred_b = best.predict(X_test)
    y_prob_b = best.predict_proba(X_test)[:,1]
    metrics_b = {
        "accuracy": accuracy_score(y_test, y_pred_b),
        "precision": precision_score(y_test, y_pred_b, zero_division=0),
        "recall": recall_score(y_test, y_pred_b, zero_division=0),
        "f1": f1_score(y_test, y_pred_b, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob_b)
    }
    print(f"Pruned tree metrics ({prefix}):", metrics_b)

    # Save pruned confusion matrix and ROC
    cm_b = confusion_matrix(y_test, y_pred_b)
    plt.figure(figsize=(5,4))
    plt.imshow(cm_b, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Confusion matrix (pruned) ({prefix})")
    plt.colorbar()
    plt.xticks(ticks, ["No","Yes"])
    plt.yticks(ticks, ["No","Yes"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm_b[i,j], ha="center", va="center", color="white" if cm_b[i,j]>cm_b.max()/2 else "black")
    plt.savefig(os.path.join(OUTPUT_DIR, f"confusion_pruned_{prefix}.png"), bbox_inches='tight')
    plt.close()

    fpr_b, tpr_b, _ = roc_curve(y_test, y_prob_b)
    plt.figure(figsize=(6,5))
    plt.plot(fpr_b, tpr_b, lw=2)
    plt.plot([0,1],[0,1], linestyle="--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC curve (pruned) ({prefix}) - AUC: {metrics_b['roc_auc']:.3f}")
    plt.savefig(os.path.join(OUTPUT_DIR, f"roc_pruned_{prefix}.png"), bbox_inches='tight')
    plt.close()

    # Plot the pruned decision tree (small depth for readability)
    # Extract tree and plot with feature names
    # Build feature names after preprocessing
    ohe_feature_names = []
    if cat_cols:
        pre = preprocessor.named_transformers_['ohe']
        cat_names = pre.get_feature_names_out(cat_cols)
        ohe_feature_names = list(cat_names)
    passthrough_names = [c for c in num_cols]
    feature_names = ohe_feature_names + passthrough_names

    # get the tree object
    tree_obj = best.named_steps['dt']
    plt.figure(figsize=(18,8))
    plot_tree(tree_obj, feature_names=feature_names, class_names=["No","Yes"], filled=True, max_depth=3, fontsize=8)
    plt.title(f"Pruned decision tree (top levels) ({prefix})")
    plt.savefig(os.path.join(OUTPUT_DIR, f"pruned_tree_{prefix}.png"), bbox_inches='tight')
    plt.close()

    # Save metrics to file
    with open(os.path.join(OUTPUT_DIR, f"metrics_{prefix}.txt"), "w") as f:
        f.write("Baseline metrics:\n")
        for k,v in metrics.items():
            f.write(f"{k}: {v}\n")
        f.write("\nPruned (best) metrics:\n")
        for k,v in metrics_b.items():
            f.write(f"{k}: {v}\n")
        f.write("\nBest params:\n")
        f.write(str(gs.best_params_))

    # feature importances
    importances = tree_obj.feature_importances_
    fi_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False).head(30)

    fi_df.to_csv(os.path.join(OUTPUT_DIR, f"feature_importances_{prefix}.csv"), index=False)

    plt.figure(figsize=(8,6))
    plt.barh(fi_df['feature'][::-1], fi_df['importance'][::-1])
    plt.title(f"Feature Importance (pruned tree) ({prefix})")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"feature_importance_{prefix}.png"))
    plt.close()

    return {
        "baseline_metrics": metrics,
        "pruned_metrics": metrics_b,
        "best_params": gs.best_params_
    }


# -------------------------
# Prepare dataset splits (with duration)
# -------------------------
X = df[all_features].copy()
y = df[TARGET_COL].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

results_with_duration = fit_and_evaluate(X_train, X_test, y_train, y_test, prefix="with_duration")

# -------------------------
# Repeat WITHOUT duration
# -------------------------
if "duration" in all_features:
    features_no_duration = [f for f in all_features if f != "duration"]
    X_nd = df[features_no_duration]
    X_train_nd, X_test_nd, y_train_nd, y_test_nd = train_test_split(X_nd, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    # Update global lists used in helper (num_cols is used; regenerate local)
    # We'll monkeypatch num_cols variable used in inner function by re-defining it here:
    num_cols = [c for c in X_nd.columns if c not in cat_cols]
    results_no_duration = fit_and_evaluate(X_train_nd, X_test_nd, y_train_nd, y_test_nd, prefix="no_duration")
else:
    print("No 'duration' column found; skipping no-duration run.")
    results_no_duration = None

# -------------------------
# Generate quick ML report (markdown)
# -------------------------
report_lines = []
report_lines.append("# Decision Tree Analysis Report\n")
report_lines.append("## Summary\n")
report_lines.append(f"- Dataset: `{DATA_PATH}`\n")
report_lines.append(f"- Rows: {len(df):,}  Columns: {df.shape[1]}\n")
report_lines.append("## Key Findings (summary of runs)\n")
report_lines.append("### With `duration` included\n")
for k,v in results_with_duration['baseline_metrics'].items():
    report_lines.append(f"- Baseline {k}: {v:.4f}\n")
for k,v in results_with_duration['pruned_metrics'].items():
    report_lines.append(f"- Pruned {k}: {v:.4f}\n")
report_lines.append(f"- Best params: {results_with_duration['best_params']}\n")

if results_no_duration:
    report_lines.append("\n### Without `duration` (real-world scenario)\n")
    for k,v in results_no_duration['baseline_metrics'].items():
        report_lines.append(f"- Baseline {k}: {v:.4f}\n")
    for k,v in results_no_duration['pruned_metrics'].items():
        report_lines.append(f"- Pruned {k}: {v:.4f}\n")
    report_lines.append(f"- Best params: {results_no_duration['best_params']}\n")

report_lines.append("\n## Recommendations\n")
report_lines.append("- Do not use `duration` in a production pre-call prediction model; it leaks post-hoc information.\n")
report_lines.append("- Use the pruned tree parameters (see metrics) or consider ensemble models like RandomForest/GradientBoosting for production.\n")
report_lines.append("- Consider cost-sensitive thresholds (tune for recall/precision trade-offs depending on business cost of false positives/negatives).\n")

with open(os.path.join(OUTPUT_DIR, "ml_report.md"), "w") as f:
    f.writelines([l + "\n" for l in report_lines])

print("All outputs saved to:", OUTPUT_DIR)
