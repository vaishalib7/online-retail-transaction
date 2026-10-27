"""Predicting Student Dropout and Academic Success Using Data Science.

Main model: early prediction at enrollment, excluding semester performance.
Optional model: later prediction after first-semester grades are available.
"""
from pathlib import Path
import zipfile, urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import f_oneway, kruskal
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, ConfusionMatrixDisplay,
                             accuracy_score, precision_recall_fscore_support)

ROOT = Path(__file__).resolve().parent
DATA_DIR, OUT_DIR = ROOT / "data", ROOT / "outputs"
DATA_DIR.mkdir(exist_ok=True); OUT_DIR.mkdir(exist_ok=True)
URL = "https://archive.ics.uci.edu/static/public/697/predict+students+dropout+and+academic+success.zip"
ZIP_PATH = DATA_DIR / "student_success.zip"

if not ZIP_PATH.exists():
    print("Downloading UCI student dataset...")
    urllib.request.urlretrieve(URL, ZIP_PATH)
with zipfile.ZipFile(ZIP_PATH) as z:
    csvs = [n for n in z.namelist() if n.lower().endswith(".csv")]
    if not csvs: raise FileNotFoundError("No CSV file found in the dataset archive.")
    csv_name = next((n for n in csvs if "data" in n.lower()), csvs[0])
    with z.open(csv_name) as f:
        df = pd.read_csv(f, sep=";")

TARGET = "Target"
if TARGET not in df.columns: raise KeyError(f"Expected target column {TARGET}")
print("Shape:", df.shape)
print("First five rows:\n", df.head())
print("\nData types:\n", df.dtypes)
print("\nMissing values:\n", df.isna().sum().sort_values(ascending=False).head(20))
print("Duplicate records:", df.duplicated().sum())
print("\nTarget counts:\n", df[TARGET].value_counts())
print("\nTarget percentages:\n", (100 * df[TARGET].value_counts(normalize=True)).round(2))

# Target distribution
target_counts = df[TARGET].value_counts().reindex(["Dropout", "Enrolled", "Graduate"])
ax = target_counts.plot(kind="bar", figsize=(8, 5))
ax.set_title("Distribution of Student Outcomes")
ax.set_xlabel("Final academic outcome"); ax.set_ylabel("Number of students")
plt.xticks(rotation=0); plt.tight_layout(); plt.savefig(OUT_DIR/"target_distribution.png", dpi=160); plt.show()

# Descriptive statistics and histograms
for col in ["Age at enrollment", "Admission grade"]:
    if col in df.columns:
        print(f"\nDescriptive statistics for {col}:\n", df[col].describe())
        plt.figure(figsize=(8,5))
        df[col].dropna().plot(kind="hist", bins=25)
        plt.title(f"Distribution of {col}"); plt.xlabel(col); plt.ylabel("Number of students")
        plt.tight_layout(); plt.savefig(OUT_DIR/(col.lower().replace(" ","_")+"_histogram.png"), dpi=160); plt.show()

# Admission grade by outcome
if "Admission grade" in df.columns:
    groups = [df.loc[df[TARGET] == c, "Admission grade"].dropna() for c in ["Dropout","Enrolled","Graduate"]]
    plt.figure(figsize=(8,5)); plt.boxplot(groups, labels=["Dropout","Enrolled","Graduate"])
    plt.title("Admission Grades by Final Outcome")
    plt.xlabel("Final academic outcome"); plt.ylabel("Admission grade")
    plt.tight_layout(); plt.savefig(OUT_DIR/"admission_grade_by_outcome.png", dpi=160); plt.show()
    stat, p = f_oneway(*groups)
    print(f"Admission grade ANOVA: F={stat:.4f}, p={p:.6g}")

# First-semester grades: statistical test and visualization
grade_col = "Curricular units 1st sem (grade)"
if grade_col in df.columns:
    groups = [df.loc[df[TARGET] == c, grade_col].dropna() for c in ["Dropout","Enrolled","Graduate"]]
    plt.figure(figsize=(8,5)); plt.boxplot(groups, labels=["Dropout","Enrolled","Graduate"])
    plt.title("First-Semester Grades by Final Outcome")
    plt.xlabel("Final academic outcome"); plt.ylabel("First-semester grade")
    plt.tight_layout(); plt.savefig(OUT_DIR/"first_semester_grade_by_outcome.png", dpi=160); plt.show()
    # ANOVA at alpha=.05; report Kruskal-Wallis as a robustness alternative.
    f_stat, p_value = f_oneway(*groups)
    h_stat, p_kw = kruskal(*groups)
    print("\nHypothesis test: first-semester grade by outcome")
    print("H0: mean first-semester grade is equal across Dropout, Enrolled, and Graduate.")
    print("H1: at least one group mean differs.")
    print(f"One-way ANOVA: F={f_stat:.4f}, p={p_value:.6g}, alpha=0.05")
    print("Conclusion:", "Reject H0; evidence of a difference in group means." if p_value < .05 else "Fail to reject H0; insufficient evidence of a mean difference.")
    print(f"Kruskal-Wallis alternative: H={h_stat:.4f}, p={p_kw:.6g}")
    print("Statistical significance does not establish practical importance or causation.")

# Age band outcome proportions
if "Age at enrollment" in df.columns:
    age_band = pd.cut(df["Age at enrollment"], bins=[0,20,25,30,40,100],
                      labels=["<=20","21-25","26-30","31-40","41+"])
    age_props = pd.crosstab(age_band, df[TARGET], normalize="index")
    print("\nOutcome proportions by age band:\n", age_props)
    age_props.plot(kind="bar", stacked=True, figsize=(9,5))
    plt.title("Outcome Proportions by Age at Enrollment")
    plt.xlabel("Age band"); plt.ylabel("Proportion of students")
    plt.tight_layout(); plt.savefig(OUT_DIR/"outcomes_by_age_band.png", dpi=160); plt.show()

# The dataset uses many integer-coded categorical fields. One-hot encode nominal-coded fields
# at enrollment, while scaling continuous fields. This list is based on the UCI data dictionary.
nominal_columns = [
    "Marital status", "Application mode", "Application order", "Course",
    "Daytime/evening attendance", "Previous qualification",
    "Nacionality", "Mother's qualification", "Father's qualification",
    "Mother's occupation", "Father's occupation", "Displaced", "Educational special needs",
    "Debtor", "Tuition fees up to date", "Gender", "Scholarship holder", "International"
]
nominal_columns = [c for c in nominal_columns if c in df.columns]
# Use early-available features only: remove all semester performance / curricular outcomes.
late_terms = ["1st sem", "2nd sem", "curricular units", "approved", "grade", "evaluations"]
exclude_early = [c for c in df.columns if c != TARGET and any(term in c.lower() for term in late_terms)]
X_early = df.drop(columns=[TARGET] + exclude_early)
y = df[TARGET]
categorical = [c for c in nominal_columns if c in X_early.columns]
numeric = [c for c in X_early.columns if c not in categorical]
preprocess = ColumnTransformer([
    ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
    ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                              ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical)
])
X_train, X_test, y_train, y_test = train_test_split(X_early, y, test_size=.2, random_state=42, stratify=y)
early_model = Pipeline([("preprocess", preprocess),
                        ("classifier", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42))])
early_model.fit(X_train, y_train)
early_pred = early_model.predict(X_test)
labels = ["Dropout", "Enrolled", "Graduate"]
print("\nEARLY PREDICTION AT ENROLLMENT")
print("Excluded late-stage features:", exclude_early)
print("Accuracy:", accuracy_score(y_test, early_pred))
print(classification_report(y_test, early_pred, labels=labels, zero_division=0))
early_report = classification_report(y_test, early_pred, labels=labels, output_dict=True, zero_division=0)
pd.DataFrame(early_report).T.to_csv(OUT_DIR/"early_classification_report.csv")
p, r, f1, _ = precision_recall_fscore_support(y_test, early_pred, labels=labels, average="macro", zero_division=0)
pd.DataFrame([{"Accuracy":accuracy_score(y_test,early_pred),"Macro precision":p,"Macro recall":r,"Macro F1":f1}]).to_csv(OUT_DIR/"early_overall_metrics.csv", index=False)
cm = confusion_matrix(y_test, early_pred, labels=labels)
ConfusionMatrixDisplay(cm, display_labels=labels).plot(values_format="d", cmap="Blues")
plt.title("Confusion Matrix — Early Prediction at Enrollment")
plt.tight_layout(); plt.savefig(OUT_DIR/"early_confusion_matrix.png", dpi=160); plt.show()

# Optional later model: allows first-semester grades, still excludes second-semester and later fields.
later_excluded = [c for c in df.columns if c != TARGET and any(term in c.lower() for term in
                  ["2nd sem", "2nd semester", "without evaluations"])]
X_later = df.drop(columns=[TARGET] + later_excluded)
cat_later = [c for c in nominal_columns if c in X_later.columns]
num_later = [c for c in X_later.columns if c not in cat_later]
later_preprocess = ColumnTransformer([
    ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num_later),
    ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                              ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_later)
])
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_later, y, test_size=.2, random_state=42, stratify=y)
later_model = Pipeline([("preprocess", later_preprocess),
                        ("classifier", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42))])
later_model.fit(X_train2, y_train2)
later_pred = later_model.predict(X_test2)
print("\nOPTIONAL LATER PREDICTION (first-semester data available)")
print("Excluded later-stage features:", later_excluded)
print(classification_report(y_test2, later_pred, labels=labels, zero_division=0))
later_report = classification_report(y_test2, later_pred, labels=labels, output_dict=True, zero_division=0)
pd.DataFrame(later_report).T.to_csv(OUT_DIR/"later_classification_report.csv")
cm2 = confusion_matrix(y_test2, later_pred, labels=labels)
ConfusionMatrixDisplay(cm2, display_labels=labels).plot(values_format="d", cmap="Blues")
plt.title("Confusion Matrix — Later Prediction")
plt.tight_layout(); plt.savefig(OUT_DIR/"later_confusion_matrix.png", dpi=160); plt.show()

# Top coefficients by absolute magnitude for each class, for the early model.
try:
    feature_names = early_model.named_steps["preprocess"].get_feature_names_out()
    clf = early_model.named_steps["classifier"]
    coef_table = pd.DataFrame(clf.coef_, columns=feature_names, index=clf.classes_)
    for cls in coef_table.index:
        print(f"\nLargest absolute coefficients for {cls} (conditional associations):")
        print(coef_table.loc[cls].abs().sort_values(ascending=False).head(10).index.to_series().to_frame("feature")
              .join(coef_table.loc[cls].rename("coefficient"), on="feature"))
    coef_table.to_csv(OUT_DIR/"early_model_coefficients.csv")
except Exception as exc:
    print("Could not export coefficient table:", exc)

print("\nResponsible-use notes: predictions are not causes or definitive judgements. Use them to offer support, audit errors and fairness, and never to penalize students.")
print("Saved outputs to:", OUT_DIR)
