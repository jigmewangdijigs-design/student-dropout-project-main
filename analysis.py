from platform import python_version

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize
from sklearn.tree import DecisionTreeClassifier


DATA_FILE = "data.csv"
RANDOM_STATE = 42
LABELS = ["Dropout", "Enrolled", "Graduate"]

# These fields are integer category codes in the UCI data dictionary. They are
# one-hot encoded so arbitrary identifiers are not treated as measurements.
CATEGORICAL_FEATURES = [
    "Marital status",
    "Application mode",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International",
]


def make_preprocessor(numeric_features, scale_numeric):
    numeric_transformer = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", numeric_transformer, numeric_features),
        ],
        verbose_feature_names_out=False,
    )


df = pd.read_csv(DATA_FILE, sep=";")
df.columns = [column.replace("\ufeff", "").strip() for column in df.columns]

X = df.drop(columns="Target")
y = df["Target"]
numeric_features = [
    column for column in X.columns if column not in CATEGORICAL_FEATURES
]

# Split before model comparison. The holdout set is not used in cross-validation
# or model selection and is evaluated only once after the winning model is chosen.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

linear_preprocessor = make_preprocessor(numeric_features, scale_numeric=True)
tree_preprocessor = make_preprocessor(numeric_features, scale_numeric=False)

models = {
    "Baseline (Most Frequent)": DummyClassifier(strategy="most_frequent"),
    "Logistic Regression": Pipeline(
        [
            ("preprocess", linear_preprocessor),
            ("model", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)),
        ]
    ),
    "Decision Tree": Pipeline(
        [
            ("preprocess", tree_preprocessor),
            ("model", DecisionTreeClassifier(random_state=RANDOM_STATE)),
        ]
    ),
    "Random Forest": Pipeline(
        [
            ("preprocess", tree_preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    ),
    "Histogram Gradient Boosting": Pipeline(
        [
            ("preprocess", tree_preprocessor),
            ("model", HistGradientBoostingClassifier(random_state=RANDOM_STATE)),
        ]
    ),
}

print("REPRODUCIBILITY")
print("Python:", python_version())
print("pandas:", pd.__version__)
print("NumPy:", np.__version__)
print("scikit-learn:", sklearn.__version__)
print("Random state:", RANDOM_STATE)

print("\nDATASET QUALITY")
print("Rows:", len(df))
print("Input features:", X.shape[1])
print("Categorical features one-hot encoded:", len(CATEGORICAL_FEATURES))
print("Numeric/ordinal features:", len(numeric_features))
print("Missing values:", int(df.isna().sum().sum()))
print("Exact duplicate rows:", int(df.duplicated().sum()))
print("Target labels are retained as strings:", ", ".join(LABELS))
print("\nClass distribution:")
print(y.value_counts().reindex(LABELS))
print("\nSplit sizes:")
print("Training:", len(X_train))
print("Untouched holdout:", len(X_test))

print("\n5-FOLD STRATIFIED CROSS-VALIDATION ON TRAINING DATA ONLY")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scoring = {
    "accuracy": "accuracy",
    "weighted_f1": "f1_weighted",
    "macro_f1": "f1_macro",
    "macro_ovr_auc": "roc_auc_ovr",
}
cv_results = {}
for name, model in models.items():
    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
    )
    cv_results[name] = scores
    print(
        f"{name}: "
        f"accuracy={scores['test_accuracy'].mean():.4f} ± {scores['test_accuracy'].std():.4f}, "
        f"weighted_f1={scores['test_weighted_f1'].mean():.4f} ± {scores['test_weighted_f1'].std():.4f}, "
        f"macro_f1={scores['test_macro_f1'].mean():.4f} ± {scores['test_macro_f1'].std():.4f}, "
        f"macro_ovr_auc={scores['test_macro_ovr_auc'].mean():.4f} ± "
        f"{scores['test_macro_ovr_auc'].std():.4f}"
    )

# Weighted F1 is the primary selection metric because the classes are imbalanced.
best_name = max(
    cv_results,
    key=lambda name: cv_results[name]["test_weighted_f1"].mean(),
)
best = models[best_name]
best.fit(X_train, y_train)
pred = best.predict(X_test)
probability = best.predict_proba(X_test)
classes = list(best.classes_)

print("\nSELECTED MODEL")
print(best_name)
print("Selection rule: highest mean training-only cross-validation weighted F1")

precision, recall, f1, support = precision_recall_fscore_support(
    y_test,
    pred,
    labels=LABELS,
    average=None,
    zero_division=0,
)

print("\nUNTOUCHED 20% HOLDOUT RESULTS")
print(f"Accuracy: {accuracy_score(y_test, pred):.4f}")
weighted = precision_recall_fscore_support(
    y_test,
    pred,
    average="weighted",
    zero_division=0,
)
print(f"Weighted precision: {weighted[0]:.4f}")
print(f"Weighted recall: {weighted[1]:.4f}")
print(f"Weighted F1: {weighted[2]:.4f}")

macro_auc = roc_auc_score(
    y_test,
    probability,
    labels=classes,
    multi_class="ovr",
    average="macro",
)
print(f"Macro ROC-AUC (one-vs-rest): {macro_auc:.4f}")

print("\nPER-CLASS HOLDOUT RESULTS")
print("Class | Precision | Recall | F1 | Support | OvR ROC-AUC")
binarized = label_binarize(y_test, classes=classes)
for label, class_precision, class_recall, class_f1, class_support in zip(
    LABELS,
    precision,
    recall,
    f1,
    support,
):
    class_index = classes.index(label)
    class_auc = roc_auc_score(
        binarized[:, class_index],
        probability[:, class_index],
    )
    print(
        f"{label} | {class_precision:.4f} | {class_recall:.4f} | "
        f"{class_f1:.4f} | {class_support} | {class_auc:.4f}"
    )

print("\nCONFUSION MATRIX (rows=true, columns=predicted)")
print("Labels:", ", ".join(LABELS))
print(confusion_matrix(y_test, pred, labels=LABELS))

rf = models["Random Forest"]
rf.fit(X_train, y_train)
feature_names = rf.named_steps["preprocess"].get_feature_names_out()
importance = pd.Series(
    rf.named_steps["model"].feature_importances_,
    index=feature_names,
).sort_values(ascending=False).head(10)
print("\nTOP RANDOM FOREST FEATURE IMPORTANCES")
print(importance)
