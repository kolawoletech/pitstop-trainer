# F1 Pit Stop Prediction - Final Ensemble Notebook Script
# Competition: playground-series-s6e5

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

try:
    from sklearn.model_selection import StratifiedKFold
except Exception:
    # Fallback for environments where model_selection cannot be resolved
    from sklearn.cross_validation import StratifiedKFold  # type: ignore
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, TargetEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

TRAIN_PATH = "../data/train.csv"
TEST_PATH = "../data/test.csv"

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

TARGET = "PitNextLap"
ID = "id"

y = train[TARGET].astype(int)
X = train.drop(columns=[TARGET]).copy()
X_test = test.copy()

median_stint = {
    "SOFT": 14,
    "MEDIUM": 17,
    "HARD": 23,
    "INTERMEDIATE": 17,
    "WET": 11,
}

for df in [X, X_test]:
    df["TyreLife_Normalized"] = df["TyreLife"] / df["Compound"].map(median_stint)
    df["TyreLife_x_Stint"] = df["TyreLife"] * df["Stint"]
    if "Position" in df.columns:
        df["TyreLife_x_Position"] = df["TyreLife"] * df["Position"]
        df["Stint_x_Position"] = df["Stint"] * df["Position"]
    if "LapNumber" in df.columns:
        df["TyreLife_x_LapNumber"] = df["TyreLife"] * df["LapNumber"]

class FEEncoder(BaseEstimator, TransformerMixin):

    def fit(self, X, y):
        df = X.copy()
        df["target"] = y

        self.rc = df.groupby(["Race","Compound"])["target"].mean().reset_index()
        self.ry = df.groupby(["Race","Year"])["target"].mean().reset_index()
        self.sy = df.groupby(["Stint","Year"])["target"].mean().reset_index()
        self.cy = df.groupby(["Compound","Year"])["target"].mean().reset_index()
        return self

    def transform(self, X):
        X = X.copy()

        X = X.merge(self.rc, on=["Race","Compound"], how="left")
        X.rename(columns={"target":"Race_Compound_Prior_Prob"}, inplace=True)

        X = X.merge(self.ry, on=["Race","Year"], how="left")
        X.rename(columns={"target":"Race_Year_Prior_Prob"}, inplace=True)

        X = X.merge(self.sy, on=["Stint","Year"], how="left")
        X.rename(columns={"target":"Stint_Year_Prior_Prob"}, inplace=True)

        X = X.merge(self.cy, on=["Compound","Year"], how="left")
        X.rename(columns={"target":"Compound_Year_Prior_Prob"}, inplace=True)

        return X

cat_driver = ["Driver"]
cat_compound = ["Compound"]
cat_race = ["Race"]

preprocessor = ColumnTransformer(
    transformers=[
        ("te_driver", TargetEncoder(random_state=42, smooth=20), cat_driver),
        ("te_race", TargetEncoder(random_state=42, smooth=20), cat_race),
        ("ohe_compound", OneHotEncoder(handle_unknown="ignore"), cat_compound),
        ("ohe_race", OneHotEncoder(handle_unknown="ignore"), cat_race),
    ],
    remainder="passthrough"
)

lgb_params = {
    "objective":"binary",
    "metric":"auc",
    "n_estimators":931,
    "learning_rate":0.05325809257332312,
    "num_leaves":83,
    "max_depth":9,
    "min_child_samples":156,
    "subsample":0.5820389823067827,
    "colsample_bytree":0.5198555301603772,
    "reg_alpha":9.692844859786788,
    "reg_lambda":3.2948430607114104e-08,
    "is_unbalance":True,
    "verbosity":-1
}

xgb_params = {
    "objective":"binary:logistic",
    "eval_metric":"logloss",
    "tree_method":"hist",
    "n_estimators":474,
    "learning_rate":0.05414044697790815,
    "max_depth":8,
    "subsample":0.9095630474675961,
    "colsample_bytree":0.5530946037784168,
    "gamma":3.352206419195436e-08,
    "reg_lambda":0.1395888561040328,
    "alpha":5.2561459162019464e-08,
}

SEEDS = [42,777,2026]

pred_lgb = np.zeros(len(X_test))
pred_xgb = np.zeros(len(X_test))
pred_cat = np.zeros(len(X_test))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for seed in SEEDS:

    oof = np.zeros(len(X))

    for tr_idx, va_idx in cv.split(X,y):

        X_tr = X.iloc[tr_idx]
        X_va = X.iloc[va_idx]

        y_tr = y.iloc[tr_idx]
        y_va = y.iloc[va_idx]

        fe = FEEncoder()
        X_tr = fe.fit_transform(X_tr, y_tr)
        X_va = fe.transform(X_va)
        X_te = fe.transform(X_test)

        Xt = preprocessor.fit_transform(X_tr, y_tr)
        Xv = preprocessor.transform(X_va)
        Xs = preprocessor.transform(X_te)

        lgb = LGBMClassifier(random_state=seed, **lgb_params)
        xgb = XGBClassifier(random_state=seed, **xgb_params)

        cat = CatBoostClassifier(
            iterations=3000,
            depth=8,
            learning_rate=0.03,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=seed,
            verbose=False
        )

        lgb.fit(Xt, y_tr)
        xgb.fit(Xt, y_tr)
        cat.fit(Xt, y_tr)

        pred_lgb += lgb.predict_proba(Xs)[:,1] / (len(SEEDS)*5)
        pred_xgb += xgb.predict_proba(Xs)[:,1] / (len(SEEDS)*5)
        pred_cat += cat.predict_proba(Xs)[:,1] / (len(SEEDS)*5)

final_pred = (
    0.50 * pred_lgb +
    0.25 * pred_xgb +
    0.25 * pred_cat
)

submission = pd.DataFrame({
    "id": test["id"],
    "PitNextLap": final_pred
})

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("saved submission.csv")