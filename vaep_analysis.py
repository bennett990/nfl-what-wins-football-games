"""
vaep_analysis.py
Step 2 of 2: train the two VAEP models (drive-scoring probability and win
probability), value every play, and make Figures 2 and 3.

Usage: run parse_plays.py first. Keep the raw files and
parsed_nfl_plays_2002_2011.csv in the same folder, then run:
    python vaep_analysis.py
Outputs: figure2.png, figure3.png, and model checks printed to the screen.
"""

import os
# Use one CPU thread so the models give identical results on every computer
os.environ["OMP_NUM_THREADS"] = "1"

import matplotlib
matplotlib.use("Agg")

# Core packages
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, roc_auc_score, classification_report
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt

pd.set_option("display.max_columns", 120)
pd.set_option("display.width", 160)

# ---------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------

CSV_PATH = Path("parsed_nfl_plays_2002_2011.csv")

df = pd.read_csv(CSV_PATH)

print("Loaded rows:", df.shape[0])

# ---------------------------------------------------------------------
# Merge in field position (ydline) from the raw files
# ---------------------------------------------------------------------
import glob

if "ydline" not in df.columns:
    raw_files = sorted(glob.glob("./*_nfl_pbp_data*.csv"))
    raw_dfs = [pd.read_csv(f, usecols=["gameid", "description", "ydline"], low_memory=False)
               for f in raw_files]
    raw_yd = pd.concat(raw_dfs).drop_duplicates(subset=["gameid", "description"])
    df = df.merge(raw_yd, on=["gameid", "description"], how="left")
    print(f"ydline merged. Non-null: {df['ydline'].notna().sum()} / {len(df)}")
else:
    print("ydline already present")

# ---------------------------------------------------------------------
# 2. Helpers
# ---------------------------------------------------------------------

def to_bool(x):
    if pd.isna(x):
        return False
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in {"true", "1", "yes", "y"}

def score_state_5(diff):
    # diff = offense score - defense score
    # Buckets based on the 8-point max-per-possession logic
    if pd.isna(diff):
        return "unknown"
    if diff < -8:
        return "losing_by_9_or_more"
    if diff < 0:
        return "losing_by_1_to_8"
    if diff == 0:
        return "tied"
    if diff <= 8:
        return "winning_by_1_to_8"
    return "winning_by_9_or_more"

def score_state_3(diff):
    if pd.isna(diff):
        return "unknown"
    if diff < 0:
        return "losing"
    if diff == 0:
        return "tied"
    return "winning"

def infer_drive_scores(poss_df):
    """Fallback if possession_successful is missing.
    This is conservative and should be customized if your parser has better terminal labels.
    """
    text = " ".join(poss_df.get("description", pd.Series(dtype=str)).fillna("").astype(str).tolist()).lower()
    outcome = " ".join(poss_df.get("outcome", pd.Series(dtype=str)).fillna("").astype(str).tolist()).lower()
    action = " ".join(poss_df.get("action", pd.Series(dtype=str)).fillna("").astype(str).tolist()).lower()
    combined = text + " " + outcome + " " + action
    scored = (
        "touchdown" in combined or
        "field goal is good" in combined or
        "extra point is good" in combined or
        "two-point conversion" in combined and "good" in combined
    )
    return scored

def build_final_team_scores(frame):
    """Reconstruct approximate final scores from offscore/defscore columns.

    Because rows are offense-relative, each row gives a score for `off` and a score for `def`.
    We take the maximum observed score for each team within the game.
    This works if scores are cumulative and non-decreasing.
    """
    records = []
    for _, r in frame[["gameid", "off", "def", "offscore", "defscore"]].dropna(subset=["gameid", "off", "def"]).iterrows():
        records.append((r["gameid"], r["off"], r["offscore"]))
        records.append((r["gameid"], r["def"], r["defscore"]))
    scores = pd.DataFrame(records, columns=["gameid", "team", "score"])
    scores["score"] = pd.to_numeric(scores["score"], errors="coerce")
    return scores.groupby(["gameid", "team"], as_index=False)["score"].max()

def add_sequence_features(frame):
    frame = frame.copy()

    # Normalize booleans
    for col in ["fumble", "penalty", "turnover", "success", "possession_successful"]:
        if col in frame.columns:
            frame[col] = frame[col].apply(to_bool)

    # Time. In this dataset, `min` appears to be game-clock minutes remaining or total game minutes.
    # We keep the raw qtr/min/sec values and also construct a numeric clock variable.
    frame["clock_seconds_raw"] = pd.to_numeric(frame.get("min"), errors="coerce") * 60 + pd.to_numeric(frame.get("sec"), errors="coerce")
    frame["score_diff"] = pd.to_numeric(frame.get("offscore"), errors="coerce") - pd.to_numeric(frame.get("defscore"), errors="coerce")
    frame["score_state_3"] = frame["score_diff"].apply(score_state_3)
    frame["score_state_5"] = frame["score_diff"].apply(score_state_5)

    # Sort by game, quarter, then descending clock (clock counts down within each quarter).
    # Including qtr prevents mis-ordering at quarter boundaries where clock_seconds_raw
    # can be identical for the last play of one quarter and the first of the next.
    frame = frame.sort_values(["gameid", "qtr", "clock_seconds_raw"], ascending=[True, True, False]).reset_index(drop=True)

    # Possession-level sequence features
    g = frame.groupby("possession_id", sort=False)
    frame["play_index_in_possession"] = g.cumcount() + 1
    frame["plays_in_possession"] = g["possession_id"].transform("size")
    frame["is_last_play_of_possession"] = frame["play_index_in_possession"].eq(frame["plays_in_possession"])

    frame["yards_gained"] = pd.to_numeric(frame.get("yards_gained"), errors="coerce").fillna(0)
    frame["penalty_yards"] = pd.to_numeric(frame.get("penalty_yards"), errors="coerce").fillna(0)
    frame["net_yards"] = frame["yards_gained"] + frame["penalty_yards"].fillna(0)

    frame["cum_yards_prior"] = g["net_yards"].cumsum() - frame["net_yards"]
    frame["cum_yards_after"] = g["net_yards"].cumsum()

    # Recent-action history. These are crucial for sequence effects.
    for lag in [1, 2, 3]:
        frame[f"lag{lag}_play_type"] = g["play_type"].shift(lag).fillna("START")
        frame[f"lag{lag}_action"] = g["action"].shift(lag).fillna("START")
        if "success" in frame.columns:
            frame[f"lag{lag}_success"] = g["success"].shift(lag).fillna(False)
        else:
            frame[f"lag{lag}_success"] = False
        frame[f"lag{lag}_yards"] = g["net_yards"].shift(lag).fillna(0)

    frame["last3_action_pattern"] = (
        frame["lag3_play_type"].astype(str) + ">" +
        frame["lag2_play_type"].astype(str) + ">" +
        frame["lag1_play_type"].astype(str)
    )

    # A simple risk label. Customize as needed.
    action_lower = frame["action"].fillna("").astype(str).str.lower()
    play_lower = frame["play_type"].fillna("").astype(str).str.lower()
    frame["risk_category"] = np.select(
        [
            action_lower.str.contains("intercept"),
            play_lower.eq("pass") & action_lower.str.contains("long"),
            play_lower.eq("pass"),
            play_lower.eq("run"),
            play_lower.str.contains("field"),
        ],
        ["turnover_play", "high_risk_pass", "pass", "run", "field_goal"],
        default="other"
    )

    return frame

df = add_sequence_features(df)

# Drive scoring label
if "possession_successful" in df.columns:
    df["drive_scores"] = df["possession_successful"].astype(bool)
else:
    drive_scores = df.groupby("possession_id").apply(infer_drive_scores).rename("drive_scores").reset_index()
    df = df.merge(drive_scores, on="possession_id", how="left")
    df["drive_scores"] = df["drive_scores"].fillna(False)

# Game winner label from reconstructed final scores
final_scores = build_final_team_scores(df)
final_scores = final_scores.rename(columns={"team": "off", "score": "final_off_score"})
df = df.merge(final_scores, on=["gameid", "off"], how="left")

final_scores_def = build_final_team_scores(df).rename(columns={"team": "def", "score": "final_def_score"})
df = df.merge(final_scores_def, on=["gameid", "def"], how="left")

df["offense_wins_game"] = df["final_off_score"] > df["final_def_score"]
df["game_tied"] = df["final_off_score"].eq(df["final_def_score"])

print(df[["gameid", "possession_id", "play_index_in_possession", "play_type", "action", "drive_scores", "offense_wins_game"]].head(10))
print("Drive scoring rate:", df["drive_scores"].mean())
print("Offense eventually wins rate:", df.loc[~df["game_tied"], "offense_wins_game"].mean())

# ---------------------------------------------------------------------
# 3. Define features
# ---------------------------------------------------------------------

preplay_numeric_features = [
    "qtr", "clock_seconds_raw", "down", "togo",
    "offscore", "defscore", "score_diff",
    "play_index_in_possession", "plays_in_possession",
    "cum_yards_prior",
    "ydline",
    "lag1_yards", "lag2_yards", "lag3_yards",
]

preplay_categorical_features = [
    "off", "def", "season",
    "score_state_3", "score_state_5",
    "lag1_play_type", "lag2_play_type", "lag3_play_type",
    "lag1_action", "lag2_action", "lag3_action",
    "last3_action_pattern",
]

# Keep only columns that exist
preplay_numeric_features = [c for c in preplay_numeric_features if c in df.columns]
preplay_categorical_features = [c for c in preplay_categorical_features if c in df.columns]

print("Numeric features:", preplay_numeric_features)
print("Categorical features:", preplay_categorical_features)

# ---------------------------------------------------------------------
# 4. Train/test split by game
# ---------------------------------------------------------------------

model_df = df.copy()
model_df = model_df.dropna(subset=["gameid", "possession_id"])
model_df = model_df[~model_df["game_tied"].fillna(False)].copy()

# Drop rows with missing core labels
model_df = model_df.dropna(subset=["drive_scores", "offense_wins_game"])

groups = model_df["gameid"]

splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
train_idx, test_idx = next(splitter.split(model_df, groups=groups))

train = model_df.iloc[train_idx].copy()
test = model_df.iloc[test_idx].copy()

print("Train:", train.shape, "Test:", test.shape)
print("Train games:", train["gameid"].nunique(), "Test games:", test["gameid"].nunique())

# ---------------------------------------------------------------------
# 5. Model factory
# ---------------------------------------------------------------------

def make_model(kind="boosting"):
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=False)),
    ])

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, preplay_numeric_features),
            ("cat", categorical_transformer, preplay_categorical_features),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )

    if kind == "logit":
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    else:
        # HistGradientBoosting is fast and handles nonlinearities well after preprocessing.
        # If CatBoost is available, it would also be a good choice.
        clf = HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=0.01,
            random_state=42
        )

    return Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", clf),
    ])

def evaluate_prob_model(model, X, y, label="model"):
    p = model.predict_proba(X)[:, 1]
    if len(np.unique(y)) > 1:
        auc = roc_auc_score(y, p)
    else:
        auc = np.nan
    out = {
        "label": label,
        "brier": brier_score_loss(y, p),
        "auc": auc,
        "mean_pred": p.mean(),
        "mean_actual": np.mean(y),
    }
    return out, p

# ---------------------------------------------------------------------
# 6. Fit models
# ---------------------------------------------------------------------

X_train = train[preplay_numeric_features + preplay_categorical_features]
X_test = test[preplay_numeric_features + preplay_categorical_features]

y_score_train = train["drive_scores"].astype(int)
y_score_test = test["drive_scores"].astype(int)

y_win_train = train["offense_wins_game"].astype(int)
y_win_test = test["offense_wins_game"].astype(int)

score_model = make_model(kind="boosting")
win_model = make_model(kind="boosting")

score_model.fit(X_train, y_score_train)
win_model.fit(X_train, y_win_train)

score_eval, p_score_test = evaluate_prob_model(score_model, X_test, y_score_test, "Drive scoring probability")
win_eval, p_win_test = evaluate_prob_model(win_model, X_test, y_win_test, "Win probability")

print("\nModel check on the test games:")
print(pd.DataFrame([score_eval, win_eval]).round(4).to_string(index=False))

# Baseline: the Brier score from always guessing the average rate
score_rate = y_score_test.mean()
win_rate = y_win_test.mean()
print("Baseline Brier (drive scoring):", round(score_rate * (1 - score_rate), 4))
print("Baseline Brier (win):", round(win_rate * (1 - win_rate), 4))

# ---------------------------------------------------------------------
# 8. Predict probabilities for all rows
# ---------------------------------------------------------------------

all_X = model_df[preplay_numeric_features + preplay_categorical_features]
model_df["p_drive_scores_before"] = score_model.predict_proba(all_X)[:, 1]
model_df["p_win_before"] = win_model.predict_proba(all_X)[:, 1]

# Sort by game, quarter, then descending clock to establish chronological order.
model_df = model_df.sort_values(["gameid", "qtr", "clock_seconds_raw"], ascending=[True, True, False]).reset_index(drop=True)
# Shift row-wise within game to get next play's state
for col in ["gameid", "possession_id", "off", "p_drive_scores_before", "p_win_before"]:
    model_df[f"next_{col}"] = model_df.groupby("gameid")[col].shift(-1)

same_possession_next = model_df["next_possession_id"].eq(model_df["possession_id"])

# After-play drive scoring probability:
# - if same possession continues, use next state's predicted drive scoring probability
# - if possession ends, the drive-score outcome is terminal: 1 if this possession scored, else 0
model_df["p_drive_scores_after"] = np.where(
    same_possession_next,
    model_df["next_p_drive_scores_before"],
    model_df["drive_scores"].astype(int)
)

# After-play win probability from the perspective of the offense that ran the current play:
# - if same offense continues, use next state's predicted offense win probability
# - if opponent gets ball, invert the next offense's predicted probability
# - if game ends, use realized final outcome
same_offense_next = model_df["next_off"].eq(model_df["off"])
has_next = model_df["next_p_win_before"].notna()

model_df["p_win_after"] = np.where(
    same_offense_next & has_next,
    model_df["next_p_win_before"],
    np.where(
        has_next,
        1 - model_df["next_p_win_before"],
        model_df["offense_wins_game"].astype(int)
    )
)

model_df["score_value"] = model_df["p_drive_scores_after"] - model_df["p_drive_scores_before"]
model_df["win_value"] = model_df["p_win_after"] - model_df["p_win_before"]

value_cols = [
    "gameid", "possession_id", "play_index_in_possession",
    "off", "def", "score_state_5", "play_type", "action", "outcome",
    "yards_gained", "success", "turnover", "penalty",
    "p_drive_scores_before", "p_drive_scores_after", "score_value",
    "p_win_before", "p_win_after", "win_value",
]

# ---------------------------------------------------------------------
# 9. Figure 3: run vs. pass win value by score state and down
# ---------------------------------------------------------------------

state_order  = ['losing_by_9_or_more', 'losing_by_1_to_8', 'tied',
                'winning_by_1_to_8', 'winning_by_9_or_more']
state_labels = ['Losing by 9+', 'Losing by 1–8', 'Tied',
                'Winning by 1–8', 'Winning by 9+']

sub = model_df[model_df['play_type'].isin(['pass', 'run']) & model_df['down'].isin([1, 2, 3])]
sub = sub.copy()
sub['down'] = sub['down'].astype(int)
means = (sub.groupby(['down', 'score_state_5', 'play_type'])['win_value']
            .mean().mul(100)          # convert to percentage points
            .unstack('play_type'))

fig, axes = plt.subplots(3, 1, figsize=(9.5, 11), sharex=True)
fig.suptitle('Win value: run vs. pass by score state and down',
             fontsize=16, fontweight='bold')

x = np.arange(len(state_order)); w = 0.38
for ax, down in zip(axes, [1, 2, 3]):
    m = means.loc[down].reindex(state_order)
    ax.bar(x - w/2, m['pass'], w, label='Pass', color='#c0392b')
    ax.bar(x + w/2, m['run'],  w, label='Run',  color='#7f8c8d')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_title(f'Down {down}', fontsize=13)
    ax.set_ylabel('Mean win value\n(percentage points)', fontsize=11)
    ax.tick_params(labelsize=11)

axes[0].legend(fontsize=11)
axes[-1].set_xticks(x)
axes[-1].set_xticklabels(state_labels, fontsize=11)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('figure3.png', dpi=300, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------------------
# 10. Print the exact values (percentage points) to check against the paper
# ---------------------------------------------------------------------

print("\nPlays used in VAEP model:", len(model_df))
print("\nMean win value (percentage points) by down and score state:")
print(means.reindex(pd.MultiIndex.from_product([[1, 2, 3], state_order],
                                                names=['down', 'score_state_5'])).round(2).to_string())
print("\nSaved: figure3.png")

# ---------------------------------------------------------------------
# 11. Figure 2: average win value by play outcome
# ---------------------------------------------------------------------

outcomes = model_df.copy()
outcomes["action"] = outcomes["action"].astype(str)
outcomes.loc[outcomes["action"].str.startswith("Run"), "action"] = "Run"

keep = ["Pass (Complete)", "Run", "Scramble", "Field Goal Attempt", "Punt",
        "Pass (Incomplete)", "Sack", "Pass (Intercepted)"]
chart_df = outcomes[outcomes["action"].isin(keep)].groupby("action")["win_value"].mean().mul(100)
chart_df = chart_df.sort_values()

colors = []
for v in chart_df:
    if v < 0:
        colors.append("#c0392b")
    else:
        colors.append("#27ae60")

fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(chart_df.index, chart_df.values, color=colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Mean win value (percentage points)")
ax.set_title("Average win value by play outcome")
ax.set_xlim(chart_df.min() - 1.2, chart_df.max() + 1.2)
for i, v in enumerate(chart_df.values):
    if v >= 0:
        ax.text(v + 0.1, i, f"{v:+.2f}", va="center", ha="left", fontsize=9)
    else:
        ax.text(v - 0.1, i, f"{v:+.2f}", va="center", ha="right", fontsize=9)
plt.tight_layout()
plt.savefig("figure2.png", dpi=300, bbox_inches="tight")
plt.close()

print("\nMean win value (percentage points) by play outcome:")
print(chart_df.round(2).to_string())
print("\nSaved: figure2.png")
