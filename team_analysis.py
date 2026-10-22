"""
team_analysis.py
Step 3 of 3: team-level analysis. Paired t-tests (Table 1), correlations
(Table 2), and the Bradley-Terry regression (Table 3).

Usage: run parse_plays.py first, then run this script in the same folder:
    python team_analysis.py
"""
import re
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import ttest_rel

warnings.filterwarnings("ignore")

result_df = pd.read_csv("parsed_nfl_plays_2002_2011.csv", low_memory=False)

# =========================================================
# Yards gained, turnovers and penalties for each play
# =========================================================
def extract_yards_gained(text):
    if pd.isna(text):
        return None
    text = str(text)
    if "no play" in text.lower():
        return None
    if re.search(r"pass incomplete", text, re.IGNORECASE):
        return 0
    m = re.search(r"for (-?\d+) yard", text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    if re.search(r"for no gain", text, re.IGNORECASE):
        return 0
    return None

result_df["yards_gained"] = result_df["description"].apply(extract_yards_gained)

def determine_turnover(row):
    outcome = ""
    if not pd.isna(row["outcome"]):
        outcome = str(row["outcome"])
    if outcome == "Interception":
        return True
    if "Fumble recovered by" in outcome:
        m = re.search(r"Fumble recovered by ([A-Z]{2,4})", outcome)
        if m and m.group(1) != row["off"]:
            return True
    # Failed 4th-down runs and passes also give the ball to the other team
    if (row["play_type"] in ("run", "pass")
            and not pd.isna(row["down"])
            and pd.to_numeric(row["down"], errors="coerce") == 4
            and "Touchdown" not in outcome
            and not pd.isna(row["yards_gained"])
            and not pd.isna(row["togo"])):
        try:
            if float(row["yards_gained"]) < float(row["togo"]):
                return True
        except (TypeError, ValueError):
            pass
    return False

result_df["turnover"] = result_df.apply(determine_turnover, axis=1)

def extract_penalty_yards(text):
    if pd.isna(text):
        return 0
    m = re.search(r"PENALTY on \S+\s+[A-Za-z][A-Za-z\s\.\-]*?\s+(\d+)\s*yard", str(text))
    if m:
        return int(m.group(1))
    return 0

def extract_penalty_team(text):
    if pd.isna(text):
        return None
    m = re.search(r"PENALTY on ([A-Z]{2,4})", str(text))
    if m:
        return m.group(1)
    return None

if "penalty_yards" not in result_df.columns:
    result_df["penalty_yards"] = result_df["description"].apply(extract_penalty_yards)
if "penalty_team" not in result_df.columns:
    result_df["penalty_team"] = result_df["description"].apply(extract_penalty_team)

# =========================================================
# Possession time
# =========================================================
result_df = result_df.sort_values(
    ["gameid", "qtr", "min", "sec"], ascending=[True, True, False, False]
).reset_index(drop=True)
result_df["total_sec_remaining"] = (
    pd.to_numeric(result_df["min"], errors="coerce").fillna(0) * 60
    + pd.to_numeric(result_df["sec"], errors="coerce").fillna(0)
)
result_df["prev_total_sec"] = result_df.groupby("gameid")["total_sec_remaining"].shift(1)
result_df["prev_off"] = result_df.groupby("gameid")["off"].shift(1)
result_df["time_elapsed"] = result_df["prev_total_sec"] - result_df["total_sec_remaining"]
result_df.loc[result_df["time_elapsed"] < 0, "time_elapsed"] = 0
result_df.loc[result_df["time_elapsed"] > 300, "time_elapsed"] = 0

# =========================================================
# Team stats for each game
# =========================================================
runpass = result_df[result_df["play_type"].isin(["run", "pass"])]
off_yards = runpass.groupby(["gameid", "off"])["yards_gained"].sum().reset_index()
off_yards.columns = ["gameid", "team", "total_offensive_yards"]
pass_yards = result_df[result_df["play_type"] == "pass"].groupby(["gameid", "off"])["yards_gained"].sum().reset_index()
pass_yards.columns = ["gameid", "team", "pass_yards"]
rush_yards = result_df[result_df["play_type"] == "run"].groupby(["gameid", "off"])["yards_gained"].sum().reset_index()
rush_yards.columns = ["gameid", "team", "rush_yards"]
turnovers = result_df.groupby(["gameid", "off"])["turnover"].sum().reset_index()
turnovers.columns = ["gameid", "team", "turnovers"]
possession = result_df.dropna(subset=["prev_off"]).groupby(["gameid", "prev_off"])["time_elapsed"].sum().reset_index()
possession.columns = ["gameid", "team", "possession_time_sec"]
penalties = result_df.dropna(subset=["penalty_team"]).groupby(["gameid", "penalty_team"])["penalty_yards"].sum().reset_index()
penalties.columns = ["gameid", "team", "penalty_yards_committed"]

game_stats = off_yards.merge(pass_yards, on=["gameid", "team"], how="outer") \
    .merge(rush_yards, on=["gameid", "team"], how="outer") \
    .merge(turnovers, on=["gameid", "team"], how="outer") \
    .merge(possession, on=["gameid", "team"], how="outer") \
    .merge(penalties, on=["gameid", "team"], how="outer").fillna(0)

def pass_share(row):
    total = row["pass_yards"] + row["rush_yards"]
    if total > 0:
        return row["pass_yards"] / total
    return None

game_stats["r_ratio"] = game_stats.apply(pass_share, axis=1)

# Official final scores (added in parse_plays.py)
off_final = result_df[["gameid", "off", "final_off_score"]].dropna()
off_final.columns = ["gameid", "team", "final_score"]
def_final = result_df[["gameid", "def", "final_def_score"]].dropna()
def_final.columns = ["gameid", "team", "final_score"]
final_scores = pd.concat([off_final, def_final]).drop_duplicates(subset=["gameid", "team"])
game_stats = game_stats.merge(final_scores, on=["gameid", "team"], how="left").dropna(subset=["final_score"])

# =========================================================
# Table 1: paired t-tests, winner vs loser
# =========================================================
def winner_loser(group):
    if len(group) < 2:
        return None
    g = group.sort_values("final_score", ascending=False).reset_index(drop=True)
    if g.loc[0, "final_score"] == g.loc[1, "final_score"]:
        return None
    w = g.loc[0]
    l = g.loc[1]
    return pd.Series({
        "gameid": group.name, "winner": w["team"], "loser": l["team"],
        "win_off_yards": w["total_offensive_yards"], "lose_off_yards": l["total_offensive_yards"],
        "win_pen_yards": w["penalty_yards_committed"], "lose_pen_yards": l["penalty_yards_committed"],
        "win_turnovers": w["turnovers"], "lose_turnovers": l["turnovers"],
        "win_possession": w["possession_time_sec"], "lose_possession": l["possession_time_sec"],
        "win_r": w["r_ratio"], "lose_r": l["r_ratio"],
    })

paired = game_stats.groupby("gameid").apply(winner_loser).dropna().reset_index(drop=True)

tests = []
for label, w, l in [("Total yards", "win_off_yards", "lose_off_yards"),
                    ("Penalty yards", "win_pen_yards", "lose_pen_yards"),
                    ("Turnovers", "win_turnovers", "lose_turnovers"),
                    ("Possession time (sec)", "win_possession", "lose_possession"),
                    ("Pass share (r)", "win_r", "lose_r")]:
    c = paired.dropna(subset=[w, l])
    t, p = ttest_rel(c[w].astype(float), c[l].astype(float))
    if p < 0.01:
        significant = "Yes"
    else:
        significant = "No"
    tests.append({"Statistic": label,
                  "Avg paired diff (W-L)": round((c[w] - c[l]).mean(), 3),
                  "t-statistic": round(t, 2),
                  "p-value": f"{p:.2e}",
                  "Significant (p < 0.01)": significant})

print(f"Games with a winner: {len(paired)}")
print("\nTable 1: paired t-tests (winner vs loser)")
print(pd.DataFrame(tests).to_string(index=False))

# =========================================================
# Table 2: correlations between the five statistics
# =========================================================
corr_data = game_stats[["total_offensive_yards", "penalty_yards_committed", "turnovers",
                        "possession_time_sec", "r_ratio"]].dropna().astype(float)
corr_data.columns = ["Total yards", "Penalty yards", "Turnovers", "Possession time", "Pass share (r)"]
print("\nTable 2: correlations")
print(corr_data.corr().round(2).to_string())

# =========================================================
# Table 3: Bradley-Terry regression (home vs away)
# =========================================================
def parse_gameid(gid):
    m = re.match(r"(\d{8})_([A-Z]{2,4})@([A-Z]{2,4})", str(gid))
    if m:
        return m.group(1), m.group(2), m.group(3)
    return None, None, None

gs_idx = game_stats.set_index(["gameid", "team"])
bt_rows = []
for gid in game_stats["gameid"].unique():
    date, away, home = parse_gameid(gid)
    if home is None:
        continue
    try:
        h = gs_idx.loc[(gid, home)]
        a = gs_idx.loc[(gid, away)]
    except KeyError:
        continue
    if h["final_score"] > a["final_score"]:
        home_wins = 1
    elif h["final_score"] < a["final_score"]:
        home_wins = 0
    else:
        continue
    if pd.isna(h["r_ratio"]) or pd.isna(a["r_ratio"]):
        r_diff = np.nan
    else:
        r_diff = h["r_ratio"] - a["r_ratio"]
    bt_rows.append({
        "gameid": gid, "home_wins": home_wins,
        "total_yards_diff": h["total_offensive_yards"] - a["total_offensive_yards"],
        "penalty_yards_diff": h["penalty_yards_committed"] - a["penalty_yards_committed"],
        "turnovers_diff": h["turnovers"] - a["turnovers"],
        "possession_diff_sec": h["possession_time_sec"] - a["possession_time_sec"],
        "r_diff": r_diff,
    })

bt_df = pd.DataFrame(bt_rows).dropna()
features = ["total_yards_diff", "penalty_yards_diff", "turnovers_diff", "possession_diff_sec", "r_diff"]
X = sm.add_constant(bt_df[features].astype(float))
y = bt_df["home_wins"].astype(int)
model = sm.Logit(y, X).fit(disp=False)

print(f"\nTable 3: Bradley-Terry regression ({len(bt_df)} games)")
print(pd.DataFrame({"Coefficient": model.params.round(4), "p-value": model.pvalues}).to_string())

# =========================================================
# Two-point conversions: expected value and each team's rate
# =========================================================
nfl_plays = result_df[~result_df["off"].isin(["AFC", "NFC"])]  # leave out the Pro Bowl
extra_points = nfl_plays[nfl_plays["play_type"] == "extra_point"]
two_points = nfl_plays[nfl_plays["play_type"] == "two_point_conversion"]

xp_rate = (extra_points["outcome"] == "XP Good").mean()
two_rate = (two_points["outcome"] == "2PT Success").mean()
print(f"\nExtra points: {len(extra_points)} attempts, {xp_rate * 100:.1f}% good, "
      f"{xp_rate * 1:.3f} expected points")
print(f"Two-point tries: {len(two_points)} attempts, {two_rate * 100:.1f}% good, "
      f"{two_rate * 2:.3f} expected points")

after_td = nfl_plays[nfl_plays["play_type"].isin(["extra_point", "two_point_conversion"])]
team_rows = []
for team, plays in after_td.groupby("off"):
    tries = plays[plays["play_type"] == "two_point_conversion"]
    team_rows.append({
        "team": team,
        "2PT attempts": len(tries),
        "post-TD plays": len(plays),
        "2PT rate %": round(len(tries) / len(plays) * 100, 1),
        "2PT success %": round((tries["outcome"] == "2PT Success").mean() * 100, 1),
    })
team_2pt = pd.DataFrame(team_rows).sort_values("2PT rate %", ascending=False)
print("\nTwo-point attempt rate by team:")
print(team_2pt.to_string(index=False))
