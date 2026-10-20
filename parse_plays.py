"""
parse_plays.py
Step 1 of 2: parse Brian Burke's raw 2002-2011 NFL play-by-play files into one
structured file (parsed_nfl_plays_2002_2011.csv).

Usage: put the ten raw files (2002_nfl_pbp_data.csv ... 2011_nfl_pbp_data.csv)
in the same folder as this script, then run:  python parse_plays.py
"""

########################################
# CELL 1 — Parser functions
########################################

import re
import pandas as pd
import numpy as np
import glob
import os

# ---------------------------------------------------------
# Helper: Extract yards
# ---------------------------------------------------------
def extract_yards(text):
    m = re.search(r'for (-?\d+) yard', text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    if re.search(r'for no gain', text, re.IGNORECASE):
        return 0
    return None

# ---------------------------------------------------------
# Helper: Extract formation
# ---------------------------------------------------------
def extract_formation(text):
    m = re.search(r'\((Shotgun|No Huddle Shotgun|No Huddle|Field Goal formation|Punt formation|Kick formation|Pass formation)\)', text)
    if m:
        val = m.group(1)
        if "No Huddle" in val and "Shotgun" in val:
            return "Shotgun (No Huddle)"
        return val
    return "Standard"

# ---------------------------------------------------------
# Helper: Extract primary player
# ---------------------------------------------------------
def extract_player(text):
    cleaned = re.sub(r'\((?:Shotgun|No Huddle Shotgun|No Huddle|Field Goal formation|Punt formation|Kick formation|Pass formation)\)\s*', '', text)
    # Strip two-point conversion prefix so the player regex can hit
    cleaned = re.sub(r'(?:shotgun\s+)?TWO-POINT CONVERSION ATTEMPT\.\s*', '', cleaned, flags=re.IGNORECASE)
    patterns = [
        r'(?:^|\)\s*)([A-Z]\.\s?[A-Za-z\'\\-]+)\s+(?:pass|sacked|scramble)',
        r'(?:^|\)\s*)([A-Z]\.\s?[A-Za-z\'\\-]+)\s+(?:left|right|up the middle)',
        r'(?:^|\)\s*)([A-Z]\.\s?[A-Za-z\'\\-]+)\s+rushes',
        r'(?:^|\)\s*)([A-Z]\.\s?[A-Za-z\'\\-]+)\s+\d+\s*yard field goal',
        r'(?:^|\)\s*)([A-Z]\.\s?[A-Za-z\'\\-]+)\s+extra point',
        r'(?:^|\)\s*)([A-Z]\.\s?[A-Za-z\'\\-]+)\s+punts',
        r'(?:^|\)\s*)([A-Z]\.\s?[A-Za-z\'\\-]+)\s+kicks',
        r'(?:^|\)\s*)([A-Z]\.\s?[A-Za-z\'\\-]+)\s+kneels',
        r'(?:^|\)\s*)([A-Z]\.\s?[A-Za-z\'\\-]+)\s+ABORTED',
        r'(?:^|\)\s*)([A-Z]\.\s?[A-Za-z\'\\-]+)\s+(?:fumbles|FUMBLES)',
    ]
    for pat in patterns:
        m = re.search(pat, cleaned)
        if m:
            return m.group(1).strip()
    return None

# ---------------------------------------------------------
# Helper: Extract intended target
# ---------------------------------------------------------
def extract_intended_target(text):
    m = re.search(r'intended for ([A-Z]\.\s?[A-Za-z\'\\-]+)', text)
    if m:
        return m.group(1)
    m = re.search(r'(?:short|deep)\s+(?:left|right|middle)\s+to\s+([A-Z]\.\s?[A-Za-z\'\\-]+)', text)
    if m:
        return m.group(1)
    # "pass to X.Name is complete/incomplete" (two-point conversion style)
    m = re.search(r'pass to ([A-Z]\.\s?[A-Za-z\'\\-]+)\s+is\s+', text)
    if m:
        return m.group(1)
    return None

# ---------------------------------------------------------
# Helper: Extract defender
# ---------------------------------------------------------
def extract_defender(text):
    parens = re.findall(r'\(([^)]+)\)', text)
    if not parens:
        return None
    skip_patterns = [
        r'^\d+:\d+', r'^Shotgun', r'^No Huddle', r'^Field Goal formation',
        r'^Punt formation', r'^Kick formation', r'^Pass formation',
        r'^Center-', r'^Holder-', r'^Timeout',
    ]
    defenders = []
    for p in parens:
        p_stripped = p.strip()
        if any(re.match(sp, p_stripped) for sp in skip_patterns):
            continue
        if re.search(r'[A-Z]\.\s?[A-Za-z]', p_stripped):
            defenders.append(p_stripped)
    if not defenders:
        return None
    return defenders[-1]

# ---------------------------------------------------------
# Helper: Detect play type
# ---------------------------------------------------------
def detect_play_type(text):
    t = text.lower()
    if "two-point conversion" in t:
        return "two_point_conversion"
    if "extra point" in t:
        return "extra_point"
    if "field goal" in t:
        return "field_goal"
    if "punts" in t:
        return "punt"
    if re.search(r'\bkicks\b', t):
        return "kickoff"
    if "kneels" in t:
        return "kneel"
    if "sacked" in t:
        return "pass"
    if "pass" in t or "scramble" in t:
        return "pass"
    if "rushes" in t:
        return "run"
    if any(w in t for w in ["left end", "right end", "left guard", "right guard",
                             "left tackle", "right tackle", "up the middle"]):
        return "run"
    if re.search(r'\b(left|right|middle)\b', t) and "pass" not in t and "field goal" not in t:
        return "run"
    return "unknown"

# ---------------------------------------------------------
# Helper: Determine outcome
# ---------------------------------------------------------
def determine_outcome(text, play_type):
    t = text.lower()
    if play_type == "two_point_conversion":
        if "attempt succeeds" in t:
            return "2PT Success"
        return "2PT Failure"
    if play_type == "extra_point":
        if "blocked" in t:
            return "XP Blocked"
        if "no good" in t or "aborted" in t:
            return "XP No Good"
        if "good" in t:
            return "XP Good"
        return "XP Unknown"
    if "touchdown" in t:
        return "Touchdown"
    if "intercept" in t:
        return "Interception"
    if "fumbles" in t and "recovered by" in t:
        m = re.search(r'recovered by ([A-Z]{2,4}(?:-[A-Z]\.\s?[A-Za-z\'\\-]+)?)', text)
        if m:
            recovery = m.group(1)
        else:
            recovery = "unknown"
        return f"Fumble recovered by {recovery}"
    if "fumbles" in t:
        return "Fumble"
    if play_type == "field_goal":
        if "no good" in t or "wide" in t or "blocked" in t or "missed" in t:
            return "Field Goal No Good"
        if "good" in t:
            return "Field Goal Good"
        return "Field Goal Attempt"
    if "incomplete" in t:
        return "Incomplete"
    if "sacked" in t:
        yards = extract_yards(text)
        if yards is not None:
            return f"Sack (loss of {abs(yards)})"
        return "Sack"
    if play_type in ("run", "pass"):
        yards = extract_yards(text)
        if yards is not None:
            if yards > 0:
                return f"Gain of {yards}"
            elif yards < 0:
                return f"Loss of {abs(yards)}"
            else:
                return "No Gain"
    if play_type == "punt":
        yards = extract_yards(text)
        if yards is not None:
            return f"Punt for {yards} yards"
        return "Punt"
    if play_type == "kickoff":
        return "Kickoff"
    if play_type == "kneel":
        return "Kneel"
    return "Unknown"

# ---------------------------------------------------------
# Helper: Build action description
# ---------------------------------------------------------
def build_action(text, play_type):
    t = text.lower()
    if play_type == "two_point_conversion":
        if "pass" in t:
            return "Two-Point Conversion (Pass)"
        if "rushes" in t or "rush" in t or any(w in t for w in ["left", "right", "middle"]):
            return "Two-Point Conversion (Rush)"
        return "Two-Point Conversion"
    if play_type == "extra_point":
        return "Extra Point"
    if play_type == "pass":
        if "intercept" in t:
            return "Pass (Intercepted)"
        if "incomplete" in t:
            return "Pass (Incomplete)"
        if "sacked" in t:
            return "Sack"
        if "scramble" in t:
            return "Scramble"
        return "Pass (Complete)"
    if play_type == "run":
        dir_match = re.search(
            r'\b(left end|right end|left guard|right guard|left tackle|right tackle|up the middle)\b', t)
        if dir_match:
            return f"Run ({dir_match.group(0)})"
        dir_match2 = re.search(r'\b(left|right|middle)\b', t)
        if dir_match2:
            return f"Run ({dir_match2.group(0)})"
        return "Run"
    if play_type == "field_goal":
        return "Field Goal Attempt"
    if play_type == "punt":
        return "Punt"
    if play_type == "kickoff":
        return "Kickoff"
    if play_type == "kneel":
        return "Kneel"
    return "Unknown"

# ---------------------------------------------------------
# Helper: Check "no play"
# ---------------------------------------------------------
def is_no_play(text):
    return "no play" in text.lower()

# ---------------------------------------------------------
# Helper: Extract penalty detail
# ---------------------------------------------------------
def extract_penalty_detail(text):
    m = re.search(r'PENALTY on ([A-Z]{2,4}(?:-[A-Z]\.\s?[A-Za-z\'\\-]+)?),\s*([^,]+)', text)
    if m:
        return f"{m.group(1).strip()}: {m.group(2).strip()}"
    m2 = re.search(r'PENALTY on ([A-Z]{2,4}(?:-[A-Z]\.\s?[A-Za-z\'\\-]+)?)', text)
    if m2:
        return m2.group(1).strip()
    return None

# =========================================================
# MAIN PARSER
# =========================================================
def parse_play(text):
    text = str(text)
    t = text.lower()

    has_penalty = "penalty" in t
    no_play = is_no_play(text)

    # Penalty + No Play
    if has_penalty and no_play:
        detail = extract_penalty_detail(text)
        if detail:
            no_play_outcome = f"No Play - {detail}"
        else:
            no_play_outcome = "No Play"
        return {
            "play_type": "penalty_no_play",
            "action": "Penalty (No Play)",
            "player": None,
            "intended_target": None,
            "defender": None,
            "formation": extract_formation(text),
            "outcome": no_play_outcome,
            "fumble": False,
            "penalty": True,
            "penalty_detail": detail,
        }

    play_type = detect_play_type(text)
    formation = extract_formation(text)
    player = extract_player(text)
    intended_target = extract_intended_target(text)
    defender = extract_defender(text)
    fumble = "fumbles" in t or "FUMBLES" in text
    outcome = determine_outcome(text, play_type)
    action = build_action(text, play_type)

    penalty_detail = None
    if has_penalty:
        penalty_detail = extract_penalty_detail(text)

    return {
        "play_type": play_type,
        "action": action,
        "player": player,
        "intended_target": intended_target,
        "defender": defender,
        "formation": formation,
        "outcome": outcome,
        "fumble": fumble,
        "penalty": has_penalty,
        "penalty_detail": penalty_detail,
    }

# =========================================================
# Load all raw files and parse every play
# =========================================================
data_dir = "."
all_files = sorted(glob.glob(os.path.join(data_dir, "*_nfl_pbp_data*.csv")))
print(f"Found {len(all_files)} raw files")

df = pd.concat([pd.read_csv(f, low_memory=False) for f in all_files], ignore_index=True)
print(f"Total rows loaded: {len(df)}")

parsed = df["description"].apply(parse_play)
result_df = pd.DataFrame(list(parsed))

metadata_cols = ["gameid", "qtr", "min", "sec", "off", "def", "down", "togo", "season", "description"]
for position, col in enumerate(metadata_cols):
    result_df.insert(position, col, df[col].values)

# =========================================================
# Did the offense itself score on this play?
# Uses the raw file order, which is chronological.
# A touchdown only counts for the offense if the offense scored it:
# interceptions, fumbles, or punts returned for a touchdown are
# scored by the defense. The team that kicks the extra point (or
# attempts the two-point conversion) right after a touchdown is the
# team that scored it.
# =========================================================
def final_play_text(text):
    # If a play was reviewed and reversed, only the text after
    # "REVERSED." describes what actually happened
    text = str(text)
    position = text.upper().rfind("REVERSED.")
    if position == -1:
        return text
    return text[position + len("REVERSED."):]

def is_real_touchdown(text):
    final_text = final_play_text(text).upper()
    if "TOUCHDOWN" not in final_text:
        return False
    if "NULLIFIED" in final_text:
        return False
    return True

# Some descriptions use different team abbreviations than the off/def columns
text_abbreviations = {
    "ARI": ["ARI", "ARZ"],
    "BAL": ["BAL", "BLT"],
    "CLE": ["CLE", "CLV"],
    "HOU": ["HOU", "HST"],
    "JAC": ["JAC", "JAX"],
    "STL": ["STL", "SL"],
}

def recovered_by(final_text, team):
    team = str(team).upper()
    names = text_abbreviations.get(team, [team])
    for name in names:
        if "RECOVERED BY " + name + "-" in final_text:
            return True
    return False

def defense_scored_fallback(row):
    # Only used when no extra point or two-point try follows the touchdown
    # (for example, an overtime touchdown that ends the game)
    final_text = final_play_text(row["description"]).upper()
    if "INTERCEPTED" in final_text:
        return True
    if recovered_by(final_text, row["def"]):
        return True
    if row["play_type"] in ["punt", "kickoff"]:
        # On kicks, a touchdown usually means the return team scored,
        # unless the kicking team recovered the ball
        if recovered_by(final_text, row["off"]):
            return False
        return True
    if row["play_type"] == "field_goal":
        # A touchdown after a missed or blocked kick was returned by the
        # defense. Fake field goals (a run or pass) count for the offense.
        if "NO GOOD" in final_text or "BLOCKED" in final_text:
            return True
        return False
    return False

pat_types = ["extra_point", "two_point_conversion"]
stop_types = ["run", "pass", "field_goal", "punt", "kickoff"]

is_td = result_df["description"].apply(is_real_touchdown)
game_ids = result_df["gameid"].values
play_types = result_df["play_type"].values
offenses = result_df["off"].values

offense_td = [False] * len(result_df)
used_fallback = 0
for i in np.flatnonzero(is_td.values):
    scoring_team = None
    for j in range(i + 1, min(i + 5, len(result_df))):
        if game_ids[j] != game_ids[i]:
            break
        if play_types[j] in pat_types:
            scoring_team = offenses[j]
            break
        if play_types[j] in stop_types:
            break

    if scoring_team is not None:
        offense_td[i] = (scoring_team == offenses[i])
    else:
        used_fallback += 1
        offense_td[i] = not defense_scored_fallback(result_df.iloc[i])

result_df["offense_td"] = offense_td

field_goal_good = result_df["outcome"].astype(str) == "Field Goal Good"
result_df["offense_scored"] = result_df["offense_td"] | field_goal_good

all_td = is_td.sum()
print(f"Touchdown plays: {all_td}")
print(f"  scored by the offense: {result_df['offense_td'].sum()}")
print(f"  scored by the defense or return team: {all_td - result_df['offense_td'].sum()}")
print(f"  decided without a following extra point/2PT: {used_fallback}")

# =========================================================
# Yards gained, penalties, turnovers, success
# =========================================================
import re
import numpy as np
 
def extract_yards_gained(text):
    if pd.isna(text):
        return None
    text = str(text)
    if "no play" in text.lower():
        return None
    m = re.search(r'for (-?\d+) yard', text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    if re.search(r'for no gain', text, re.IGNORECASE):
        return 0
    return None
 
def extract_penalty_yards(text):
    if pd.isna(text):
        return 0
    text = str(text)
    m = re.search(r'PENALTY on \S+\s+[A-Za-z][A-Za-z\s\.\-]*?\s+(\d+)\s*yard', text)
    if m:
        return int(m.group(1))
    return 0
 
def extract_penalty_team(text):
    if pd.isna(text):
        return None
    text = str(text)
    m = re.search(r'PENALTY on ([A-Z]{2,4})', text)
    if m:
        return m.group(1)
    return None
 
def determine_turnover(row):
    if pd.isna(row['outcome']):
        outcome = ""
    else:
        outcome = str(row['outcome'])
    if outcome == "Interception":
        return True
    if "Fumble recovered by" in outcome:
        m = re.search(r'Fumble recovered by ([A-Z]{2,4})', outcome)
        if m and m.group(1) != row['off']:
            return True
    if (row['play_type'] in ('run', 'pass')
            and not pd.isna(row['down']) and pd.to_numeric(row['down'], errors='coerce') == 4
            and "Touchdown" not in outcome
            and not pd.isna(row.get('yards_gained')) and not pd.isna(row['togo'])):
        try:
            if float(row['yards_gained']) < float(row['togo']):
                return True
        except (TypeError, ValueError):
            pass
    return False
 
# Add columns
result_df['yards_gained'] = result_df['description'].apply(extract_yards_gained)
result_df['penalty_yards'] = result_df['description'].apply(extract_penalty_yards)
result_df['penalty_team'] = result_df['description'].apply(extract_penalty_team)
result_df['turnover'] = result_df.apply(determine_turnover, axis=1)
 
# Vectorized success column (NFL: 40% of togo on 1st, 60% on 2nd, 100% on 3rd/4th)
down_num = pd.to_numeric(result_df['down'], errors='coerce')
togo_num = pd.to_numeric(result_df['togo'], errors='coerce')
threshold = np.where(down_num == 1, 0.4 * togo_num,
            np.where(down_num == 2, 0.6 * togo_num,
            np.where(down_num.isin([3, 4]), togo_num, np.nan)))
runpass = result_df['play_type'].isin(['run', 'pass'])
applicable = runpass & result_df['yards_gained'].notna() & down_num.notna() & togo_num.notna()
success = pd.Series([None] * len(result_df), index=result_df.index, dtype=object)
success_bool = result_df['yards_gained'] >= threshold
success[applicable] = success_bool[applicable].astype(bool)
result_df['success'] = success
 
print(f"Added columns. Shape: {result_df.shape}")
print(f"yards_gained non-null: {result_df['yards_gained'].notna().sum()}")
print(f"Run/pass success rate: {(result_df['success'] == True).sum() / (result_df['success'].notna()).sum() * 100:.1f}%")
print(f"Total turnovers: {result_df['turnover'].sum()}")
print(f"Total penalty yards: {result_df['penalty_yards'].sum()}")
 
# Save updated CSV

# =========================================================
# Add the score at the time of each play
# =========================================================
raw_scores = df[["gameid", "description", "offscore", "defscore"]].drop_duplicates(subset=["gameid", "description"])
result_df = result_df.merge(raw_scores, on=["gameid", "description"], how="left")

# =========================================================
# Possessions (drives) and whether each one scored
# =========================================================
result_df = result_df.sort_values(
    ['gameid', 'qtr', 'min', 'sec'],
    ascending=[True, True, False, False]
).reset_index(drop=True)

qtr_num = pd.to_numeric(result_df['qtr'], errors='coerce')
result_df['half'] = np.where(qtr_num <= 2, 1, np.where(qtr_num <= 4, 2, 3))

drive_types = ['run', 'pass', 'field_goal', 'punt', 'penalty_no_play']
result_df['is_drive_play'] = result_df['play_type'].isin(drive_types)

drive_df = result_df[result_df['is_drive_play']].copy()
drive_df['prev_off'] = drive_df.groupby('gameid')['off'].shift(1)
drive_df['prev_half'] = drive_df.groupby('gameid')['half'].shift(1)
drive_df['new_possession'] = (
    (drive_df['off'] != drive_df['prev_off']) |
    (drive_df['half'] != drive_df['prev_half'])
)
first_indices = drive_df.groupby('gameid').head(1).index
drive_df.loc[first_indices, 'new_possession'] = True

drive_df['possession_num'] = drive_df.groupby('gameid')['new_possession'].cumsum()
drive_df['possession_id'] = (
    drive_df['gameid'].astype(str)
    + '_p' + drive_df['possession_num'].astype(int).astype(str).str.zfill(2)
    + '_' + drive_df['off'].astype(str)
)

# Determine success: the offense scored a TD or made a FG
# (defensive touchdowns and safeties do not count as the offense scoring)
drive_df['_score'] = drive_df['offense_scored']
success_map = drive_df.groupby('possession_id')['_score'].any()
drive_df['possession_successful'] = drive_df['possession_id'].map(success_map)

# Merge back into result_df
result_df['possession_id'] = None
result_df['possession_successful'] = None
result_df.loc[drive_df.index, 'possession_id'] = drive_df['possession_id'].values
result_df.loc[drive_df.index, 'possession_successful'] = drive_df['possession_successful'].values

total_possessions = drive_df['possession_id'].nunique()
successful = drive_df.groupby('possession_id')['possession_successful'].first().sum()
print(f"Total possessions: {total_possessions}")
print(f"Successful (TD or FG): {int(successful)} ({successful/total_possessions*100:.1f}%)")

# Save back
result_df = result_df.drop(columns=['half', 'is_drive_play', 'offense_td', 'offense_scored'])
result_df.to_csv('parsed_nfl_plays_2002_2011.csv', index=False)
print("\nSaved updated parsed_nfl_plays_2002_2011.csv with possession columns")

# Preview
sample = result_df[result_df['gameid'] == result_df['gameid'].iloc[0]]
print("\nFirst 15 plays of one game:")
print(sample[['gameid', 'qtr', 'off', 'play_type', 'outcome', 'possession_id', 'possession_successful']].head(15).to_string(index=False))
