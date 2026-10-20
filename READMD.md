# What Wins Football Games? A Play-by-Play Analysis of the NFL, 2002–2011

Code for the paper *What Wins Football Games? A Play-by-Play Analysis of the NFL, 2002–2011* by Bennett Huang.

## Data

The analysis uses Brian Burke's publicly available NFL play-by-play data for the 2002–2011 seasons (Advanced Football Analytics, 2010):
https://www.advancedfootballanalytics.com/2010/04/play-by-play-data.html

The data files are not included in this repository. Download the ten season files (`2002_nfl_pbp_data.csv` through `2011_nfl_pbp_data.csv`) and place them in the same folder as the scripts.

## Requirements

Python 3.13 and the package versions in `requirements.txt`:

```
pip install -r requirements.txt
```

## How to run

Run the two scripts in order:

1. `python parse_plays.py`
   Parses the written description of every play (play type, yards gained, penalties, turnovers, possessions) and saves `parsed_nfl_plays_2002_2011.csv`.
2. `python vaep_analysis.py`
   Trains the two VAEP models (probability the current drive scores, and probability the offense wins the game), assigns a win value to every play, and saves `figure2.png` and `figure3.png`. Model checks (AUC and Brier score) are printed to the screen.

The models use a fixed random seed and a single CPU thread, so results are identical on every run with the listed package versions. Other scikit-learn versions build slightly different models, so individual win values can differ by about 0.1 percentage points from the paper. Training takes several minutes.
