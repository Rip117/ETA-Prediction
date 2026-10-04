# Midnight Express

Short-horizon forecasting of Indian Railways train delays, built on live
delay data from NTES (National Train Enquiry System).

## What this project does

NTES shows how late each train is right now, but keeps no history. This
project collects delay snapshots every ~20 minutes, joins them with the
train timetable, and tests whether a model can forecast a train's delay
at the next snapshot better than the simplest possible rule:
"it will be as late as it was last time" (persistence).

## Project structure

```
src/
  Components/
    data_ingestion.py       builds (previous -> now) rows, time-based split
    data_transformation.py  imputing, scaling, one-hot encoding
    model_trainer.py        trains models, compares them with persistence
  Pipeline/                 (prediction pipeline, not written yet)
  logger.py                 timestamped log files in logs/
  exception.py              custom exception with file and line number
  utils.py                  helper functions (save_object)
scripts/
  collect_delays.py         polls NTES and appends to data/delay_history.csv
  backup_data.py            Backs Up the .csv file under data/delay_history.csv
notebooks/
  01_eda.ipynb              exploration and experiments
```

## Setup

Tested with Python 3.13 on Windows. Create a virtual environment and install
the dependencies:

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` contains what the pipeline needs to run. If you also want
to open the notebooks in `notebooks/` and make plots, install the development
extras instead (this includes everything in `requirements.txt`):

```
pip install -r requirements-dev.txt
```


## Reproducing the results

The data is not in this repository, so you have to collect it yourself.

**1. Get the timetable (once).** Clone [railpull](https://github.com/shwetankg07/railpull)
into the project root and follow its README:

```
git clone https://github.com/shwetankg07/railpull.git
cd railpull
pip install -r requirements.txt
python ntes/crawl.py
python transform/export.py
cd ..
```

The crawl takes a few hours and is resumable. It writes
`railpull/data/out/trains.csv`.

**2. Collect delay snapshots.** From the project root:

```
python scripts/collect_delays.py
```

Leave it running. It appends to `data/delay_history.csv` every 20 minutes
or so. At least one week of data is recommended.

**3. Train and evaluate.**

```
python -m src.Components.model_trainer
```

This runs ingestion, transformation and training, then prints the MAE of
each model next to the persistence benchmark. Logs go to `logs/`, outputs
to `artifacts/`.

## Method

- **Target:** `delta = delay_now - delay_at_previous_snapshot`. Predicting
  the change makes the benchmark simple: persistence predicts `delta = 0`.
- **Features:** previous delay, recent trend, time since the previous
  snapshot, hour, weekday, train type, distance, stops, travel time,
  number of running days, scheduled average speed.
- **Split:** by time. The model trains on earlier snapshots and is
  tested on later ones.
- **Metric:** MAE in minutes, always reported next to persistence.

## Results

Not Enough Data Yet

| Model | MAE (min) |
|---|---|
| Persistence | ... |
| Gradient Boosting | ... |

Findings so far:
- Timetable features alone had no predictive power on trains the model
  had not seen (about 58 vs 57 min MAE against a constant baseline).
- A random row split gave a misleadingly good score because the model
  memorized individual trains.
- On the first few snapshots, no model beat persistence by a meaningful
  margin.

## Limitations

- The poller only lists trains that are at least ~5 minutes late. On-time
  trains and trains that recover are missing, so results describe delayed
  trains only.
- Results cover only the period during which data was collected, with no
  seasonality (fog, monsoon, festivals).
- Forecast horizon is one snapshot (about 20 minutes).

## Data and responsible use

Data comes from NTES through the unofficial `ntes-client` library, via
railpull. Collection is deliberately slow. Data is not redistributed in
this repository. See railpull's "Responsible use" section before
collecting.

## Credits

Crawler and delay poller: [railpull](https://github.com/shwetankg07/railpull).
