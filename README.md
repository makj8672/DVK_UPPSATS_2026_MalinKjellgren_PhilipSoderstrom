# DVK_UPPSATS_2026_MalinKjellgren_PhilipSoderstrom

This repository contains the Python implementation for the bachelor’s thesis *Logistic Regression-Enhanced Rule-Based Trading with Technical Indicators: An Empirical Study on XAU/USD*, addressing the research question:

> What impact does logistic regression have on the financial performance of a rule-based trading strategy based on technical indicators, applied to the XAU/USD market, evaluated across historical backtesting?

Public repository (as referenced in the thesis): [github.com/makj8672/DVK_UPPSATS_2026_MalinKjellgren_PhilipSoderstrom](https://github.com/makj8672/DVK_UPPSATS_2026_MalinKjellgren_PhilipSoderstrom)


## Requirements

- **Python 3.11.9** (version used in the thesis; 3.11.x should work)
- **MetaTrader 5** desktop terminal (for **Option 1** and **Option 3** live data)
- Python packages: `pandas`, `numpy`, `scikit-learn`, `MetaTrader5`, `ta` (install with the command below)

Package versions are **not** pinned in the thesis: use whatever current releases work with **Python 3.11.9** (or 3.11.x) on your install date. For a fully frozen environment later, run `pip freeze` in a known-good venv and archive that list if you need exact replication.

## Installation

From the repository root, create a virtual environment, then install dependencies:

```bash
python -m venv .venv
```

**Windows (PowerShell):**

```powershell
.\.venv\Scripts\Activate.ps1
pip install pandas numpy scikit-learn MetaTrader5 ta
```

**macOS / Linux:**

```bash
source .venv/bin/activate
pip install pandas numpy scikit-learn MetaTrader5 ta
```

For live fetches, start MetaTrader 5, log in to an account with **XAUUSD** history available, then run the script.

## Usage

Edit `main.py` so exactly one data source path is active (see comments at the top of the `if __name__ == "__main__":` block):

- **Option 1** — Fetch live hourly data from MetaTrader 5 (default)
- **Option 2** — Load the frozen study snapshot (`snapshots/mt5_snapshot_20260416_133057Z.csv`) to reproduce thesis results without MT5 data at run time
- **Option 3** — Fetch live data once from MetaTrader 5, then save a new timestamped CSV under `snapshots/`

Run the experiment from the repository root:

```bash
python main.py
```

The script prints tuning tables, backtest summaries for the rule-based baseline and the logistic-regression-filtered strategy, and the probability-interval table.

## Snapshot

The file `snapshots/mt5_snapshot_20260416_133057Z.csv` is the dataset snapshot used for the reported thesis results. Use **Option 2** in `main.py` to rerun the pipeline on that exact file.

## AI assistance

`snapshot_io.py` was refactored with assistance from the Cursor AI-powered code editor, the final version was reviewed and integrated by the authors. 
