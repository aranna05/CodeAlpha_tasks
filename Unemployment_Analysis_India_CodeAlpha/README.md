# Unemployment in India — COVID-19 Impact Analysis

A reproducible Python data-science project based on the two CSV files supplied in `archive.zip`.

## Deliverables
- `Unemployment_Analysis_India.ipynb` — guided, presentation-ready notebook.
- `unemployment_analysis.py` — script version of the analysis.
- `Unemployment in India.csv` and `Unemployment_Rate_upto_11_2020.csv` — original source files.
- `outputs/` — generated figures and cleaned datasets (created when the script runs).

## Run it
```bash
python -m pip install pandas numpy matplotlib seaborn jupyter
jupyter notebook Unemployment_Analysis_India.ipynb
```
Or:
```bash
python unemployment_analysis.py
```

## Analytical design
1. Clean whitespace, dates, numeric fields, blank records and duplicates.
2. Keep the two source files separate because their geographic granularity differs.
3. Reconstruct a weighted monthly rate using an implied unemployed count:
   `unemployed = employed * rate / (100 - rate)`.
4. Compare pre-COVID (through February 2020) with March 2020 onward.
5. Explore rural/urban differences, state-level April 2020 hotspots, and labour-force participation.
6. Translate descriptive patterns into policy options while stating limitations.

## Important interpretation notes
- The reconstructed national series is an approximation, not an official national statistic.
- The March 2020 split is descriptive; it does not identify a causal COVID effect.
- The area-level file spans only 14 months, insufficient to establish recurring annual seasonality.
- The data ends in October 2020 and should not be used to describe today's unemployment situation.
