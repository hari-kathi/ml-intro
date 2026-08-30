# Module 03 — Exercises

Work in a fresh cell at the bottom of `notebook.ipynb`, so everything the lesson built is already loaded: `covid` (all countries, cleaned), `four`, `us`, `wide`, `wide_smooth`, `long_smooth`, `merged`, `pops`, and `deaths_30`. Easiest first.

## 1. Date practice

- What day of the week was the very first date in `covid`? (Hint: `covid["date"].min()` gives one `Timestamp`, and a single Timestamp has `.day_name()` directly — no `.dt` needed, since `.dt` is only for whole columns.)
- Add a column `month` to `covid` using `.dt.month`. Across all countries and years combined, which calendar month has the most reported cases? (A `groupby` on your new column.)
- How many days does the dataset cover? Subtracting two dates (`covid["date"].max() - covid["date"].min()`) gives a **Timedelta** — a length of time — with a `.days` attribute.

## 2. Date filtering

- Using masks with `pd.Timestamp`, count South Africa's reported cases in December 2021 (its Omicron month) and December 2020. Roughly how many times larger was Omicron?
- The lesson's window stopped at March 2023. Find the last date on which the United States reported a *nonzero* `new_cases` value. (Filter to the US and `new_cases > 0`, then take the date column's `.max()`.)

## 3. Rolling averages

- Japan is not in `FOCUS`. Build a table of Japan's rows sorted by date, add a `cases_smoothed` column with a 7-day rolling mean, and plot raw vs smoothed — the same before/after figure the lesson made for the US.
- On the same axes, add a **28-day** rolling average of Japan's cases in a third color. What does the wider window smooth away that the 7-day window kept? When might a scientist prefer each?

## 4. Pivot and melt

- Build `wide_deaths`: pivot `four` with dates as the index, countries as columns, and `new_deaths` as the values. Check its `shape` — it should match `wide`.
- Smooth all four columns with one `.rolling(7).mean()` and plot the four death curves with the lesson's `COUNTRY_COLORS` loop. Does the US Omicron *deaths* peak look as dominant as its *cases* peak did?
- Melt your smoothed table back to long format (remember `reset_index()` first). Confirm the shape is 4,384 rows × 3 columns.

## 5. Merge practice

- `deaths_30` already holds total deaths, population, and continent for 30 countries. Sort it by `deaths_per_million`, descending. Which three countries reported the most deaths per person? Is your intuition about "rich country = safe country" holding up?
- Now compute each of the 30 countries' total reported **cases** per million (start from `covid`, `groupby` country to sum `new_cases`, merge with `pops`, divide). Which country tops this list, and is it the same one that topped deaths per million?
- After your merge, run the post-merge safety check from the lesson. How many of the 237 countries got `NaN`? Why is `how="left"` still the right choice here (what would you have *not seen* with an inner merge that silently dropped them)?

## 6. Challenge: your own per-million comparison

Pick **four countries of your own** from the 30 in `pops` (choose at least one from each of three different continents) and rebuild the lesson's per-million reveal for them, end to end:

1. Filter `covid` to your countries and the lesson's date window (`.isin`, plus the `START`/`END` timestamps).
2. Pivot to wide, smooth everything with one `.rolling(7).mean()`, melt back to long.
3. Merge in `pops` — and run the safety check.
4. Compute `cases_per_million`.
5. Plot the four smoothed per-million curves with `sns.lineplot`, one fixed color per country.

Finish it like a scientist: a title that states the *finding* ("X's waves were the tallest per person, but Y's lasted longer"), labeled axes with units, and one or two sentences in a markdown cell interpreting what you see — including one reason the comparison might still be unfair even after normalizing by population.
