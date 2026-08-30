# pandas cheat sheet

*One page for mid-exercise lookups. Taught in [Module 02](../../modules/02-pandas-essentials/README.md) (basics) and Module 03 (wrangling — see the [curriculum](../curriculum.md#module-03--data-wrangling-with-pandas)). `df` is always a DataFrame.*

## Load

```python
import pandas as pd
from datasets import load_dataset

# HuggingFace dataset -> pandas table (the course's standard loading move)
df = load_dataset("SIH/palmer-penguins", split="train").to_pandas()

df = pd.read_csv("file.csv")     # or: read a local CSV file
```

## Inspect (the first-look ritual — run on every new dataset)

```python
df.head()          # first 5 rows, to see what the data looks like
df.shape           # (number of rows, number of columns)
df.info()          # every column's type + how many non-missing values
df.describe()      # mean, min, max, quartiles for each numeric column
```

## Select

```python
df["body_mass_g"]                    # one column (a Series)
df[["species", "body_mass_g"]]       # several columns (note the double brackets: a list inside)
df.loc[3]                            # the row whose index LABEL is 3
df.iloc[0]                           # the first row by POSITION (iloc = integer location)
df.loc[0:4, "species"]               # rows 0-4 of one column (loc includes both endpoints)
```

## Filter (boolean masks)

```python
df[df["species"] == "Gentoo"]                 # rows where the condition is True
df[df["body_mass_g"] > 5000]                  # numeric condition
# combine conditions with & (and) / | (or) -- parentheses are required:
df[(df["species"] == "Gentoo") & (df["sex"] == "female")]
big = df[df["body_mass_g"] > 5000].copy()     # .copy() before adding columns to a filtered table
```

## Missing values

```python
df.isna().sum()                          # count missing values per column
df.dropna()                              # drop every row that has any missing value
df.dropna(subset=["body_mass_g"])        # drop rows missing only in this column
df["sex"] = df["sex"].fillna("unknown")  # fill missing values with a stand-in
```

## New columns

```python
# arithmetic on columns works element by element, like NumPy arrays:
df["mass_kg"] = df["body_mass_g"] / 1000
df["bill_ratio"] = df["bill_length_mm"] / df["bill_depth_mm"]
```

## Counting categories

```python
df["species"].value_counts()             # how many rows of each species
pd.crosstab(df["species"], df["island"]) # 2-way count table: species x island
```

## Group-by (split → apply → combine)

```python
df.groupby("species")["body_mass_g"].mean()      # mean mass per species
df.groupby("species").size()                     # rows per group
# several statistics at once with .agg (short for aggregate):
df.groupby("species")["body_mass_g"].agg(["mean", "std", "count"])
df.groupby(["species", "sex"])["body_mass_g"].mean()   # group by two columns
```

## Merge (combine two tables on a shared column)

```python
# keep only rows whose 'country' appears in BOTH tables ("inner" join):
merged = pd.merge(cases, populations, on="country", how="inner")
# how="left" instead: keep every row of the first table, blanks where no match
```

## Pivot & melt (reshape between long and wide)

```python
# long -> wide: one row per date, one COLUMN per country
wide = df.pivot(index="date", columns="country", values="new_cases")

# wide -> long: back to one row per (date, country) pair
long = wide.reset_index().melt(id_vars="date",
                               var_name="country", value_name="new_cases")
```

## Dates

```python
df["date"] = pd.to_datetime(df["date"])   # text -> real datetime type
df["date"].dt.year                        # .dt unlocks date parts: .month, .day too
df = df.sort_values("date")               # rolling windows need time order
# 7-day rolling average -- the smoothing every COVID dashboard used:
df["smoothed"] = df["new_cases"].rolling(window=7).mean()
```

## Plot

```python
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(figsize=(8, 5))        # make an empty figure + axes to draw on
df["body_mass_g"].hist(ax=ax, bins=30)        # quick histogram of one column
df.plot(x="date", y="smoothed", ax=ax)        # line plot straight from the table

# seaborn: prettier statistical plots; hue= means "one color per category"
sns.scatterplot(data=df, x="flipper_length_mm", y="body_mass_g",
                hue="species", ax=ax)
sns.boxplot(data=df, x="species", y="body_mass_g", ax=ax)
ax.set_title("State the finding, not the variable names")
```
