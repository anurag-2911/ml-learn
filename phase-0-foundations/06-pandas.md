# 06 · Pandas: Interrogating Real Datasets

**Phase 0 — Foundations** · Estimated time: 1 week · Prerequisites: [05 · NumPy: Thinking in Arrays](05-numpy.md), [04 · Classes, Files, JSON and Errors](04-python-oop-files-errors.md)

> About 80% of real machine-learning work is not building models. It is loading messy data, cleaning it, filtering it and asking it questions, and pandas is the tool everyone uses for that. This lesson uses pandas to interrogate the real passenger list of the Titanic, dig through personal data, and rescue a deliberately broken dataset. The Titanic data in this lesson is exactly the data fed to the first real ML models in lessons [11](../phase-2-classical-ml/11-first-model-knn.md), [13](../phase-2-classical-ml/13-logistic-regression.md) and [15](../phase-2-classical-ml/15-decision-trees.md), so get to know it well.

## What this lesson builds

- **Project 1 — Titanic interrogation:** a script (or notebook) that answers 12 concrete questions about the Titanic passengers with code, each answer printed with a label.
- **Project 2 — Personal data:** an analysis of a personal dataset (expenses, YouTube history, or a study log) plus a short written file with 5 insights found in it.
- **Project 3 — Messy data rescue:** a cleaning script that turns a deliberately broken CSV into a tidy, correctly-typed DataFrame.

## Concepts covered

- **DataFrame**: pandas' main object, a table with named columns, like a spreadsheet in code.
- **Series**: a single column of a DataFrame (one-dimensional, with a name).
- **read_csv / head / info / describe**: loading a table and getting a first look at it.
- **Selecting rows and columns**: picking out exactly the slice of the table that is needed, with `loc` (by label) and `iloc` (by position).
- **Boolean filtering**: keeping only the rows where some condition is true.
- **groupby + aggregation**: splitting rows into groups (e.g. by sex) and computing one number per group (e.g. mean survival).
- **sorting and value_counts**: ranking rows and counting how often each value appears.
- **Missing values**: finding (`isna`), filling (`fillna`) and dropping (`dropna`) the holes in real data.
- **Creating new columns**: computing new information from existing columns.

## Before starting

Requirements: the venv from [lesson 01](01-environment-setup.md) and comfort with NumPy arrays from [lesson 05](05-numpy.md). A DataFrame is essentially a dictionary of NumPy arrays with labels bolted on.

```bash
cd ~/ml/ml-learn
source .venv/bin/activate
pip install pandas seaborn
mkdir -p work/06-pandas
cd work/06-pandas
```

Seaborn is a plotting library (featured in the next lesson), but this lesson uses it for only one thing: it can download small classic datasets with no signup. Check that everything works. The first call downloads the data, so an internet connection is needed once; after that, the data is cached:

```bash
python3 -c "import seaborn as sns; df = sns.load_dataset('titanic'); print(df.shape)"
```

**Checkpoint:** it prints `(891, 15)`, which means 891 passengers and 15 columns.

Any of these ways of working is fine: a plain `.py` file run with `python3`, the interactive `python3` prompt, or a Jupyter notebook, if one was set up in lesson 01. Notebooks are especially useful in this lesson, because the work involves constantly looking at tables.

## Project 1 — Titanic interrogation

**Goal:** Answer 12 concrete questions about the Titanic passengers, each with a few lines of pandas. Put them in `work/06-pandas/titanic_questions.py` (or a notebook), printing each answer with a label like `Q3: survival rate by sex = ...`.

Start with this and build up:

```python
import seaborn as sns
import pandas as pd

df = sns.load_dataset("titanic")   # df is a DataFrame
print(df.head())                   # first 5 rows
print(df.info())                   # column names, types, missing counts
print(df.describe())               # stats for the numeric columns
```

Key columns: `survived` (1 = survived, 0 = died), `pclass`/`class` (ticket class), `sex`, `age`, `fare` (ticket price), `sibsp`/`parch` (siblings+spouses / parents+children aboard), `embark_town`, `alone`.

**Milestones**

- [ ] **Q1 — First look.** Load the data; use `head()`, `info()` and `describe()`. Write down (as comments) three things that stand out. Checkpoint: `info()` shows 891 rows and that `age` and `deck` have fewer than 891 non-null values.
- [ ] **Q2 — Overall survival rate.** What fraction of passengers survived? (Hint from lesson 05: the mean of a 0/1 column is the fraction of 1s.) Checkpoint: about 0.38.
- [ ] **Q3 — Survival by sex.** Use `df.groupby("sex")["survived"].mean()`. Checkpoint: female about 0.74, male about 0.19. The "women and children first" order is visible in the data.
- [ ] **Q4 — Survival by class.** Same idea with `pclass`. Checkpoint: first class about 0.63, third class about 0.24.
- [ ] **Q5 — Age of survivors vs non-survivors.** Group by `survived`, take the mean of `age`. Checkpoint: survivors about 28.3 years, non-survivors about 30.6.
- [ ] **Q6 — Children.** Using boolean filtering (`df[df["age"] < 10]`), how many passengers were under 10, and what was their survival rate? Checkpoint: around 60 children, with a survival rate clearly above the overall 0.38.
- [ ] **Q7 — Where did people board?** Use `value_counts()` on `embark_town`. Checkpoint: Southampton is by far the biggest, with 644 passengers.
- [ ] **Q8 — The most expensive ticket.** Sort by `fare` descending with `sort_values` and look at the top rows with `head()`. Checkpoint: the maximum fare is about 512.33, and three passengers paid it.
- [ ] **Q9 — The oldest passengers.** Show the 10 oldest passengers (their age, sex, class and whether they survived). Checkpoint: the oldest passenger was 80 and survived.
- [ ] **Q10 — Missing values audit.** Use `df.isna().sum()` to count missing values per column. Checkpoint: `age` is missing 177 values and `deck` is missing 688. With so many gaps, `deck` is nearly useless.
- [ ] **Q11 — Fill the age gap.** Make a copy (`df2 = df.copy()`), fill missing ages with the median age using `fillna`, and verify with `isna().sum()`. Checkpoint: `df2["age"]` has 0 missing values, and its median is unchanged.
- [ ] **Q12 — Family size.** Create a new column `family_size = df["sibsp"] + df["parch"] + 1`. Compare survival rates of people travelling alone vs with family (the `alone` column, or `family_size == 1`). Checkpoint: alone about 0.30, with family about 0.51.
- [ ] **Bonus question — the full picture.** Group by *two* columns at once: `df.groupby(["sex", "pclass"])["survived"].mean()`. Checkpoint: first-class women survived at about 0.97; third-class men at about 0.14. One line of code is enough to show the social structure of the tragedy.
- [ ] **Save the dataset for lesson 07.** Write the DataFrame to disk next to the script: `df.to_csv("titanic.csv", index=False)` (`index=False` stops pandas writing the row numbers as an extra column). The next lesson's charts read this exact file. Checkpoint: `work/06-pandas/titanic.csv` exists, and loading it back with `pd.read_csv("titanic.csv")` shows the familiar `(891, 15)` shape.

<details><summary>Hints</summary>

- `df["age"]` gives a Series; `df[["age", "fare"]]` (double brackets) gives a smaller DataFrame. Most "wrong shape" confusion comes down to this.
- Boolean filtering: `df[df["sex"] == "female"]` keeps the rows where the condition is true. Combine conditions with `&` and `|`, and wrap each side in parentheses: `df[(df["age"] < 10) & (df["survived"] == 1)]`.
- `loc` selects by label/condition, `iloc` by row number: `df.loc[df["fare"] > 500, ["sex", "age", "fare"]]` vs `df.iloc[0:5]`. For "rows where X, columns Y, Z" in one go, use `loc`.
- `groupby` returns a grouped object that does nothing until it is aggregated: follow it with `.mean()`, `.sum()`, `.count()` or `.agg(["mean", "count"])`.

</details>

**Definition of done:** running the script prints all 12 labeled answers, each checkpoint number matches, and `titanic.csv` sits in the work folder, ready for lesson 07.

## Project 2 — Personal data

**Goal:** Apply the same interrogation to *personal* data, and write up 5 insights. Analyzing personally meaningful data is where pandas stops being homework and becomes a practical tool.

Pick ONE data source:

1. **Bank or expense CSV**: most banking apps can export transactions as CSV (comma-separated values: a plain-text table, one row per line). It is the best option if available.
2. **YouTube watch history**: search for "Google Takeout" and export the YouTube history (it arrives as JSON; use `pd.read_json` or the `json` module from lesson 04).
3. **Hand-made study log**: no exports? Make one by hand. Spend 15 minutes writing a CSV of the ML study done so far, one row per session:

```csv
date,lesson,minutes,focus
2026-09-01,02-python-basics,90,high
2026-09-02,02-python-basics,45,low
2026-09-03,03-data-structures,120,high
```

Give it 25+ rows (reconstruct them from memory; approximate is fine, since real data is approximate too).

**Milestones**

- [ ] Load the CSV with `pd.read_csv("...")` and inspect it with `head()`, `info()`, `describe()`. Checkpoint: every column has the expected type. If a date or number column shows dtype `object` (pandas-speak for "text"), fix it (see hints).
- [ ] Clean what needs cleaning: parse dates with `pd.to_datetime`, drop or fill missing values, rename cryptic columns with `df.rename(columns={...})`.
- [ ] Ask at least 5 questions with `groupby`, boolean filtering, `sort_values` and `value_counts`. Examples: spending by category per month, the top 10 most-watched channels, the busiest weekday for study, the longest study streak.
- [ ] Create at least one new column that makes a question answerable, for example `df["weekday"] = df["date"].dt.day_name()` or an `is_weekend` True/False column.
- [ ] Write `work/06-pandas/insights.md`: 5 numbered insights, each one sentence of finding + the number that backs it ("I watched 3x more YouTube on weekends than weekdays: 41 vs 13 videos/day on average").

<details><summary>Hints</summary>

- `pd.read_csv("file.csv", parse_dates=["date"])` parses dates while loading. Once a column is a real datetime, the `.dt` accessor unlocks `.dt.month`, `.dt.day_name()`, and more.
- Money columns sometimes load as text because of currency symbols or commas: `df["amount"].str.replace(",", "").str.replace("€", "").astype(float)`.
- Group by month with `df.groupby(df["date"].dt.to_period("M"))["amount"].sum()`.
- If the bank CSV uses `;` as separator (common in Europe), pass `sep=";"` to `read_csv`.

</details>

**Definition of done:** `insights.md` exists with 5 insights, and every number in it was computed in code, not estimated by eye.

## Project 3 — Messy data rescue

**Goal:** Real-world CSVs are never clean. Take a deliberately broken expense log and turn it into a tidy DataFrame: correct types, no duplicates, no format chaos, missing values handled deliberately.

Save this as `work/06-pandas/messy_expenses.csv`. It is the starting material, warts and all:

```csv
name,category,amount,date
Coffee,food,3.50,2026-01-05
coffee  ,Food,3.50,2026-01-05
Groceries,FOOD,42.10,2026-01-06
Bus ticket,transport,,2026-01-06
GROCERIES,food,55.00,Jan 7 2026
Gym membership,health,25.00,2026/01/08
gym membership,Health,25.00,2026/01/08
Movie night,entertainment,12.50,09-01-2026
Rent,housing,850.00,2026-01-10
Coffee,food,3.75,Jan 11 2026
electricity,housing,64.30,2026-01-12
Coffee,,3.75,2026-01-13
Taxi,transport,18.00,14-01-2026
RENT,housing,850.00,2026-01-10
```

**Milestones**

- [ ] Load it and diagnose. Find as many problems as possible and list them as comments in `work/06-pandas/rescue.py`. Checkpoint: the comments name at least four kinds of problems (inconsistent capitalization, trailing spaces, missing values, duplicate rows, four different date formats).
- [ ] **Fix the text columns.** Use the `.str` accessor (string methods that work on a whole Series at once) to strip whitespace and normalize case, for example `df["category"] = df["category"].str.strip().str.lower()`. Checkpoint: `df["category"].value_counts()` shows exactly one entry per real category, and "Food"/"FOOD"/"food" have merged.
- [ ] **Fix the dates.** Convert the `date` column to real datetimes despite the mixed formats. Checkpoint: `df["date"].dtype` prints `datetime64[ns]` and `df["date"].min()` is January 5, 2026.
- [ ] **Handle missing values deliberately.** Decide per column and write the reasoning down as a comment: the missing `category` can be filled ("food", since it is coffee), but is a missing `amount` fillable, or should that row be dropped? There is no single right answer; having a *reason* is the skill.
- [ ] **Drop the duplicates.** Use `df.duplicated()` to inspect and `df.drop_duplicates()` to remove, but watch for the hidden ones: after the text cleaning, "Coffee/food" and "coffee  /Food" became identical, and the duplicated rent should go too. Checkpoint: 11 rows remain.
- [ ] **Prove it is tidy.** Save with `df.to_csv("clean_expenses.csv", index=False)`, reload it, and run a groupby: total spend per category. Checkpoint: the reloaded frame has 11 rows, no missing values except any kept on purpose, and housing is the biggest category at over 900.

<details><summary>Hints</summary>

- `pd.to_datetime(df["date"], format="mixed", dayfirst=True)` handles multiple formats in one column (pandas 2.x). `dayfirst=True` tells pandas that `14-01-2026` means January 14, not month 14.
- Order of operations matters: clean the text *before* dropping duplicates, or the disguised duplicates survive.
- `df.duplicated(keep=False)` marks *all* copies of a duplicate (not just the later ones), which is useful for checking by eye before deleting.
- Chain `.str` calls freely: `.str.strip().str.lower()`. Each call returns a new Series.

</details>

**Definition of done:** `rescue.py` runs top to bottom, turns the messy CSV into `clean_expenses.csv` with 11 tidy rows, and contains a comment explaining every cleaning decision.

## Stretch goals

- **Pivot tables:** redo the Titanic bonus question with `df.pivot_table(values="survived", index="sex", columns="pclass")`. It gives the same numbers, arranged as a 2×3 grid. Compare them with the groupby result.
- **Age bands:** use `pd.cut(df["age"], bins=[0, 12, 18, 40, 60, 80])` to bucket Titanic passengers into age groups, then compute survival per group. Does "children first" hold in every class?
- **Merge two tables:** split the Project 2 data into two CSVs sharing a key column (e.g. transactions + a category→budget table) and join them back with `pd.merge`. Merging tables is a daily task in real data work.
- **Speed check:** answer Titanic Q3 using only the pure-Python dicts and loops from lesson 03. Notice the difference in code length: that is why pandas exists.

## Getting unstuck

- **Read errors bottom-up.** The last line names the problem; `KeyError: 'Age'` almost always means a misspelled or wrongly-capitalized column name. Print `df.columns` to see the real names.
- **Look at the data, constantly.** Before and after every operation: `df.head()`, `df.shape`, `df.dtypes`. Most bugs are "the data was never what I assumed".
- **`SettingWithCopyWarning`:** pandas' famous warning when code writes into a filtered slice. The fix is usually one `loc`: `df.loc[df["age"] < 10, "group"] = "child"` instead of `df[df["age"] < 10]["group"] = "child"`.
- **Filtering with `and`/`or` raises "truth value is ambiguous":** use `&` and `|` with parentheses around each condition instead.
- Ask an AI assistant for a **hint, not a solution** ("why might groupby().mean() return NaN for one group?", not "write my cleaning script"), and type all code by hand. Muscle memory with pandas is precisely the skill this lesson is meant to build.

## Resources

- [10 minutes to pandas](https://pandas.pydata.org/docs/user_guide/10min.html) — the official whirlwind tour; skim before Project 1, revisit after.
- [Kaggle pandas course](https://www.kaggle.com/learn/pandas) — free interactive exercises in the browser; great as a Project 1 warm-up or for extra practice afterwards.
- The official pandas docs at pandas.pydata.org — bookmark it; the search box answers most "how do I..." questions.

## Skills unlocked

- [ ] I can load a CSV, and use `head`, `info` and `describe` to get oriented in under a minute.
- [ ] I can explain the difference between a DataFrame and a Series, and between `loc` and `iloc`.
- [ ] I can filter rows with boolean conditions, including combined ones with `&` and `|`.
- [ ] I can answer "average of X per group Y" questions with `groupby` without looking anything up.
- [ ] I can find missing values with `isna` and decide, with a reason, whether to fill or drop them.
- [ ] I can create new columns from existing ones, including from dates with `.dt` and text with `.str`.
- [ ] I can take a messy real-world CSV and produce a tidy, correctly-typed table from it.

## Next up

Numbers in tables answer questions, but pictures reveal patterns that nobody thought to ask about, so the next lesson turns these DataFrames into charts: [07 · Data Visualization: Charts That Answer Questions](07-data-visualization.md).
