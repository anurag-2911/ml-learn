# 07 · Data Visualization: Charts That Answer Questions

**Phase 0 — Foundations** · Estimated time: 4-5 days · Prerequisites: [05 · NumPy](05-numpy.md), [06 · Pandas](06-pandas.md)

> A table of 891 rows tells you almost nothing at a glance. The right chart tells you the story in two seconds. In this lesson you learn the small set of chart types that cover 95% of real data work — and, more importantly, the discipline of asking a question first and choosing the chart that answers it. From this lesson onward, **every project in this curriculum should include at least one plot**: when your ML models misbehave later, plotting the data and the training curves is how you will debug them. Plots are not decoration; they are your eyes.

## What you will build

- **Project 1 — Chart sampler:** a notebook with 6 charts of the Titanic data, each headed by a question and followed by a one-sentence answer.
- **Project 2 — Weather story:** a 4-chart notebook telling the story of one year of weather in a city you choose, from real data you download yourself.
- **Project 3 — EDA mini-report:** a one-notebook exploratory report on a dataset of your choice: 5 questions, 5 charts, 5 answers — the exact ritual you will repeat at the start of every ML project.

## Concepts you will learn by doing

- The matplotlib **figure/axes** model — a *figure* is the whole canvas, an *axes* is one plot drawn on it
- The four workhorse charts: **line** (change over time), **bar** (compare categories), **scatter** (relationship between two numbers), **histogram** (shape of one number's distribution)
- Labels, titles and legends — the difference between a chart and a readable chart
- **seaborn** one-liners: `histplot`, `boxplot`, `heatmap`, `pairplot` — statistical charts in one call
- When to use which chart type (a small decision guide you will internalize by using it)
- Saving figures to PNG files with `savefig`
- The "one question per chart" discipline

## Before you start

Check you can do these (all from earlier lessons): load a CSV into a pandas DataFrame, select columns, filter rows, and use `groupby`. If `groupby` feels shaky, skim your lesson 06 work first.

Activate your venv (created in [lesson 01](01-environment-setup.md)) and install the plotting libraries:

```bash
cd ~/ml/ml-learn
source .venv/bin/activate
pip install matplotlib seaborn
```

Create your work folder:

```bash
mkdir -p work/07-data-visualization
```

**Where will the plots appear?** Work in Jupyter notebooks inside VS Code (as in lesson 06): plots render right under the cell. If you ever plot from a plain `.py` script instead, what `plt.show()` does depends on your system:

- **macOS:** it opens the chart in a window, and the script waits until you close it.
- **Windows (WSL2) and Linux:** it opens no window, because the Python 3.13 from lesson 01 comes without tkinter, the toolkit matplotlib uses for windows on Linux. Save to a file with `plt.savefig("myplot.png")` instead and open the PNG in VS Code. If you would rather have windows, install tkinter with `sudo apt install -y python3.13-tk` (on Fedora, `sudo dnf install -y python3.13-tkinter`) and run the script again — on Windows, the window then opens on your Windows desktop.

**Data:** reuse the Titanic CSV you saved at the end of lesson 06's Project 1 (copy it in: `cp work/06-pandas/titanic.csv work/07-data-visualization/`). If you don't have it anymore, seaborn bundles the same data — `sns.load_dataset("titanic")` returns it as a DataFrame (needs internet once). Project 2's weather download is explained inside that project.

## Project 1 — Chart sampler

**Goal:** Produce 6 charts of the Titanic data, each one answering a specific stated question — and learn the figure/axes model, the four workhorse chart types, and labeling along the way.

**Milestones**

- [ ] Create `work/07-data-visualization/01_chart_sampler.ipynb`. In the first cell import everything and load the data:

  ```python
  import pandas as pd
  import matplotlib.pyplot as plt
  import seaborn as sns

  df = pd.read_csv("titanic.csv")   # or: df = sns.load_dataset("titanic")
  df.head()
  ```

- [ ] Make your first plot the explicit way, so you meet the figure/axes model before any shortcuts:

  ```python
  fig, ax = plt.subplots(figsize=(8, 5))   # fig = the canvas, ax = one plot on it
  ax.hist(df["age"].dropna(), bins=30)
  ax.set_xlabel("Age (years)")
  ax.set_ylabel("Number of passengers")
  ax.set_title("Q1: How old were the Titanic passengers?")
  ```

  A *histogram* chops a numeric column into equal-width buckets ("bins") and draws a bar per bucket showing how many values fall in it — it shows the *shape* of one variable. Checkpoint: the tallest bars sit in the 20-30 age range, with a smaller bump for young children.
- [ ] Under the chart, write your answer in a markdown cell, one sentence ("Mostly young adults in their 20s-30s, plus a noticeable group of small children."). Do this for every chart in this lesson — a chart without its answer is homework left unfinished.
- [ ] **Chart 2 (bar):** "Q2: Did first-class passengers survive more often?" Compute `df.groupby("pclass")["survived"].mean()` and plot it as a bar chart (the Series method `.plot.bar(ax=ax)` works, or `ax.bar(...)`). A *bar chart* compares a number across a few categories. Label the y-axis "survival rate". Checkpoint: bars step downward — roughly 0.63 for 1st class, 0.47 for 2nd, 0.24 for 3rd.
- [ ] **Chart 3 (bar):** "Q3: Did 'women and children first' show up in the data?" Bar chart of survival rate by sex. Checkpoint: about 0.74 for women vs about 0.19 for men — a huge gap.
- [ ] **Chart 4 (scatter):** "Q4: Is there a pattern connecting fare, age and survival?" A *scatter plot* puts one dot per row, revealing the relationship between two numeric columns. Draw fare (y) vs age (x), colored by survival:

  ```python
  fig, ax = plt.subplots(figsize=(8, 5))
  scatter = ax.scatter(df["age"], df["fare"], c=df["survived"], alpha=0.6)
  ax.legend(*scatter.legend_elements(), title="Survived")
  ```

  `alpha=0.6` makes dots translucent so overlaps stay visible; the *legend* is the little box explaining what each color means — always add one when color carries meaning. Checkpoint: most dots crowd below fare 100, and the expensive tickets are mostly survivor-colored.
- [ ] **Chart 5 (boxplot):** "Q5: How does age differ across classes?" One seaborn line: `sns.boxplot(data=df, x="pclass", y="age")`. A *boxplot* summarizes a distribution as a box (middle 50% of values, line at the median) with whiskers for the rest — perfect for comparing distributions across categories. Checkpoint: the 1st-class median age is clearly the highest (money takes time to earn).
- [ ] **Chart 6 (heatmap):** "Q6: Which numeric columns move together?" A *heatmap* paints a grid of numbers as colors so patterns pop out. Feed it a *correlation matrix* — a table where each cell holds the correlation between two columns: a number from -1 to +1 measuring how strongly they move together (+1 they rise together, -1 one rises as the other falls, 0 unrelated), computed with `.corr()`:

  ```python
  corr = df[["survived", "pclass", "age", "sibsp", "parch", "fare"]].corr()
  sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
  ```

  Checkpoint: `pclass` vs `fare` is strongly negative (around -0.55) — lower class number, pricier ticket — and `pclass` vs `survived` is around -0.34.
- [ ] Save your favorite chart to a file: `fig.savefig("survival_by_class.png", dpi=150, bbox_inches="tight")`. Checkpoint: the PNG appears in your work folder and opens in VS Code.
- [ ] Add a final markdown cell: your own 4-line "when to use which chart" cheat sheet — trend over time → line; compare categories → bar; two numbers related? → scatter; shape of one number → histogram (boxplot to compare shapes, heatmap for a grid of numbers).

<details><summary>Hints</summary>

- `ax.hist` chokes on missing values in older matplotlib versions — `.dropna()` first. Lesson 06 taught you that `age` has missing entries; plotting is where that bites.
- Pandas objects plot themselves: `series.plot.bar(ax=ax)` draws onto an axes you made, so you keep control of labels and title.
- If two plots merge into one chart, you drew both onto the same axes — create a fresh `fig, ax = plt.subplots()` per chart.

</details>

**Definition of done:** a notebook with 6 labeled charts (title, axis labels, legend where color matters), each preceded by a question and followed by a one-sentence answer.

## Project 2 — Weather story

**Goal:** Download a year of real daily weather for a city you care about and tell its story in 4 charts — your first plots from data *you* fetched, with all its real-world mess.

**Milestones**

- [ ] Download the data. Open-Meteo's archive API serves historical weather as CSV with no signup. This command grabs 2024 for Berlin — swap in your own city's latitude/longitude (find them at https://open-meteo.com, or just search "\<your city> latitude longitude"):

  ```bash
  cd ~/ml/ml-learn/work/07-data-visualization
  curl "https://archive-api.open-meteo.com/v1/archive?latitude=52.52&longitude=13.41&start_date=2024-01-01&end_date=2024-12-31&daily=temperature_2m_max,temperature_2m_min,precipitation_sum&timezone=auto&format=csv" -o weather.csv
  ```

  Checkpoint: `wc -l weather.csv` shows roughly 370 lines.
- [ ] **Look at the raw file before parsing it** — a habit for life. Open `weather.csv` in VS Code. Notice the real column header row is not line 1: there are metadata lines above it. Load it in a new notebook `02_weather_story.ipynb` with `pd.read_csv("weather.csv", skiprows=N)` (count N yourself), then convert the time column: `df["time"] = pd.to_datetime(df["time"])`. The other column names carry their units, like `temperature_2m_max (°C)`; rename them to the short names the code below uses: `df.columns = ["time", "temperature_2m_max", "temperature_2m_min", "precipitation_sum"]`. Checkpoint: `df.dtypes` shows `datetime64` for time and `float64` for the temperature columns, and `len(df)` is 365 or 366.
- [ ] **Chart 1 (line):** daily max temperature across the whole year. A *line chart* connects points in time order — the default for anything measured over time: `ax.plot(df["time"], df["temperature_2m_max"])`. Question: "What did the year feel like?" Checkpoint: a broad seasonal wave — a hill or valley shape depending on hemisphere, not random noise.
- [ ] **Chart 2 (line, two series):** max and min temperature on the same axes — call `ax.plot` twice with `label="max"` / `label="min"`, then `ax.legend()`. Question: "How big is the daily swing, and does it change with season?"
- [ ] **Chart 3 (bar):** average max temperature per month. Make a month column (`df["time"].dt.month`), then `groupby("month")` and `.mean()` — the lesson 06 split-apply-combine move feeding a chart. Checkpoint: 12 bars with a clear seasonal shape (for most non-equatorial cities, warmest and coldest months differ by 10 °C or more).
- [ ] **Chart 4 (your call):** find and mark the hottest and coldest days. Get them with `df.loc[df["temperature_2m_max"].idxmax()]` (and `idxmin` on the min column), then design a chart that shows them — for example the year line with the two days marked using `ax.scatter` on top plus `ax.annotate` to write the dates on the chart. Checkpoint: two labeled points sitting exactly on the line's peak and trough.
- [ ] Top the notebook with a markdown title cell and finish with a 3-sentence "weather story" summarizing what the four charts revealed. Save chart 4 as `weather_story.png`.

<details><summary>Hints</summary>

- If `read_csv` gives one giant column or weird column names, your `skiprows` is off — recount the metadata lines in the raw file, including any blank line.
- X-axis date labels overlapping? `fig.autofmt_xdate()` rotates them.
- Two lines, one color, no legend? You forgot `label=` in `plot` or forgot to call `ax.legend()`.
- `idxmax()` returns the row *index* of the maximum — feed it to `df.loc[...]` to get the whole row, including the date.

</details>

**Definition of done:** a notebook that downloads nothing at run time (reads your local CSV), renders 4 labeled charts, and ends with a short written story a friend could read without seeing the code.

## Project 3 — EDA mini-report

**Goal:** Run your first full *EDA* — exploratory data analysis, the "interview the dataset before modeling it" step — on a dataset you pick yourself: 5 questions, 5 charts, 5 one-sentence answers. This is exactly the first hour of every ML project you will ever do.

**Milestones**

- [ ] Pick a dataset that genuinely interests you — curiosity is fuel. Easy no-signup options: any of seaborn's built-in sets (`sns.get_dataset_names()` lists them; `penguins`, `tips` and `diamonds` are good), another Open-Meteo download (compare two cities?), or any CSV from the wild. Aim for at least 300 rows and a mix of numeric and categorical columns.
- [ ] Create `03_eda_report.ipynb`. First interrogate without plotting, using your lesson 06 toolkit: `df.shape`, `df.head()`, `df.dtypes`, `df.describe()`, `df.isna().sum()`. In a markdown cell note the size, the column meanings, and any missing data. Checkpoint: you can say out loud what one row represents.
- [ ] **Before any chart, write your 5 questions** in one markdown cell. Real questions with unknown answers ("Do heavier penguins have longer flippers?" "Do smokers tip differently?"), not chart orders ("plot X"). This is the one-question-per-chart discipline — it stops the aimless plotting that eats afternoons.
- [ ] Get a fast overview with one seaborn line: `sns.pairplot(df, hue="<a category column>")` — a grid of scatter plots for every pair of numeric columns, colored by a category (histograms on the diagonal). Checkpoint: at least one panel makes you go "huh" — refine a question if so.
- [ ] Answer each question with exactly one chart, choosing the type deliberately from your Project 1 cheat sheet, plus seaborn's `histplot` and `boxplot` where they fit. Structure every one the same: markdown question → code cell → markdown one-sentence answer. Vary the types — a report with 5 histograms means the questions were too similar.
- [ ] Polish pass: every chart has a title, axis labels, and a legend if color carries meaning. Restart the kernel and "Run All" — everything runs clean top to bottom. Checkpoint: a friend could read only the markdown cells and charts and learn 5 true things.
- [ ] Commit all three notebooks to git (lesson 01 workflow): `git add`, `git commit -m "Lesson 07: data visualization projects"`.

<details><summary>Hints</summary>

- Struggling to find 5 questions? Use the templates: How is \<numeric\> distributed? Does \<numeric\> differ across \<category\>? Are \<numeric\> and \<numeric\> related? How does \<numeric\> change over time? Which \<category\> is most common?
- `pairplot` on many columns is slow and unreadable — pass 3-4 numeric columns via `vars=[...]`.
- A "boring" answer ("no relationship") is a valid, useful answer — say it in the sentence and move on. In ML, knowing what *doesn't* predict is half the job.

</details>

**Definition of done:** one notebook, 5 questions, 5 deliberately-chosen charts, 5 one-sentence answers, runs clean top to bottom.

## Stretch goals

- Rebuild your 6 Titanic charts as a single figure with `fig, axes = plt.subplots(2, 3, figsize=(18, 10))` — one dashboard image, saved as PNG.
- Restyle a whole notebook with one line — `sns.set_theme()` or `plt.style.use("ggplot")` — and note what changed.
- In the weather story, add a 7-day rolling-average line on top of the noisy daily line: `df["temperature_2m_max"].rolling(7).mean()`.
- Take the Kaggle data visualization micro-course (link below) and redo one of its exercises on *your* dataset instead of theirs.

## If you get stuck

- **Blank or missing plot?** In notebooks, the plot renders when the cell ends — make the plotting calls the last thing in the cell. From a `.py` script, `plt.show()` opens a window on macOS, but on Windows (WSL2) and Linux it opens none unless you installed tkinter (see Before you start), and may warn `FigureCanvasAgg is non-interactive, and thus cannot be shown`. There, don't fight `plt.show()`: `savefig` to PNG and open it in VS Code.
- **Two charts on top of each other?** You reused an axes. One `fig, ax = plt.subplots()` per chart.
- **A mysterious `Text(0.5, ...)` line under your plot** is just the return value of the last call, not an error — ignore it or end the cell with `plt.show()`.
- **Seaborn errors about a column** usually mean a name mismatch — print `df.columns` and check spelling and case (your Titanic columns are all lowercase: `age`, not `Age`; a Titanic CSV from Kaggle capitalizes them).
- Standing advice: read the error message bottom-up (the last line names the problem); print the thing you're plotting (`df["age"].head()`, `.shape`, `.dtype`) before blaming the chart; ask an AI assistant for a **hint**, not a solution; and type all code yourself — no pasting.

## Resources

- [Matplotlib quick start](https://matplotlib.org/stable/users/explain/quick_start.html) — the figure/axes model from the source; read after Project 1 and it will all click.
- [Seaborn tutorial](https://seaborn.pydata.org/tutorial.html) — gallery-style tour of every seaborn plot; skim to know what exists, return when you need one.
- [Kaggle: Data Visualization](https://www.kaggle.com/learn/data-visualization) — free short course with in-browser exercises; good structured practice after Project 3 (account needed for the exercises).

## Skills unlocked

- [ ] I can explain the difference between a figure and an axes, and create both with `plt.subplots()`
- [ ] I can pick the right chart for a question: line vs bar vs scatter vs histogram — and say why
- [ ] I can make any chart readable: title, axis labels, legend
- [ ] I can use seaborn's `histplot`, `boxplot`, `heatmap` and `pairplot` on a DataFrame
- [ ] I can download a real CSV from an API with `curl` and handle its quirks (metadata rows, dates as strings)
- [ ] I can chain `groupby` into a chart to answer a "compare across categories" question
- [ ] I can save a publication-ready PNG with `savefig`
- [ ] I can run a 5-question EDA on a dataset I've never seen

## Next up

You now have the full Python data toolkit — next you build the math engine underneath ML by coding it yourself: [08 · Linear Algebra by Writing Your Own](../phase-1-math/08-linear-algebra-by-code.md).
