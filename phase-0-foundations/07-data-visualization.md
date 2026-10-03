# 07 · Data Visualization: Charts That Answer Questions

**Phase 0 — Foundations** · Estimated time: 4-5 days · Prerequisites: [05 · NumPy](05-numpy.md), [06 · Pandas](06-pandas.md)

> A table of 891 rows reveals almost nothing at a glance; the right chart tells the story in two seconds. This lesson teaches the small set of chart types that cover 95% of real data work and, more importantly, the discipline of asking a question first and then choosing the chart that answers it. The three projects use the Titanic data, a year of weather in one city and a freely chosen dataset. From this lesson on, **every project in this curriculum should include at least one plot**: when ML models misbehave later, plotting the data and the training curves is how to debug them. Plots are not decoration; they are the way to see what is going on.

## What this lesson builds

- **Project 1 — Chart sampler:** a notebook with 6 charts of the Titanic data, each headed by a question and followed by a one-sentence answer.
- **Project 2 — Weather story:** a 4-chart notebook that tells the story of one year of weather in a chosen city, from real data downloaded in the project.
- **Project 3 — EDA mini-report:** a one-notebook exploratory report on a freely chosen dataset: 5 questions, 5 charts, 5 answers. This is the exact ritual to repeat at the start of every ML project.

## Concepts covered

- The matplotlib **figure/axes** model: a *figure* is the whole canvas, and an *axes* is one plot drawn on it
- The four workhorse charts: **line** (change over time), **bar** (compare categories), **scatter** (relationship between two numbers), **histogram** (shape of one number's distribution)
- Labels, titles and legends: the difference between a chart and a readable chart
- **seaborn** one-liners (`histplot`, `boxplot`, `heatmap`, `pairplot`): statistical charts in one call
- When to use which chart type (a small decision guide, learned by using it)
- Saving figures to PNG files with `savefig`
- The "one question per chart" discipline

## Before starting

Check that these skills from earlier lessons are in place: loading a CSV into a pandas DataFrame, selecting columns, filtering rows and using `groupby`. If `groupby` feels shaky, skim the lesson 06 work first.

Activate the venv (created in [lesson 01](01-environment-setup.md)) and install the plotting libraries:

```bash
cd ~/ml/ml-learn
source .venv/bin/activate
pip install matplotlib seaborn
```

Create the work folder for this lesson:

```bash
mkdir -p work/07-data-visualization
```

**Where will the plots appear?** Work in Jupyter notebooks inside VS Code (as in lesson 06): plots render right under the cell. If a plot ever comes from a plain `.py` script instead, what `plt.show()` does depends on the system:

- **macOS:** it opens the chart in a window, and the script waits until the window is closed.
- **Windows (WSL2) and Linux:** it opens no window, because the Python 3.13 from lesson 01 comes without tkinter, the toolkit matplotlib uses for windows on Linux. Save to a file with `plt.savefig("myplot.png")` instead and open the PNG in VS Code. To get windows anyway, install tkinter with `sudo apt install -y python3.13-tk` (on Fedora, `sudo dnf install -y python3.13-tkinter`) and run the script again. On Windows, the window then opens on the Windows desktop.

**Data:** reuse the Titanic CSV saved at the end of lesson 06's Project 1 (copy it in: `cp work/06-pandas/titanic.csv work/07-data-visualization/`). If that file is gone, seaborn bundles the same data: `sns.load_dataset("titanic")` returns it as a DataFrame (this needs internet once). Project 2's weather download is explained inside that project.

## Project 1 — Chart sampler

**Goal:** Produce 6 charts of the Titanic data, each one answering a specific stated question, and learn the figure/axes model, the four workhorse chart types and labeling along the way.

**Milestones**

- [ ] Create `work/07-data-visualization/01_chart_sampler.ipynb`. In the first cell, import everything and load the data:

  ```python
  import pandas as pd
  import matplotlib.pyplot as plt
  import seaborn as sns

  df = pd.read_csv("titanic.csv")   # or: df = sns.load_dataset("titanic")
  df.head()
  ```

- [ ] Make the first plot the explicit way, so that the figure/axes model comes before any shortcuts:

  ```python
  fig, ax = plt.subplots(figsize=(8, 5))   # fig = the canvas, ax = one plot on it
  ax.hist(df["age"].dropna(), bins=30)
  ax.set_xlabel("Age (years)")
  ax.set_ylabel("Number of passengers")
  ax.set_title("Q1: How old were the Titanic passengers?")
  ```

  A *histogram* chops a numeric column into equal-width buckets ("bins") and draws a bar per bucket showing how many values fall in it. It shows the *shape* of one variable. Checkpoint: the tallest bars sit in the 20-30 age range, with a smaller bump for young children.
- [ ] Under the chart, write the answer in a markdown cell, as one sentence ("Mostly young adults in their 20s-30s, plus a noticeable group of small children."). Do this for every chart in this lesson: a chart without its answer is homework left unfinished.
- [ ] **Chart 2 (bar):** "Q2: Did first-class passengers survive more often?" Compute `df.groupby("pclass")["survived"].mean()` and plot it as a bar chart (the Series method `.plot.bar(ax=ax)` works, and so does `ax.bar(...)`). A *bar chart* compares a number across a few categories. Label the y-axis "survival rate". Checkpoint: the bars step downward (roughly 0.63 for 1st class, 0.47 for 2nd and 0.24 for 3rd).
- [ ] **Chart 3 (bar):** "Q3: Did 'women and children first' show up in the data?" Draw a bar chart of survival rate by sex. Checkpoint: about 0.74 for women vs about 0.19 for men, a huge gap.
- [ ] **Chart 4 (scatter):** "Q4: Is there a pattern connecting fare, age and survival?" A *scatter plot* puts one dot per row, revealing the relationship between two numeric columns. Draw fare (y) vs age (x), colored by survival:

  ```python
  fig, ax = plt.subplots(figsize=(8, 5))
  scatter = ax.scatter(df["age"], df["fare"], c=df["survived"], alpha=0.6)
  ax.legend(*scatter.legend_elements(), title="Survived")
  ```

  `alpha=0.6` makes the dots translucent so that overlaps stay visible. The *legend* is the little box that explains what each color means; always add one when color carries meaning. Checkpoint: most dots crowd below fare 100, and the expensive tickets are mostly survivor-colored.
- [ ] **Chart 5 (boxplot):** "Q5: How does age differ across classes?" One seaborn line: `sns.boxplot(data=df, x="pclass", y="age")`. A *boxplot* summarizes a distribution as a box (middle 50% of values, line at the median) with whiskers for the rest. It is ideal for comparing distributions across categories. Checkpoint: the 1st-class median age is clearly the highest (money takes time to earn).
- [ ] **Chart 6 (heatmap):** "Q6: Which numeric columns move together?" A *heatmap* paints a grid of numbers as colors so that patterns stand out. Feed it a *correlation matrix*: a table where each cell holds the correlation between two columns. A correlation is a number from -1 to +1 that measures how strongly two columns move together (+1: they rise together; -1: one rises as the other falls; 0: unrelated). `.corr()` computes the matrix:

  ```python
  corr = df[["survived", "pclass", "age", "sibsp", "parch", "fare"]].corr()
  sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
  ```

  Checkpoint: `pclass` vs `fare` is strongly negative, around -0.55 (a lower class number goes with a pricier ticket), and `pclass` vs `survived` is around -0.34.
- [ ] Pick a favorite chart and save it to a file: `fig.savefig("survival_by_class.png", dpi=150, bbox_inches="tight")`. Checkpoint: the PNG appears in the work folder and opens in VS Code.
- [ ] Add a final markdown cell and write a 4-line "when to use which chart" cheat sheet in it: trend over time → line; compare categories → bar; two numbers related? → scatter; shape of one number → histogram (boxplot to compare shapes, heatmap for a grid of numbers).

<details><summary>Hints</summary>

- `ax.hist` chokes on missing values in older matplotlib versions, so call `.dropna()` first. Lesson 06 showed that `age` has missing entries; plotting is where that bites.
- Pandas objects plot themselves: `series.plot.bar(ax=ax)` draws onto an axes made beforehand, so its labels and title can still be set as usual.
- If two plots merge into one chart, both were drawn onto the same axes. Create a fresh `fig, ax = plt.subplots()` per chart.

</details>

**Definition of done:** a notebook with 6 labeled charts (title, axis labels, legend where color matters), each preceded by a question and followed by a one-sentence answer.

## Project 2 — Weather story

**Goal:** Download a year of real daily weather for a city of personal interest and tell its story in 4 charts. These are the first plots from data fetched straight from the source, with all its real-world mess.

**Milestones**

- [ ] Download the data. Open-Meteo's archive API serves historical weather as CSV with no signup. This command downloads 2024 for Berlin; swap in the chosen city's latitude/longitude (find them at https://open-meteo.com, or search "\<your city> latitude longitude"):

  ```bash
  cd ~/ml/ml-learn/work/07-data-visualization
  curl "https://archive-api.open-meteo.com/v1/archive?latitude=52.52&longitude=13.41&start_date=2024-01-01&end_date=2024-12-31&daily=temperature_2m_max,temperature_2m_min,precipitation_sum&timezone=auto&format=csv" -o weather.csv
  ```

  Checkpoint: `wc -l weather.csv` shows roughly 370 lines.
- [ ] **Look at the raw file before parsing it.** Make it a habit for life. Open `weather.csv` in VS Code. Notice that the real column header row is not line 1: there are metadata lines above it. Load the file into a new notebook, `02_weather_story.ipynb`, with `pd.read_csv("weather.csv", skiprows=N)` (count N in the raw file), then convert the time column: `df["time"] = pd.to_datetime(df["time"])`. The other column names carry their units, like `temperature_2m_max (°C)`; rename them to the short names that the code below uses: `df.columns = ["time", "temperature_2m_max", "temperature_2m_min", "precipitation_sum"]`. Checkpoint: `df.dtypes` shows `datetime64` for time and `float64` for the temperature columns, and `len(df)` is 365 or 366.
- [ ] **Chart 1 (line):** daily max temperature across the whole year. A *line chart* connects points in time order and is the default for anything measured over time: `ax.plot(df["time"], df["temperature_2m_max"])`. Question: "What did the year feel like?" Checkpoint: a broad seasonal wave (a hill or valley shape, depending on the hemisphere), not random noise.
- [ ] **Chart 2 (line, two series):** max and min temperature on the same axes. Call `ax.plot` twice, with `label="max"` and `label="min"`, then call `ax.legend()`. Question: "How big is the daily swing, and does it change with season?"
- [ ] **Chart 3 (bar):** average max temperature per month. Make a month column (`df["time"].dt.month`), then use `groupby("month")` and `.mean()`. This is the split-apply-combine move from lesson 06, now feeding a chart. Checkpoint: 12 bars with a clear seasonal shape (for most non-equatorial cities, warmest and coldest months differ by 10 °C or more).
- [ ] **Chart 4 (free choice):** find and mark the hottest and coldest days. Get them with `df.loc[df["temperature_2m_max"].idxmax()]` (and `idxmin` on the min column), then design a chart that shows them. For example, draw the year line, mark the two days on top of it with `ax.scatter`, and write the dates on the chart with `ax.annotate`. Checkpoint: two labeled points sitting exactly on the line's peak and trough.
- [ ] Top the notebook with a markdown title cell and finish with a 3-sentence "weather story" summarizing what the four charts revealed. Save chart 4 as `weather_story.png`.

<details><summary>Hints</summary>

- If `read_csv` gives one giant column or weird column names, the `skiprows` value is off. Recount the metadata lines in the raw file, including any blank line.
- X-axis date labels overlapping? `fig.autofmt_xdate()` rotates them.
- Two lines, one color, no legend? Either `label=` is missing in `plot`, or `ax.legend()` was never called.
- `idxmax()` returns the row *index* of the maximum. Feed it to `df.loc[...]` to get the whole row, including the date.

</details>

**Definition of done:** a notebook that downloads nothing at run time (it reads the local CSV), renders 4 labeled charts, and ends with a short written story that a friend could read without seeing the code.

## Project 3 — EDA mini-report

**Goal:** Run a first full *EDA* (exploratory data analysis, the "interview the dataset before modeling it" step) on a freely chosen dataset: 5 questions, 5 charts, 5 one-sentence answers. This is exactly the first hour of every ML project.

**Milestones**

- [ ] Pick a genuinely interesting dataset; curiosity keeps the work going. Easy no-signup options: any of seaborn's built-in sets (`sns.get_dataset_names()` lists them; `penguins`, `tips` and `diamonds` are good), another Open-Meteo download (compare two cities?), or any CSV from the wild. Aim for at least 300 rows and a mix of numeric and categorical columns.
- [ ] Create `03_eda_report.ipynb`. First interrogate the data without plotting, using the lesson 06 toolkit: `df.shape`, `df.head()`, `df.dtypes`, `df.describe()`, `df.isna().sum()`. In a markdown cell, note the size, the column meanings and any missing data. Checkpoint: say out loud what one row represents.
- [ ] **Before any chart, write the 5 questions** in one markdown cell. Use real questions with unknown answers ("Do heavier penguins have longer flippers?" "Do smokers tip differently?"), not chart orders ("plot X"). This is the one-question-per-chart discipline, and it stops the aimless plotting that eats afternoons.
- [ ] Get a fast overview with one seaborn line: `sns.pairplot(df, hue="<a category column>")`. It draws a grid of scatter plots for every pair of numeric columns, colored by a category (with histograms on the diagonal). Checkpoint: at least one panel gives a "huh" moment; refine a question if so.
- [ ] Answer each question with exactly one chart, choosing the type deliberately from the Project 1 cheat sheet, plus seaborn's `histplot` and `boxplot` where they fit. Give every one the same structure: markdown question → code cell → markdown one-sentence answer. Vary the types: a report with 5 histograms means the questions were too similar.
- [ ] Polish pass: every chart has a title, axis labels, and a legend if color carries meaning. Restart the kernel and "Run All": everything runs clean top to bottom. Checkpoint: a friend could read only the markdown cells and charts and learn 5 true things.
- [ ] Commit all three notebooks to git (lesson 01 workflow): `git add`, `git commit -m "Lesson 07: data visualization projects"`.

<details><summary>Hints</summary>

- Struggling to find 5 questions? Use the templates: How is \<numeric\> distributed? Does \<numeric\> differ across \<category\>? Are \<numeric\> and \<numeric\> related? How does \<numeric\> change over time? Which \<category\> is most common?
- `pairplot` on many columns is slow and unreadable. Pass 3-4 numeric columns via `vars=[...]`.
- A "boring" answer ("no relationship") is a valid, useful answer. Say it in the sentence and move on. In ML, knowing what does *not* predict is half the job.

</details>

**Definition of done:** one notebook, 5 questions, 5 deliberately-chosen charts, 5 one-sentence answers, runs clean top to bottom.

## Stretch goals

- Rebuild the 6 Titanic charts as a single figure with `fig, axes = plt.subplots(2, 3, figsize=(18, 10))`: one dashboard image, saved as PNG.
- Restyle a whole notebook with one line (`sns.set_theme()` or `plt.style.use("ggplot")`) and note what changed.
- In the weather story, add a 7-day rolling-average line on top of the noisy daily line: `df["temperature_2m_max"].rolling(7).mean()`.
- Take the Kaggle data visualization micro-course (link below) and redo one of its exercises on a freely chosen dataset instead of the course's data.

## Getting unstuck

- **Blank or missing plot?** In notebooks, the plot renders when the cell ends, so make the plotting calls the last thing in the cell. From a `.py` script, `plt.show()` opens a window on macOS. On Windows (WSL2) and Linux it opens none unless tkinter is installed (see Before starting), and it may warn `FigureCanvasAgg is non-interactive, and thus cannot be shown`. There, do not fight `plt.show()`: `savefig` to PNG and open the file in VS Code.
- **Two charts on top of each other?** The code reused an axes. Use one `fig, ax = plt.subplots()` per chart.
- **A mysterious `Text(0.5, ...)` line under the plot** is only the return value of the last call, not an error. Ignore it, or end the cell with `plt.show()`.
- **Seaborn errors about a column** usually mean a name mismatch. Print `df.columns` and check the spelling and case (the Titanic columns in this lesson are all lowercase: `age`, not `Age`; a Titanic CSV from Kaggle capitalizes them).
- Standing advice: read the error message bottom-up (the last line names the problem); print what is being plotted (`df["age"].head()`, `.shape`, `.dtype`) before blaming the chart; ask an AI assistant for a **hint**, not a solution; and type all code by hand, with no pasting.

## Resources

- [Matplotlib quick start](https://matplotlib.org/stable/users/explain/quick_start.html) — the figure/axes model from the source; read after Project 1 and it will all click.
- [Seaborn tutorial](https://seaborn.pydata.org/tutorial.html) — gallery-style tour of every seaborn plot; skim to know what exists, and return when one is needed.
- [Kaggle: Data Visualization](https://www.kaggle.com/learn/data-visualization) — free short course with in-browser exercises; good structured practice after Project 3 (account needed for the exercises).

## Skills unlocked

- [ ] I can explain the difference between a figure and an axes, and create both with `plt.subplots()`
- [ ] I can pick the right chart for a question (line vs bar vs scatter vs histogram) and say why
- [ ] I can make any chart readable: title, axis labels, legend
- [ ] I can use seaborn's `histplot`, `boxplot`, `heatmap` and `pairplot` on a DataFrame
- [ ] I can download a real CSV from an API with `curl` and handle its quirks (metadata rows, dates as strings)
- [ ] I can chain `groupby` into a chart to answer a "compare across categories" question
- [ ] I can save a publication-ready PNG with `savefig`
- [ ] I can run a 5-question EDA on a dataset I have never seen

## Next up

With the full Python data toolkit in place, the next lesson builds the math engine underneath ML by coding it from scratch: [08 · Linear Algebra by Writing Your Own](../phase-1-math/08-linear-algebra-by-code.md).
