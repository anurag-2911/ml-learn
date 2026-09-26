# 20 · Capstone: End-to-End ML Project (Kaggle)

**Phase 2 — Classical ML** · Estimated time: 2 weeks · Prerequisites: [14 · Model Evaluation](14-model-evaluation.md), [16 · Ensembles](16-ensembles-random-forest-boosting.md), [19 · Feature Engineering and Pipelines](19-feature-engineering-pipelines.md)

> This is the lesson where you stop doing exercises and start doing machine learning. You will enter a real Kaggle competition — a public contest where thousands of people build models on the same dataset and a leaderboard ranks everyone's predictions — and work it like a professional: frame the problem, explore the data, submit a deliberately dumb baseline, then climb past it one idea at a time, keeping score the whole way. At the end you will have a public, well-documented project you can show anyone. **MILESTONE: when you finish this lesson, you can genuinely call yourself a junior ML practitioner.**

## What you will build

- A complete Kaggle competition entry for **House Prices: Advanced Regression Techniques** — an EDA notebook, at least five real leaderboard submissions (from dumb baseline to tuned ensemble), an error-analysis notebook, and a professional README with a scoreboard of every attempt.

## Concepts you will learn by doing

- **The full ML workflow** — frame → explore → baseline → features → models → tune → error analysis → report, the loop every real project follows.
- **Baseline first, always** — the simplest possible prediction, submitted before anything clever, so every later idea has something honest to beat.
- **Validation scoreboard** — a local score (cross-validation) that tracks the leaderboard, so you can test ideas in seconds instead of burning submissions.
- **Error analysis** — reading the rows your model gets most wrong, which tells you what to fix next better than any tutorial can.
- **Hyperparameter search** — systematically trying model settings with GridSearchCV (exhaustive) or Optuna (a library that searches smartly).
- **Ensembling by blending** — averaging predictions from different models to get a score better than any one of them.
- **The project README** — the write-up that turns a folder of notebooks into evidence you can do this job.

## Before you start

You need everything from lessons [12](12-linear-regression.md), [14](14-model-evaluation.md), [15](15-decision-trees.md), [16](16-ensembles-random-forest-boosting.md) and [19](19-feature-engineering-pipelines.md): pipelines, cross-validation, one-hot encoding, gradient boosting. If any of those feel shaky, skim your own code from those lessons first — it is the best reference you own.

**Kaggle needs a free account.** That is fine here — it is the one signup this curriculum asks for, and it is worth it: Kaggle is where much of the ML world practices in public. Sign up at [kaggle.com](https://www.kaggle.com), then on the [House Prices competition page](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) click **Join Competition** and accept the rules (downloads fail with a 403 error until you do).

Set up the Kaggle command-line tool and your work folder:

```bash
cd ~/ml/ml-learn
source .venv/bin/activate
pip install kaggle optuna xgboost lightgbm
mkdir -p work/20-end-to-end-ml-project/data
```

On Kaggle: click your avatar → **Settings** → **API** → **Create New Token**. That downloads `kaggle.json`. Move it into place:

```bash
mkdir -p ~/.kaggle
mv /mnt/c/Users/<YourWindowsUser>/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

Then download and unzip the data:

```bash
cd work/20-end-to-end-ml-project
kaggle competitions download -c house-prices-advanced-regression-techniques -p data
unzip data/house-prices-advanced-regression-techniques.zip -d data
```

You should see `train.csv` (houses with known sale prices — your training data), `test.csv` (houses whose prices you must predict), `sample_submission.csv` (the exact format Kaggle expects back) and `data_description.txt` (what every column means — you will live in this file).

One thing to know up front: this competition scores you on **RMSE of the log of the price** — root-mean-squared error (the `rmse` one-liner you added to your metrics library in lesson 16: `np.sqrt(np.mean((y_pred - y)**2))`) computed on `log(SalePrice)` instead of the raw price, so a $10k mistake on a cheap house hurts more than on a mansion. Lower is better. Every score below refers to this number.

## Project 1 — Climb the House Prices leaderboard

**Goal.** Predict the sale price of houses in Ames, Iowa from 79 features, working through seven professional-style milestones — each one ends with something submitted, measured, or written down.

**Milestones**

*Milestone 1 — EDA: interrogate the data (days 1–2).* EDA (exploratory data analysis) means looking at the data until you understand it, before modeling anything.

- [ ] Create `eda.ipynb` in your work folder. Load `train.csv` with pandas. Checkpoint: `df.shape` shows **(1460, 81)**.
- [ ] Answer these 8 questions, each as a markdown heading followed by the chart or table that answers it and one written sentence of conclusion:
  1. What does the distribution of `SalePrice` look like — and what does it look like after `np.log`? (Checkpoint: raw is skewed with a long right tail; logged is roughly bell-shaped.)
  2. Which 10 numeric columns correlate most strongly with `SalePrice`? (Checkpoint: `OverallQual` and `GrLivArea` top the list.)
  3. Which columns have missing values, and for which ones does "missing" actually mean "the house doesn't have this" (read `data_description.txt` — e.g. no pool, no garage)?
  4. How does price relate to living area (`GrLivArea`)? Scatter-plot it. (Checkpoint: mostly linear, with a few huge cheap houses bottom-right — remember those outliers.)
  5. How does price vary by `Neighborhood`? (A sorted boxplot answers this in one picture.)
  6. Does overall quality (`OverallQual`) relate to price linearly or faster-than-linearly?
  7. Do newer or recently remodeled houses (`YearBuilt`, `YearRemodAdd`) sell for more?
  8. Pick one question of your own that the data made you curious about, and answer it.

*Milestone 2 — Submit a dumb baseline (day 3).* Yes, actually submit it. This makes every mechanical piece work (file format, upload, score) while nothing clever is at stake, and it plants the flag every idea must beat.

- [ ] Predict the **median** training `SalePrice` for every single test house. Build the file from `sample_submission.csv` so the format is exactly right:

```python
import pandas as pd
train = pd.read_csv("data/train.csv")
sub = pd.read_csv("data/sample_submission.csv")
sub["SalePrice"] = train["SalePrice"].median()
sub.to_csv("submission_baseline.csv", index=False)
```

- [ ] Submit it: `kaggle competitions submit -c house-prices-advanced-regression-techniques -f submission_baseline.csv -m "median baseline"`. Checkpoint: your name is on the leaderboard with a score around **0.40** (i.e. bad — that is the point).
- [ ] Start `SCOREBOARD.md` in your work folder: a table with columns *attempt, local CV score, leaderboard score, what changed*. Add row 1. Every submission from now on gets a row — this discipline is the actual lesson.

*Milestone 3 — Linear model + features (days 4–6).* Now build a local validation scoreboard so you can test ideas without spending submissions (Kaggle allows only a handful per day).

- [ ] Write `evaluate(model)` using 5-fold `cross_val_score` on `log1p(SalePrice)` with RMSE, mirroring the competition metric. Checkpoint: it scores your median baseline around 0.40 locally — local and leaderboard roughly agree, which is what makes the scoreboard trustworthy.
- [ ] Build a scikit-learn `Pipeline` (lesson 19): impute missing values, one-hot encode categoricals, then `Ridge` regression on the log target (`Ridge` = linear regression plus a penalty that shrinks weights to prevent overfitting — the regularization from lesson 19; it copes with the many one-hot columns far better than plain `LinearRegression`). Predict, `np.expm1` back to dollars, submit. Checkpoint: leaderboard around **0.15–0.20** — a massive jump from 0.40.
- [ ] Improve the features using what EDA told you: fill "missing means none" columns with `"None"`/0, add a `TotalSF` (total square footage) feature, log-transform skewed numerics, try dropping the outlier houses from question 4. Test each idea against your local CV, keep what helps, log every attempt in `SCOREBOARD.md`. Checkpoint: local CV below **0.14**.

*Milestone 4 — Gradient boosting (days 7–8).*

- [ ] Swap the final estimator for gradient boosting (lesson 16): try `HistGradientBoostingRegressor`, XGBoost or LightGBM with default settings. Checkpoint: it beats your best linear model's local CV without any tuning.
- [ ] Submit your best boosted model. Checkpoint: leaderboard around **0.13–0.14**, and your `SCOREBOARD.md` now tells a story of steady descent.

*Milestone 5 — Tune and ensemble (days 9–10).*

- [ ] Tune the boosted model: `GridSearchCV` over a small grid (learning rate, tree depth, number of trees), or let Optuna search for you — its homepage has a 10-line quickstart; search for "Optuna quickstart". Checkpoint: tuning buys a small but real local CV improvement (expect ~0.002–0.005, not miracles).
- [ ] Blend: average the predictions of your tuned boosted model and your best linear model (in log space). Try weights like 0.5/0.5 and 0.7/0.3, chosen by local CV. Checkpoint: the blend beats both parents — this is why ensembling wins competitions.
- [ ] Submit the final blend. Checkpoint: leaderboard around **0.125 or lower**, comfortably in the upper half. (Ignore the impossible ~0.00 scores at the very top — those are people who looked up the true answers, which are public for this old competition. You are competing honestly against yourself and the baseline.)

*Milestone 6 — Error analysis: read your mistakes (day 11).*

- [ ] Using cross-validated predictions on the *training* set (`cross_val_predict` — test-set answers are hidden, so mistakes can only be studied where you know the truth), compute each house's error and pull the **20 worst predictions** into a dataframe alongside their features.
- [ ] Study them and write what you find in the notebook: Are they mostly cheap or expensive houses? Odd neighborhoods? Unusual sale conditions (check `SaleCondition`)? Checkpoint: at least three written observations, and one concrete "what I would try next" per pattern.
- [ ] Try the single most promising fix, measure it on local CV, and record the result — even if it does not help. Negative results belong on the scoreboard too.

*Milestone 7 — The README that gets you hired (days 12–14).*

- [ ] Write `README.md` in your work folder with: (1) the problem in two sentences, (2) three EDA findings with one chart, (3) your approach as the story of the workflow you just lived, (4) the full scoreboard table from `SCOREBOARD.md`, (5) what the error analysis revealed, (6) "what I'd try next" — a ranked list of 3+ ideas you didn't get to. Checkpoint: a stranger could understand what you did and why in five minutes without opening a notebook.
- [ ] Commit everything, **make your repo public on GitHub, and share the README** — with a friend, or anywhere fellow learners gather. A public, honest write-up of a real project is worth more than any certificate. This is your milestone moment: you have now executed the complete ML workflow, end to end, on a problem nobody pre-chewed for you.

<details><summary>Hints</summary>

- **Train and test must be preprocessed identically.** Fit your pipeline on training data only and call `.predict()` on test — never `fit` anything on test data (that is leakage, lesson 19). If one-hot encoding breaks on unseen categories, `OneHotEncoder(handle_unknown="ignore")` is your friend.
- **Work in log space everywhere.** Fit on `np.log1p(y)`, blend predictions in log space, and only `np.expm1` at the very end when writing the submission file.
- **When local CV improves but the leaderboard doesn't** (or vice versa), suspect the outliers you found in EDA question 4 — try your pipeline with and without dropping them and watch both numbers.
- **For the worst-20 analysis**, `df.assign(err=np.abs(pred - actual)).nlargest(20, "err")` gets you the table; sorting by *relative* error (`err / actual`) tells a different, equally interesting story.

</details>

**Definition of done.** At least five scored submissions on the Kaggle leaderboard ending at ~0.125 or better, a scoreboard logging every attempt, a written error analysis of your 20 worst predictions, and a README you would be proud to show an interviewer.

## Stretch goals

- **Stacking**: instead of hand-picked blend weights, train a small linear model whose inputs are the other models' cross-validated predictions — a proper stacked ensemble (try `StackingRegressor`).
- **Second competition, from memory**: enter **Spaceship Titanic** on Kaggle (a classification problem) and repeat all seven milestones in one week, using this lesson only as a checklist. Doing the workflow twice is what makes it yours.
- **Feature importance deep-dive**: use permutation importance (`sklearn.inspection.permutation_importance` — it shuffles one feature at a time and measures how much the score drops) on your final model and check whether the top features match your EDA intuitions; investigate any surprise.
- **Read the masters**: after finishing, read two or three top-voted solution write-ups in the competition's discussion tab and note what they did that you didn't.

## If you get stuck

- `403 Forbidden` on download almost always means you haven't clicked **Join Competition** and accepted the rules on the website.
- Kaggle rejects your submission file? Open it and `sample_submission.csv` side by side: same column names (`Id,SalePrice`), same number of rows (1459), no index column (`index=False`).
- Score got dramatically *worse* after a change? You probably predicted log-prices but submitted them as dollars — check that your submission's values are in the hundreds of thousands, not around 12.
- A pipeline crash mid-`cross_val_score` usually means a column with NaN slipped past your imputer — print `X.isna().sum()[lambda s: s > 0]`.
- Standing advice: read the error message bottom-up (the last line names the real problem), print shapes and a few values at every step, ask an AI assistant for a **hint** rather than a solution, and type all code yourself — copy-pasting teaches your clipboard, not you.

## Resources

- [House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) — your competition: data, rules, leaderboard, and the discussion tab full of ideas.
- [Kaggle: Intermediate Machine Learning](https://www.kaggle.com/learn/intermediate-machine-learning) — short free course covering missing values, pipelines, leakage and XGBoost; a perfect companion to milestones 3–5.
- [scikit-learn.org](https://scikit-learn.org) — API reference for `Pipeline`, `cross_val_score`, `GridSearchCV`, `StackingRegressor`.

## Skills unlocked

- [ ] I can take a raw dataset and run the full workflow — frame, EDA, baseline, features, models, tune, analyze errors, report — without a tutorial holding my hand.
- [ ] I always establish a dumb baseline before building anything clever, and I can say why.
- [ ] I keep a validation scoreboard and can explain why local CV must mirror the real metric.
- [ ] I can engineer features from EDA findings and measure honestly whether each one helped.
- [ ] I can tune hyperparameters with GridSearchCV or Optuna and blend models into an ensemble that beats its parts.
- [ ] I can read my model's worst mistakes and turn them into a ranked list of next experiments.
- [ ] I have a public project README that shows my work like a professional.

## Next up

You have mastered classical ML — now for the ideas behind the current AI revolution, starting with the simplest neural network there is: [21 · Neural Networks: The Intuition (Perceptron and XOR)](../phase-3-deep-learning/21-neural-networks-intuition.md).
