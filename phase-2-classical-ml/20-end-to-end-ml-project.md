# 20 · Capstone: End-to-End ML Project (Kaggle)

**Phase 2 — Classical Machine Learning** · Estimated time: 2 weeks · Prerequisites: [14 · Model Evaluation](14-model-evaluation.md), [16 · Ensembles](16-ensembles-random-forest-boosting.md), [19 · Feature Engineering and Pipelines](19-feature-engineering-pipelines.md)

> This capstone lesson moves from doing exercises to doing machine learning. The project is an entry in a real Kaggle competition: a public contest where thousands of people build models on the same dataset, and a leaderboard ranks everyone's predictions. The work follows the steps a professional takes: frame the problem, explore the data, submit a deliberately dumb baseline, then climb past it one idea at a time, keeping score the whole way. The result is a public, well-documented project that can be shown to anyone. **MILESTONE: anyone who finishes this lesson can genuinely be called a junior ML practitioner.**

## What this lesson builds

- A complete Kaggle competition entry for **House Prices: Advanced Regression Techniques**, made up of an EDA notebook, at least five real leaderboard submissions (from dumb baseline to tuned ensemble), an error-analysis notebook, and a professional README with a scoreboard of every attempt.

## Concepts covered

- **The full ML workflow**: frame → explore → baseline → features → models → tune → error analysis → report, the loop every real project follows.
- **Baseline first, always**: the simplest possible prediction, submitted before anything clever, so every later idea has something honest to beat.
- **Validation scoreboard**: a local score (cross-validation) that tracks the leaderboard, so ideas can be tested in seconds instead of using up submissions.
- **Error analysis**: reading the rows the model gets most wrong, which shows what to fix next better than any tutorial can.
- **Hyperparameter search**: systematically trying model settings with GridSearchCV (exhaustive) or Optuna (a library that searches smartly).
- **Ensembling by blending**: averaging predictions from different models to get a score better than any one of them.
- **The project README**: the write-up that turns a folder of notebooks into evidence that its author can do this job.

## Before starting

This lesson needs everything from lessons [12](12-linear-regression.md), [14](14-model-evaluation.md), [15](15-decision-trees.md), [16](16-ensembles-random-forest-boosting.md) and [19](19-feature-engineering-pipelines.md): pipelines, cross-validation, one-hot encoding, gradient boosting. If any of those feel shaky, first skim the code in those lessons' work folders. It is the best reference at hand.

**Kaggle needs a free account.** It is the one signup this lesson asks for, and it is worth it: Kaggle is where much of the ML world practices in public. Sign up at [kaggle.com](https://www.kaggle.com), then on the [House Prices competition page](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) click **Join Competition** and accept the rules (downloads fail with a 403 error until the rules are accepted).

Set up the Kaggle command-line tool and the work folder:

```bash
cd ~/ml/ml-learn
source .venv/bin/activate
pip install kaggle optuna xgboost lightgbm
mkdir -p work/20-end-to-end-ml-project/data
```

XGBoost and LightGBM need OpenMP, a helper library that lets them use all the computer's CPU cores. **macOS:** their Mac versions look for it where Homebrew installs it, so on an Apple Silicon Mac with Homebrew, run `brew install libomp`. On a Mac without Homebrew (Intel Macs, and macOS 14 or older), `import xgboost` and `import lightgbm` fail. Skip both and use scikit-learn's `HistGradientBoostingRegressor` in Milestone 4, which brings its own copy of OpenMP. **Windows (WSL2) and Linux:** LightGBM needs Ubuntu's OpenMP package, which WSL's Ubuntu does not include, so run `sudo apt install -y libgomp1`. Then `python -c "import xgboost, lightgbm"` should finish without an error (on a Mac without Homebrew, skip this check).

On Kaggle, click the avatar → **Settings** → **API** → **Generate New Token** (or go straight to [kaggle.com/settings/api](https://www.kaggle.com/settings/api)), and copy the token that Kaggle shows. It works like a password for the Kaggle account, so never put it in code or in the repository. The Kaggle tool reads it from the file `~/.kaggle/access_token`. Create the folder and open that file in nano:

```bash
mkdir -p ~/.kaggle
nano ~/.kaggle/access_token
```

Paste the token (Cmd+V on a Mac, Ctrl+Shift+V in Windows and Linux terminals), save with Ctrl+O and Enter, and exit with Ctrl+X. Lock the file so that only its owner can read it:

```bash
chmod 600 ~/.kaggle/access_token
```

Then download and unzip the data:

```bash
cd work/20-end-to-end-ml-project
kaggle competitions download -c house-prices-advanced-regression-techniques -p data
unzip data/house-prices-advanced-regression-techniques.zip -d data
```

The `data` folder should now contain `train.csv` (houses with known sale prices: the training data), `test.csv` (houses whose prices must be predicted), `sample_submission.csv` (the exact format Kaggle expects back) and `data_description.txt` (what every column means; this is the file to check most often).

One thing to know up front: this competition scores submissions on **RMSE of the log of the price**. That is root-mean-squared error (the `rmse` one-liner added to the metrics library in lesson 16: `np.sqrt(np.mean((y_pred - y)**2))`) computed on `log(SalePrice)` instead of the raw price, so a $10k mistake on a cheap house hurts more than on a mansion. Lower is better. Every score below refers to this number.

## Project 1 — Climb the House Prices leaderboard

**Goal.** Predict the sale price of houses in Ames, Iowa from 79 features, working through seven professional-style milestones. Each one ends with something submitted, measured, or written down.

**Milestones**

*Milestone 1 — EDA: interrogate the data (days 1–2).* EDA (exploratory data analysis) means looking at the data until it is well understood, before modeling anything.

- [ ] Create `eda.ipynb` in the work folder. Load `train.csv` with pandas. Checkpoint: `df.shape` shows **(1460, 81)**.
- [ ] Answer these 8 questions, each as a markdown heading followed by the chart or table that answers it and one written sentence of conclusion:
  1. What does the distribution of `SalePrice` look like, and what does it look like after `np.log`? (Checkpoint: raw is skewed with a long right tail; logged is roughly bell-shaped.)
  2. Which 10 numeric columns correlate most strongly with `SalePrice`? (Checkpoint: `OverallQual` and `GrLivArea` top the list.)
  3. Which columns have missing values, and for which ones does "missing" actually mean "the house doesn't have this" (read `data_description.txt`; for example, no pool or no garage)?
  4. How does price relate to living area (`GrLivArea`)? Scatter-plot it. (Checkpoint: mostly linear, with a few huge cheap houses bottom-right. Remember those outliers.)
  5. How does price vary by `Neighborhood`? (A sorted boxplot answers this in one picture.)
  6. Does overall quality (`OverallQual`) relate to price linearly or faster-than-linearly?
  7. Do newer or recently remodeled houses (`YearBuilt`, `YearRemodAdd`) sell for more?
  8. Pick one more question that the data raises, and answer it.

*Milestone 2 — Submit a dumb baseline (day 3).* Submitting even this baseline makes every mechanical piece work (file format, upload, score) while nothing clever is at stake. It also sets the mark that every idea must beat.

- [ ] Predict the **median** training `SalePrice` for every single test house. Build the file from `sample_submission.csv` so the format is exactly right:

```python
import pandas as pd
train = pd.read_csv("data/train.csv")
sub = pd.read_csv("data/sample_submission.csv")
sub["SalePrice"] = train["SalePrice"].median()
sub.to_csv("submission_baseline.csv", index=False)
```

- [ ] Submit it: `kaggle competitions submit -c house-prices-advanced-regression-techniques -f submission_baseline.csv -m "median baseline"`. Checkpoint: the account's name is on the leaderboard with a score around **0.40** (i.e. bad; that is the point).
- [ ] Start `SCOREBOARD.md` in the work folder: a table with columns *attempt, local CV score, leaderboard score, what changed*. Add row 1. Every submission from now on gets a row. This discipline is the actual lesson.

*Milestone 3 — Linear model + features (days 4–6).* Now build a local validation scoreboard, so that ideas can be tested without spending submissions (Kaggle allows only a handful per day).

- [ ] Write `evaluate(model)` using 5-fold `cross_val_score` on `log1p(SalePrice)` with RMSE, mirroring the competition metric. Pass `scoring="neg_root_mean_squared_error"` (scikit-learn negates error scores so that higher is always better, so flip the sign of the mean), and wrap the median baseline as a model with `DummyRegressor(strategy="median")` from `sklearn.dummy`. Checkpoint: it scores the median baseline around 0.40 locally. Local and leaderboard scores roughly agree, which is what makes the scoreboard trustworthy.
- [ ] Build a scikit-learn `Pipeline` (lesson 19): impute missing values, one-hot encode categoricals, then `Ridge` regression on the log target. (`Ridge` is linear regression plus the regularization from lesson 19, a penalty that shrinks weights to prevent overfitting. It copes with the many one-hot columns far better than plain `LinearRegression`.) Predict, `np.expm1` back to dollars, submit. Checkpoint: leaderboard around **0.15–0.20**, a big jump from 0.40.
- [ ] Improve the features using what the EDA showed: fill "missing means none" columns with `"None"`/0, add a `TotalSF` (total square footage) feature, log-transform skewed numerics, and try dropping the outlier houses from question 4. Test each idea against the local CV, keep what helps, and log every attempt in `SCOREBOARD.md`. Checkpoint: local CV below **0.14**.

*Milestone 4 — Gradient boosting (days 7–8).*

- [ ] Swap the final estimator for gradient boosting (lesson 16): try `HistGradientBoostingRegressor`, XGBoost or LightGBM with default settings. `HistGradientBoostingRegressor` needs dense input, so for it create the encoder as `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`. Checkpoint: local CV lands around 0.125–0.14, usually a little *worse* than the feature-engineered Ridge. That is normal on a dataset this small: a linear model fed good features is hard to beat, and the boosted model earns its place in Milestone 5's blend.
- [ ] Submit the best boosted model and log it in `SCOREBOARD.md` even if it does not beat Ridge: a step that goes up is part of the story. Checkpoint: leaderboard around **0.13–0.14**.

*Milestone 5 — Tune and ensemble (days 9–10).*

- [ ] Tune the boosted model: `GridSearchCV` over a small grid (learning rate, tree depth, number of trees), or let Optuna do the search. Its homepage has a 10-line quickstart; search for "Optuna quickstart". Checkpoint: tuning buys a small but real local CV improvement: about 0.005 for `HistGradientBoostingRegressor` or LightGBM, and about 0.01–0.015 for XGBoost, whose defaults (trees 6 levels deep, learning rate 0.3) are too aggressive for a dataset this small.
- [ ] Blend: average the predictions of the tuned boosted model and the best linear model (in log space). Try weights like 0.5/0.5, 0.7/0.3 and 0.3/0.7, and keep whichever scores best on local CV (usually the one that gives more weight to the better parent). Checkpoint: the blend beats both parents. This is why ensembling wins competitions.
- [ ] Submit the final blend. Checkpoint: leaderboard around **0.125 or lower**, comfortably in the upper half. (Ignore the impossible ~0.00 scores at the very top. They come from people who looked up the true answers, which are public for this old competition. The honest comparison is with earlier attempts and with the baseline.)

*Milestone 6 — Error analysis: read the model's mistakes (day 11).*

- [ ] Using cross-validated predictions on the *training* set (`cross_val_predict`; test-set answers are hidden, so mistakes can only be studied where the true prices are known), compute each house's error and pull the **20 worst predictions** into a dataframe alongside their features.
- [ ] Study them and write the findings in the notebook: Are they mostly cheap or expensive houses? Odd neighborhoods? Unusual sale conditions (check `SaleCondition`)? Checkpoint: at least three written observations, and one concrete "what I would try next" per pattern.
- [ ] Try the single most promising fix, measure it on local CV, and record the result, even if it does not help. Negative results belong on the scoreboard too.

*Milestone 7 — The README that gets people hired (days 12–14).*

- [ ] Write `README.md` in the work folder with: (1) the problem in two sentences, (2) three EDA findings with one chart, (3) the approach as the story of the workflow just completed, (4) the full scoreboard table from `SCOREBOARD.md`, (5) what the error analysis revealed, (6) "what I'd try next": a ranked list of 3+ ideas not yet tried. Checkpoint: a stranger could understand what was done and why in five minutes without opening a notebook.
- [ ] Keep the competition data out of git: anyone can download it from Kaggle, so the repo needs only the code and the write-up. From the repo root, run `echo "work/20-end-to-end-ml-project/data/" >> .gitignore` (if the data was already committed, also run `git rm -r --cached work/20-end-to-end-ml-project/data`). Checkpoint: `git status -u` lists nothing inside `data/`. Then commit and push everything else, **open `work/20-end-to-end-ml-project/README.md` on the fork's GitHub page (the fork is already public), and share it** with a friend or anywhere fellow learners gather. A public, honest write-up of a real project is worth more than any certificate. This step marks the lesson's milestone: the complete ML workflow, carried out end to end on a problem that nobody had simplified in advance.

<details><summary>Hints</summary>

- **Train and test must be preprocessed identically.** Fit the pipeline on training data only and call `.predict()` on test. Never `fit` anything on test data (that is leakage, lesson 19). If one-hot encoding breaks on unseen categories, use `OneHotEncoder(handle_unknown="ignore")`.
- **Work in log space everywhere.** Fit on `np.log1p(y)`, blend predictions in log space, and only `np.expm1` at the very end when writing the submission file.
- **When local CV improves but the leaderboard doesn't** (or vice versa), suspect the outliers found in EDA question 4. Try the pipeline with and without dropping them, and watch both numbers.
- **For the worst-20 analysis**, `df.assign(err=np.abs(pred - actual)).nlargest(20, "err")` gives the table; sorting by *relative* error (`err / actual`) tells a different, equally interesting story.

</details>

**Definition of done.** At least five scored submissions on the Kaggle leaderboard ending at ~0.125 or better, a scoreboard logging every attempt, a written error analysis of the 20 worst predictions, and a README that its author would be proud to show an interviewer.

## Stretch goals

- **Stacking**: instead of hand-picked blend weights, train a small linear model whose inputs are the other models' cross-validated predictions. The result is a proper stacked ensemble (try `StackingRegressor`).
- **Second competition, from memory**: enter **Spaceship Titanic** on Kaggle (a classification problem) and repeat all seven milestones in one week, using this lesson only as a checklist. Doing the workflow twice is what makes it second nature.
- **Feature importance deep-dive**: use permutation importance (`sklearn.inspection.permutation_importance`, which shuffles one feature at a time and measures how much the score drops) on the final model and check whether the top features match the intuitions from EDA; investigate any surprise.
- **Read the masters**: after finishing, read two or three top-voted solution write-ups in the competition's discussion tab and note what they did that this project did not.

## Getting unstuck

- `Authentication required to call the Kaggle API` means the Kaggle tool found no valid token: check that `cat ~/.kaggle/access_token` prints the token copied from Kaggle, or generate a new one and save it again. Kaggle's older kind of key works too: **Create Legacy API Key** (under **Legacy API Credentials** on the same settings page) downloads a `kaggle.json` file. Move it into place with `mv ~/Downloads/kaggle.json ~/.kaggle/` on macOS and Linux, or on Windows (WSL2), where Windows files are under `/mnt/c/Users/`, with `mv /mnt/c/Users/YOUR-WINDOWS-NAME/Downloads/kaggle.json ~/.kaggle/`; then run `chmod 600 ~/.kaggle/kaggle.json`.
- `403 Forbidden` on download almost always means the competition has not been joined yet. Click **Join Competition** on the website and accept the rules.
- Kaggle rejects the submission file? Open it and `sample_submission.csv` side by side: same column names (`Id,SalePrice`), same number of rows (1459), no index column (`index=False`).
- Score got dramatically *worse* after a change? The predictions were probably log-prices, submitted as if they were dollars. Check that the submission's values are in the hundreds of thousands, not around 12.
- A pipeline crash mid-`cross_val_score` usually means a column with NaN slipped past the imputer. To find it, print `X.isna().sum()[lambda s: s > 0]`.
- `Sparse data was passed for X, but dense data is required` from `HistGradientBoostingRegressor` means the one-hot encoder's output is sparse (mostly zeros, stored compactly), which that model does not accept. Create the encoder as `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`.
- Standing advice: read the error message bottom-up (the last line names the real problem), print shapes and a few values at every step, ask an AI assistant for a **hint** rather than a solution, and type all code by hand: copy-pasting teaches the clipboard, not the learner.

## Resources

- [House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) — the competition for this lesson: data, rules, leaderboard, and the discussion tab full of ideas.
- [Kaggle: Intermediate Machine Learning](https://www.kaggle.com/learn/intermediate-machine-learning) — short free course covering missing values, pipelines, leakage and XGBoost; a good companion to milestones 3–5.
- [scikit-learn.org](https://scikit-learn.org) — API reference for `Pipeline`, `cross_val_score`, `GridSearchCV`, `StackingRegressor`.

## Skills unlocked

- [ ] I can take a raw dataset and run the full workflow (frame, EDA, baseline, features, models, tune, analyze errors, report) without a tutorial guiding every step.
- [ ] I always establish a dumb baseline before building anything clever, and I can say why.
- [ ] I keep a validation scoreboard and can explain why local CV must mirror the real metric.
- [ ] I can engineer features from EDA findings and measure honestly whether each one helped.
- [ ] I can tune hyperparameters with GridSearchCV or Optuna and blend models into an ensemble that beats its parts.
- [ ] I can read a model's worst mistakes and turn them into a ranked list of next experiments.
- [ ] I have a public project README that presents the work professionally.

## Next up

With classical ML covered, the next lesson turns to the ideas behind the current wave of AI, starting with the simplest neural network there is: [21 · Neural Networks: The Intuition (Perceptron and XOR)](../phase-3-deep-learning/21-neural-networks-intuition.md).
