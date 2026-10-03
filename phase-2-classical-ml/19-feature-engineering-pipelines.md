# 19 · Feature Engineering and Pipelines

**Phase 2 — Classical Machine Learning** · Estimated time: 1 week · Prerequisites: [16 · Ensembles](16-ensembles-random-forest-boosting.md), [14 · Model Evaluation](14-model-evaluation.md), [06 · Pandas](../phase-0-foundations/06-pandas.md)

> On tabular data (data in rows and columns, like spreadsheets), the winning move in applied machine learning is rarely a fancier model. It is better *features*, the input columns fed to the model, plus the discipline to measure them honestly. This lesson teaches both. It invents new features for the Titanic dataset and proves with numbers which ones help, hunts down three subtle data leaks that make models look better than they are, and then wraps everything into a single leak-proof sklearn Pipeline object that the capstone reuses. This unglamorous skill wins Kaggle competitions and real jobs alike.

## What this lesson builds

- **Project 1 — Feature workshop:** a Titanic feature-engineering script plus a scoreboard table proving exactly how many accuracy points each new feature is worth.
- **Project 2 — The leak hunt:** written diagnoses and fixed versions of three subtly-broken preprocessing snippets, each hiding a data leak.
- **Project 3 — One pipeline to rule them all:** a single reusable `Pipeline` object (imputing + scaling + encoding + model) tuned end-to-end with `GridSearchCV` and used as the template for lesson 20, plus a side-by-side comparison of plain, Ridge and Lasso linear regression.

## Concepts covered

- Numeric scaling: `StandardScaler` vs `MinMaxScaler`, and which models actually care (revisiting [lesson 12](12-linear-regression.md))
- Regularization: Ridge and Lasso, which are linear regression plus a penalty that shrinks weights to fight overfitting (the idea that [lessons 12](12-linear-regression.md) and [13](13-logistic-regression.md) promised for this lesson)
- Categorical encoding: one-hot vs ordinal, and when each is right
- Imputing missing values, and treating "it's missing" as a signal in itself
- Engineering features from domain sense: ratios, interactions, date parts, text lengths
- sklearn `Pipeline` and `ColumnTransformer` for bundling preprocessing + model
- Why pipelines prevent leakage: everything fits on training data only
- Target leakage: real war stories of models that were secretly cheating

## Before starting

- This lesson needs [lesson 16](16-ensembles-random-forest-boosting.md) finished, with a working random forest on Titanic, and cross-validation from [lesson 14](14-model-evaluation.md) (testing a model on several train/test splits and averaging the scores).
- Activate the venv and confirm that the tools are installed:

```bash
cd ~/ml/ml-learn
source .venv/bin/activate
pip install pandas scikit-learn matplotlib
```

- Create the work folder:

```bash
mkdir -p work/19-feature-engineering-pipelines
cd work/19-feature-engineering-pipelines
```

- **Dataset:** this lesson needs the *original* Kaggle-style Titanic file, the one with the `Name` and `Cabin` columns and capitalized headers (`Pclass`, `SibSp`, `Embarked`, ...). **The seaborn dataset loaded in earlier lessons will not work here**: its columns are lowercase and it has no `Name` or `Cabin` at all, and Features 1 and 4 below depend on exactly those. Download a no-signup mirror straight into this folder:

```bash
curl -L -o train.csv https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv
```

  (Or get `train.csv` from the Titanic competition on [kaggle.com](https://www.kaggle.com). That requires a free account, which lesson 20 needs anyway.) Sanity check: the header line of `train.csv` lists `Name` and `Cabin` among its columns.

## Project 1 — Feature workshop on Titanic

**Goal:** Engineer at least 5 new features from the raw Titanic columns and measure, with cross-validation, exactly how much each one improves the lesson-16 random forest. Opinions do not count; only the scoreboard counts.

**Milestones**

- [ ] Create `features.py`. Load `train.csv` and build a **baseline**: the lesson-16 random forest using only the raw usable columns (`Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`; encode `Sex`/`Embarked` as numbers and fill missing `Age` with the median for now). Score it with `cross_val_score(model, X, y, cv=5)` and record the mean. Checkpoint: baseline mean accuracy around 0.80–0.83.
- [ ] Start a `SCOREBOARD.md` file with a markdown table: columns `feature added`, `CV accuracy`, `change vs baseline`. The first row is the baseline. Each new feature adds one row.
- [ ] **Feature 1 — `title`:** every passenger's `Name` contains a title ("Mr.", "Mrs.", "Master.", "Rev.", ...). Extract it with a pandas string operation, group rare titles into an "Other" bucket, and one-hot encode it. *One-hot encoding* means turning one category column into several 0/1 columns, one per category. Models need numbers, and one-hot avoids inventing a fake order between categories. Re-run CV and add a scoreboard row. Checkpoint: `df["title"].value_counts()` shows Mr, Miss, Mrs, Master as the top four.
- [ ] **Feature 2 — `family_size`:** `SibSp + Parch + 1` (siblings/spouses + parents/children + self). One line of domain sense: people traveled in groups, and groups lived or died together. Score it and add a row.
- [ ] **Feature 3 — `is_alone`:** 1 when `family_size == 1`, else 0. This is an *interaction-style* feature: a new column that captures a pattern the model would otherwise have to discover itself. Score it and add a row.
- [ ] **Feature 4 — `deck`:** the first letter of `Cabin` (A–G, plus a single passenger on deck T). Most values are missing. Keep them as their own category `"U"` for unknown, because *missingness is a signal*: not having a recorded cabin correlates with lower-class tickets. Score it and add a row.
- [ ] **Feature 5 — `fare_per_person`:** `Fare / family_size`, a *ratio feature*. A £60 fare means something different for a family of six than for a solo traveler. Score it and add a row.
- [ ] Invent at least one feature beyond these five (ideas: `age_bucket` child/adult/senior; `name_length`, since text lengths are surprisingly predictive; `Pclass * is_alone`). Score it honestly even if it flops: flops belong on the scoreboard too.
- [ ] Final run: all features together. Checkpoint: CV accuracy typically lands around 0.81–0.84, often within a point of the baseline and sometimes slightly below it with a random forest. Gaps under about one point are cross-validation noise (print `scores.std()` to see it). Also note which features *did not* help; knowing that is the skill.

<details><summary>Hints</summary>

- Titles: `df["Name"].str.extract(r" ([A-Za-z]+)\.")` grabs the word ending in a period. Print `value_counts()` before deciding which titles are "rare".
- Use `pd.get_dummies(df, columns=["title", "deck", ...])` for quick one-hot encoding in this project; Project 3 does it the production way.
- Keep a function `evaluate(df_features)` that builds X/y, runs `cross_val_score`, and returns the mean. Then each new feature is two lines plus a function call.
- Fix `random_state` on the model and use the same `cv=5` everywhere, or the scoreboard compares noise, not features.
</details>

**Definition of done:** `SCOREBOARD.md` has 7+ rows with real CV numbers, and the single feature that was worth the most can be named out loud.

## Project 2 — The leak hunt

**Goal:** *Data leakage* is when information from outside the training data (often from the test set, or from the answer itself) sneaks into training, producing scores that collapse in the real world. Below are three snippets that each look reasonable and are each broken. Diagnose and fix all three.

**Milestones**

- [ ] Create `leak_hunt.md`. For each snippet below, write: (a) where the leak is, (b) why the reported score is too optimistic, (c) the corrected code. Then actually run the broken and fixed versions on Titanic and record both scores.

- [ ] **Snippet A — the eager scaler:**

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)          # scale everything first
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=42)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
```

  The scaler computed the mean and standard deviation of *all* rows, including the test rows. The test set has quietly influenced training. Checkpoint: the write-up names the exact line and states the rule "fit on train only, transform both".

- [ ] **Snippet B — the eager imputer:**

```python
imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(X)                # fill missing ages using ALL rows
scores = cross_val_score(model, X, y, cv=5)
print(scores.mean())
```

  Same disease, sneakier setting: inside each CV fold, the "unseen" validation rows already contributed to the median used to fill them. On Titanic the score barely moves, which is exactly why this leak survives code review in real companies. Checkpoint: the fix puts the imputer *inside* the cross-validation (hint: that is what Pipelines are for; a `Pipeline` passed to `cross_val_score` refits the imputer per fold).

- [ ] **Snippet C — the answer in disguise:**

```python
df["group_survival_rate"] = df.groupby("Ticket")["Survived"].transform("mean")
X = df[["Pclass", "Fare", "group_survival_rate"]]
scores = cross_val_score(model, X, y, cv=5)
print(scores.mean())                        # wow, 0.90+!
```

  This is *target leakage*: the new feature is computed **from the target column itself**. Each passenger's own outcome is baked into their group's average. Checkpoint: the broken version scores suspiciously high (0.90+); after the fix (drop the feature, or compute it per-fold from training rows only, excluding the current row), the score returns to earth.

- [ ] Add a short "war stories" section to `leak_hunt.md`: in the learner's own words, retell why this matters. Two famous patterns to look up: models predicting hospital outcomes using data that was only recorded *after* the outcome (e.g. treatments given), and Kaggle competitions won by exploiting leaked identifiers rather than real signal. The lesson: **a score that jumps too easily is an accusation, not a gift.**

<details><summary>Hints</summary>

- For A and B, `from sklearn.pipeline import make_pipeline` then `make_pipeline(SimpleImputer(), StandardScaler(), model)` handed to `cross_val_score` is the whole fix.
- To *see* leak C, print `df[["Ticket", "Survived", "group_survival_rate"]].head(10)`: passengers on solo tickets have a "feature" that literally equals their answer.
- Rule of thumb for spotting target leakage: "Would I know this value at prediction time, before the outcome exists?" If not, it is leakage.
</details>

**Definition of done:** `leak_hunt.md` diagnoses all three leaks with before/after scores, and the fit-on-train-only rule can be stated from memory.

## Project 3 — One pipeline to rule them all

**Goal:** Build one `Pipeline` object that takes the raw Titanic dataframe and does everything (impute, scale, encode, predict) with every step fitted on training data only, then tune preprocessing and model hyperparameters *together* with `GridSearchCV`. This object is the reusable template for lesson 20.

**Milestones**

- [ ] Read the [sklearn Pipeline guide](https://scikit-learn.org/stable/modules/compose.html) sections on `Pipeline` and `ColumnTransformer` (skim, 15 minutes). A `Pipeline` chains steps so that `.fit()` fits them in order on training data; a `ColumnTransformer` applies different steps to different columns.
- [ ] In `pipeline.py`, split the columns into two lists: `numeric_features` (e.g. `Age`, `Fare`, `family_size`, `fare_per_person`) and `categorical_features` (e.g. `Sex`, `Embarked`, `title`, `deck`, `Pclass`). Reuse the Project 1 feature-engineering code to create the engineered columns first.
- [ ] Build the preprocessor. Skeleton to adapt (fill in the pieces; this is the shape, not the solution):

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])
preprocess = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
])
```

- [ ] Chain it with the model into one object: `clf = Pipeline([("preprocess", preprocess), ("model", RandomForestClassifier(random_state=42))])`. Run `cross_val_score(clf, X, y, cv=5)` on the **raw** X (no manual preprocessing beforehand). Checkpoint: CV accuracy within a point or two of the Project 1 best, but now with zero leakage by construction, because each fold refits the imputer, scaler and encoder on that fold's training rows only.
- [ ] Tune preprocessing and model together. In a pipeline, parameters are addressed as `stepname__substep__param` (double underscores walk down the tree). Grid-search at least: `preprocess__num__imputer__strategy` (`"median"` vs `"mean"`), `model__n_estimators` (e.g. 100, 300), and `model__max_depth` (e.g. 5, 10, None). Print `grid.best_params_` and `grid.best_score_`. Checkpoint: `GridSearchCV(clf, param_grid, cv=5)` runs without errors and `best_score_` matches or beats the plain CV score.
- [ ] Prove reusability: with `joblib.dump(grid.best_estimator_, "titanic_pipeline.joblib")`, save the fitted pipeline, reload it in a fresh Python session, build a hand-made dataframe with one imaginary passenger in the `train.csv` columns, run it through the Project 1 feature-engineering code (the pipeline expects `title`, `family_size`, `deck` and `fare_per_person` to exist already), and call `.predict()` on it. Checkpoint: one `.predict()` call returns 0 or 1, with no manual imputing, scaling or encoding. (The first stretch goal starts moving the feature engineering inside the pipeline too.)
- [ ] Swap `RandomForestClassifier` for another model from earlier lessons (logistic regression, gradient boosting) by changing **one line**. Notice that scaling now actually matters for logistic regression ([lesson 12](12-linear-regression.md) foreshadowed this: gradient-based and distance-based models care about feature scale; trees do not). Then swap `StandardScaler` for `MinMaxScaler` in the numeric transformer (`MinMaxScaler` squeezes each column into the range 0 to 1; `StandardScaler` centers it at mean 0 with standard deviation 1) and compare the CV scores of logistic regression and the random forest.
- [ ] **Regularization — Ridge and Lasso:** **Ridge** is linear regression plus a penalty on the size of the weights. The penalty *shrinks* them, which tames overfitting. **Lasso**'s penalty can shrink a weight all the way to zero, deleting a useless feature automatically. In `regularization.py`, load the California housing data from [lesson 12](12-linear-regression.md) with `fetch_california_housing(as_frame=True, return_X_y=True)`, split it with `train_test_split(X, y, test_size=0.2, random_state=42)`, and fit `LinearRegression()`, `Ridge(alpha=1000)` and `Lasso(alpha=0.01)` (all in `sklearn.linear_model`), each inside `make_pipeline(StandardScaler(), model)`. The scaler matters here: the penalty charges every weight the same price, so without scaling, a feature's units would decide how hard its weight is shrunk. Print the three models' weights side by side, one row per feature, and each model's test R². Checkpoint: Lasso sets the `Population` weight to exactly zero (it may print as `-0.000000`), Ridge pulls `Latitude` and `Longitude` from about -0.9 to about -0.5, and the three test R² scores land within about 0.03 of each other.

  Why no real gain? With 8 features and over 16,000 training rows there is little overfitting to tame. Regularization pays off when features are many compared with rows, like the many one-hot columns of lesson 20, which uses `Ridge` from its first pipeline. The same penalty has also been at work inside `LogisticRegression` since lesson 13: its `C` parameter is the inverse of the penalty strength, so a smaller `C` shrinks the weights harder.

<details><summary>Hints</summary>

- `handle_unknown="ignore"` on `OneHotEncoder` avoids an error when a CV fold or future data contains a category the training fold never saw.
- If `ColumnTransformer` complains about column names, compare the names its error lists (it lists every missing one at once) with `X.columns.tolist()` to find typos or engineered columns that were never created.
- `grid.cv_results_` is a dict that can be loaded into `pd.DataFrame` to see every combination's score, not just the winner.
- Keep the grid small (≤ 24 combinations) or a 5-fold search gets slow; widen it once it works.
- `pipe[-1]` is the last step of a pipeline, so `pipe[-1].coef_` holds the fitted model's weights. `pd.DataFrame({"linear": ..., "ridge": ..., "lasso": ...}, index=X.columns)` lines the three sets of weights up by feature name.
- Ridge limits the *total* size of the weights, not each weight separately, so a small weight such as `HouseAge` can grow a little while the large ones fall.
</details>

**Definition of done:** `pipeline.py` produces a tuned, saved, reloadable pipeline that predicts in one call from a dataframe that has the Project 1 columns, and the reason it cannot leak can be explained. `regularization.py` prints the three models' weights side by side, with Lasso's zero visible.

## Stretch goals

- Add a custom transformer: write a class with `fit` and `transform` methods (or use `FunctionTransformer`) that does the title-extraction *inside* the pipeline, so even feature engineering happens per-fold.
- Try `OrdinalEncoder` instead of one-hot for `deck` and `Pclass` and measure the difference. Then write two sentences on when an ordinal (ordered) encoding is justified.
- Add a `missing_Age` indicator column (1 if `Age` was missing) via `SimpleImputer(add_indicator=True)` and check whether missingness itself predicts survival.
- Do the [Kaggle feature engineering course](https://www.kaggle.com/learn/feature-engineering) and apply one technique it teaches (e.g. mutual information ranking) to the Titanic features.

## Getting unstuck

- **Shape errors after `ColumnTransformer`:** its output is a plain array with reordered columns. Use `preprocess.get_feature_names_out()` to see what came out, and print `.shape` before and after each step.
- **"could not convert string to float":** a text column slipped into the numeric list, or raw X was passed to a bare model instead of the pipeline. Print `X.dtypes`.
- **Scores identical for every grid combination:** the tuned parameter probably has no effect on this data, for example an imputer strategy on columns with no missing values. A misspelled name is not the cause: sklearn raises `ValueError: Invalid parameter ...` for a name that does not exist, and `clf.get_params().keys()` lists every legal `step__param` name. Load `grid.cv_results_` into `pd.DataFrame` to see which parameter actually changes the score.
- Standing advice: read the error traceback bottom-up (the last line names the real problem), print shapes and `head()`s liberally, ask an AI assistant for a **hint** not a solution, and type all code by hand (muscle memory is the point).

## Resources

- [sklearn Pipeline and ColumnTransformer guide](https://scikit-learn.org/stable/modules/compose.html) — the official docs for everything in Project 3; the examples are short and worth typing out.
- [Kaggle: Feature Engineering course](https://www.kaggle.com/learn/feature-engineering) — free short course; good second pass on mutual information, target encoding done safely, and clustering features.

## Skills unlocked

- [ ] I can explain the difference between StandardScaler and MinMaxScaler, and name which models need scaling and which do not.
- [ ] I can explain what Ridge and Lasso add to linear regression, and why only Lasso can delete a feature.
- [ ] I can choose between one-hot and ordinal encoding and justify the choice.
- [ ] I can impute missing values without leaking, and I know when missingness itself is a feature.
- [ ] I can invent features from domain sense (ratios, interactions, extracted text) and measure their worth with cross-validation instead of guessing.
- [ ] I can spot data leakage in someone else's code, including target leakage, and explain why the score was a lie.
- [ ] I can build a ColumnTransformer + Pipeline that goes from raw dataframe to prediction in one fitted object.
- [ ] I can grid-search preprocessing and model hyperparameters together using `step__param` names.
- [ ] I can save a fitted pipeline with joblib and reload it for predictions elsewhere.

## Next up

The next lesson puts every Phase-2 skill together on a real competition, reusing the pipeline object from this lesson: [20 · Capstone: End-to-End ML Project (Kaggle)](20-end-to-end-ml-project.md).
