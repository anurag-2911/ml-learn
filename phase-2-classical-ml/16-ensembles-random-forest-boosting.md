# 16 · Ensembles: Random Forests and Gradient Boosting

**Phase 2 — Classical ML** · Estimated time: 1-2 weeks · Prerequisites: [15 · Decision Trees from Scratch](15-decision-trees.md), [14 · Model Evaluation](14-model-evaluation.md), [13 · Logistic Regression](13-logistic-regression.md)

> In lesson 15 you built a decision tree and watched it memorize noise the moment you let it grow deep. This lesson holds the fix, and it is delightfully weird: instead of building one better model, you build a hundred mediocre ones and let them vote. That idea — the **ensemble** (a group of models combined into one) — powers random forests and gradient boosting, and here is the industry secret stated plainly: for most tabular business data (spreadsheet-shaped rows and columns), gradient boosting is *still* the strongest tool in existence. Deep learning wins at perception — images, audio, text — but on tables, the models you build this week routinely beat neural networks. Kaggle leaderboards prove it year after year.

## What you will build

- **Project 1 — Crowd wisdom demo**: a script that trains 100 of your lesson-15 trees on random resamples of Titanic and a plot showing majority-vote accuracy climbing as the crowd grows from 1 to 100 trees.
- **Project 2 — Random forest from scratch**: a `RandomForest` class with `fit`/`predict`, built from your own tree code plus random feature subsets, benchmarked head-to-head against scikit-learn's version.
- **Project 3 — Boosting in practice**: tuned XGBoost models on Titanic and California housing, plus a final results table comparing every model you have built in Phase 2.

## Concepts you will learn by doing

- **Wisdom of crowds**: averaging many noisy opinions cancels their individual errors (variance reduction).
- **Bootstrap sampling**: building a "new" dataset by sampling your rows *with replacement* — the heart of **bagging**.
- **Random feature subsets**: why forcing each split to ignore most columns makes trees disagree, and why disagreement is the whole point (decorrelation).
- **Out-of-bag evaluation**: free test data hiding inside bagging — the rows each tree never saw.
- **Boosting**: building models *in sequence*, where each new model focuses on fixing the previous ones' mistakes.
- **Gradient boosting** at intuition level: each new tree predicts the current ensemble's errors.
- **XGBoost in practice**: the library that wins tabular ML competitions, and how to tune it honestly with cross-validation.
- **When deep learning is NOT the answer**: recognizing tabular problems where boosting is the right tool.

## Before you start

1. You need your decision tree code from lesson 15 and your metrics library from lesson 14. Confirm they exist:

   ```bash
   ls work/15-decision-trees/ work/14-model-evaluation/
   ```

2. You need a local Titanic CSV. Lesson 15 loaded the data through seaborn, which caches it outside your work folder — so you probably have no `titanic.csv` file yet; step 4 below creates one.

3. Activate the venv at the repo root and install XGBoost (scikit-learn downloads California housing for you, as in lesson 12). On a Mac, XGBoost also needs Homebrew's OpenMP library (libomp), which lets it use all your CPU cores:

   - **macOS on Apple Silicon with macOS 15 or later:** run `brew install libomp` before the commands below.
   - **macOS on an Intel Mac, or on macOS 14 and older:** without Homebrew, XGBoost cannot load, so leave out the last two lines below. In Project 3, use scikit-learn's own gradient boosting instead, which brings its own OpenMP: `HistGradientBoostingClassifier` and `HistGradientBoostingRegressor` from `sklearn.ensemble`, wherever the lesson uses `XGBClassifier` and `XGBRegressor`. They take the same `max_depth` and `learning_rate`, and `max_iter` in place of `n_estimators`. Or do Project 3 in [Google Colab](https://colab.research.google.com), a free cloud notebook with XGBoost preinstalled.

   ```bash
   cd ~/ml/ml-learn
   source .venv/bin/activate
   pip install xgboost
   python -c "import xgboost; print(xgboost.__version__)"
   ```

4. Create your work folder and save the Titanic data into it via seaborn (if you saved your own `titanic.csv` in an earlier lesson, copying that here works too):

   ```bash
   mkdir -p work/16-ensembles
   python3 -c "import seaborn as sns; sns.load_dataset('titanic').to_csv('work/16-ensembles/titanic.csv', index=False)"
   ```

5. Copy (or import) your lesson-15 tree into this folder so you can modify it freely without breaking lesson 15.

## Project 1 — Crowd wisdom demo

**Goal**: Prove to yourself, with a plot, that 100 mediocre trees voting together beat any single one of them. This is bagging — **b**ootstrap **agg**regat**ing** — and once you see the curve you will never forget it.

**Milestones**

- [ ] Load Titanic and prepare it exactly as you did in lesson 15 (same features, same train/test split — use a fixed random seed so results are repeatable). Keep roughly 80% for training, 20% for testing.
- [ ] Write a function `bootstrap_sample(X, y)` that returns a new dataset the *same size* as the original, built by picking rows at random **with replacement** (the same row can appear twice; on average about 37% of rows are left out entirely). This is a bootstrap sample. Checkpoint: run it, then count how many *unique* original rows made it in — you should see roughly 63% of them.
- [ ] Train one of your lesson-15 trees (depth limit around 5-8) on one bootstrap sample and record its test accuracy. Repeat 100 times, storing all 100 trees and all 100 individual accuracies. Checkpoint: individual tree accuracies scatter over a visible range — typically somewhere around 0.74-0.82. That scatter *is* variance: each tree overfits its own sample differently.
- [ ] Write `majority_vote(trees, X)`: every tree predicts, and each row's final answer is whichever class got more votes. For k = 1, 2, 3, ..., 100, compute the test accuracy of the first k trees voting together.
- [ ] Plot ensemble accuracy vs. number of trees (matplotlib, from lesson 7). Add a horizontal dashed line at the *best* single tree's accuracy. Checkpoint: the curve is noisy at the left, climbs, then flattens — and the flat part sits at or above your best single tree, and clearly above the *average* tree. The crowd beats its members.
- [ ] One-paragraph note in a comment or README: why does averaging reduce variance but not bias? (Hint: 100 copies of the *same* wrong opinion don't cancel; 100 *different* wrong opinions do.)
- [ ] **Out-of-bag bonus**: for each tree, the ~37% of rows it never trained on are its **out-of-bag (OOB)** rows — a free, built-in test set. For each training row, collect votes only from trees that did *not* see it, and compute OOB accuracy. Checkpoint: OOB accuracy lands within a couple of points of your held-out test accuracy — you just evaluated without spending any test data.

<details><summary>Hints</summary>

- `np.random.choice(n, size=n, replace=True)` gives you bootstrap row indices in one line.
- For majority vote with 0/1 labels, stack all predictions into a `(n_trees, n_rows)` array; the vote is just `mean(axis=0) >= 0.5`.
- If every tree gets identical accuracy, you probably forgot `replace=True`, or you're accidentally training on the full dataset each time.
- For OOB tracking, store each tree's *in-bag index set*; a row is out-of-bag for a tree if its index isn't in that set.

</details>

**Definition of done**: A saved plot showing majority-vote accuracy vs. crowd size 1→100, where the flattened ensemble beats your best single tree, plus a computed OOB accuracy close to test accuracy.

## Project 2 — Random forest from scratch

**Goal**: Bagging alone helps, but the trees are still too similar — they all grab the same strong features (on Titanic, everybody splits on sex first). A **random forest** fixes this by letting each split consider only a small random subset of features, forcing the trees to disagree. Build it, then race scikit-learn.

**Milestones**

- [ ] Modify your lesson-15 tree so that at *every split*, instead of scanning all features for the best split, it first picks a random subset of features (a common default: the square root of the number of features, rounded) and scans only those. Make the subset size a parameter, `max_features`.
- [ ] Sanity-check the modified tree alone: train it a few times on the full Titanic training set. Checkpoint: different runs now produce *different* root splits — the trees have become individuals. (Your original tree picked the same root every time.)
- [ ] Write a `RandomForest` class with `__init__(n_trees=100, max_depth=8, max_features='sqrt')`, `fit(X, y)` (bootstrap sample + randomized tree, n_trees times), and `predict(X)` (majority vote). This is Project 1's loop, packaged the way scikit-learn packages models — a shape you'll now recognize everywhere.
- [ ] Train your forest on Titanic and evaluate with your lesson-14 metrics (accuracy, precision, recall). Checkpoint: test accuracy around 0.80 or better, and at least as good as your Project 1 bagging ensemble — usually a touch better.
- [ ] Now the race: train `sklearn.ensemble.RandomForestClassifier` with the same `n_estimators` and `max_depth` on the same split. Checkpoint: your forest lands within a few points of sklearn's accuracy. Sklearn will be dramatically faster — that's optimized C code, not a smarter algorithm. The *ideas* are identical, and now you own them.
- [ ] Ask sklearn's forest which features mattered: print `model.feature_importances_` next to your feature names, sorted. Checkpoint: sex and fare (or ticket class) sit near the top — the forest rediscovered what you found by hand in lesson 15.

<details><summary>Hints</summary>

- The *only* change inside the tree is where you loop over candidate features: loop over `np.random.choice(n_features, size=k, replace=False)` instead of `range(n_features)`.
- Pick the random subset *per split*, not per tree — per-tree is a different (weaker) variant.
- If your forest does *worse* than plain bagging, your `max_features` may be too small for Titanic's handful of features; try `max_features=3`.
- Keep `n_trees=20` while debugging; go to 100 only when it works. Print progress every 10 trees so you know it's alive.

</details>

**Definition of done**: A `RandomForest` class whose `fit`/`predict` matches sklearn's `RandomForestClassifier` within a few points of accuracy on Titanic, plus a sorted feature-importance printout.

## Project 3 — Boosting in practice

**Goal**: Bagging builds trees *in parallel* and averages them. **Boosting** builds them *in sequence*: each new small tree is trained to correct the mistakes the ensemble has made so far. **Gradient boosting** is the modern form — each new tree predicts the current errors (the gradient of the loss, hence the name), and its predictions are added on with a small **learning rate** (a shrink factor that keeps any single tree from overcorrecting). You will not build this one from scratch; you'll master the tool everyone actually uses: XGBoost.

**Milestones**

- [ ] Warm up your intuition with 20 lines of code: fit a depth-2 *regression* tree to some 1-D toy data — use `sklearn.tree.DecisionTreeRegressor(max_depth=2)` for both stages (or your lesson-15 regression-tree stretch goal if you built it; your lesson-15 *classifier* can't fit residuals) — then compute the residuals (true minus predicted), fit a second tree *to the residuals*, and add the two predictions. Checkpoint: the two-tree sum fits the data visibly better than either tree alone. That is gradient boosting's entire trick — everything else is refinement.
- [ ] Train a default `xgboost.XGBClassifier` on your Titanic split:

  ```python
  from xgboost import XGBClassifier
  model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1)
  model.fit(X_train, y_train)
  print(model.score(X_test, y_test))
  ```

  Checkpoint: accuracy in the same neighborhood as your random forest (Titanic is small and noisy — don't expect miracles; expect parity or a small edge).
- [ ] Tune it honestly with cross-validation (lesson 14): use `sklearn.model_selection.GridSearchCV` over a small grid — `n_estimators` in {50, 100, 300}, `max_depth` in {2, 3, 4}, `learning_rate` in {0.03, 0.1, 0.3} — with `cv=5`. Print the best parameters and best CV score. Checkpoint: you can explain the pattern you see: lower learning rate wants more trees (smaller steps, more of them), and deep trees overfit small data.
- [ ] Switch to regression: load California housing (`sklearn.datasets.fetch_california_housing`) — 20,000+ rows of census data predicting median house value. Train `xgboost.XGBRegressor`, tune the same three knobs, and evaluate with RMSE (root-mean-squared error — one line: `np.sqrt(np.mean((y_pred - y)**2))`; add an `rmse` function to your lesson-14 `mymetrics.py` now and validate it against sklearn). Checkpoint: tuned test RMSE around 0.45-0.55 (the target is in units of $100k, so that's roughly ±$50k typical error). Compare against plain `LinearRegression` on the same split — XGBoost should be clearly better, because house prices depend on *interactions* (location × income) that a straight line can't express.
- [ ] Build the Phase-2 grand finale table. On identical Titanic splits, evaluate: your lesson-13 logistic regression, your lesson-15 single tree, your Project-2 forest, sklearn's forest, and tuned XGBoost. Write it as a markdown table in `work/16-ensembles/RESULTS.md` with columns: model, accuracy, training time, "from scratch?". Checkpoint: the ensembles sit at the top, and you can say *why* each row landed where it did.
- [ ] Read the table and write three sentences at the bottom of RESULTS.md on the industry secret: on tabular data like this, boosted trees and forests are the state of the art — reach for deep learning when the input is images, audio, or raw text (Phase 3 onward), not when it's a spreadsheet. Optional: `pip install lightgbm` and add LightGBM (Microsoft's faster cousin of XGBoost, same ideas) as one more row. **macOS:** it needs libomp just like XGBoost, so skip it on a Mac without Homebrew. **Windows (WSL2) and Linux:** it needs Ubuntu's OpenMP library, which WSL's Ubuntu does not include, so first run `sudo apt install -y libgomp1`.

<details><summary>Hints</summary>

- If XGBoost complains about your labels or dtypes, make sure `y` is integer 0/1 and `X` is a numeric NumPy array or DataFrame — no strings left over from Titanic preprocessing.
- `GridSearchCV(model, param_grid, cv=5, n_jobs=-1)` uses all CPU cores; the 27-combination grid takes a couple of minutes, not hours.
- For the residual warm-up, `plt.scatter` the data and `plt.plot` each stage's prediction — seeing tree 2 chase tree 1's leftover errors is the moment it clicks.
- A suspiciously *perfect* score usually means the target leaked into your features or you evaluated on training data.

</details>

**Definition of done**: Tuned XGBoost models on both datasets with CV-chosen hyperparameters, and a RESULTS.md table comparing at least five models on Titanic with a written conclusion.

## Stretch goals

- Implement out-of-bag scoring *inside* your `RandomForest` class as `oob_score_`, computed during `fit`, and verify it against sklearn's `oob_score=True`.
- Write gradient boosting from scratch for regression: start from the mean, loop "fit small tree to residuals, add prediction × learning_rate", and watch training RMSE fall each round. ~40 lines using your lesson-15 tree.
- Plot validation error vs. `n_estimators` for XGBoost with a huge tree count (2000) and a tiny learning rate — find the point where more trees stop helping, then read about `early_stopping_rounds` in the XGBoost docs and use it.
- Feature-importance face-off: compare your forest's importances with XGBoost's on California housing. Do they agree on the top three features?

## If you get stuck

- **Ensemble not beating single trees?** Check that your trees actually differ: print each tree's root split. If they're identical, your bootstrap sampling or feature subsetting isn't wired in.
- **Shape errors in the vote?** Print `predictions.shape` — you want `(n_trees, n_rows)` before voting. A transposed array votes across trees instead of rows and produces garbage silently.
- **Wildly different numbers each run?** Set seeds (`np.random.seed`, `random_state=`) everywhere while debugging; remove them only when checking robustness.
- Standing advice: read error tracebacks from the bottom up; print shapes and a few actual values at every step; when stuck 30+ minutes, ask an AI assistant for a *hint* ("what concept am I missing?"), never the solution; and type every line yourself — copy-paste teaches nothing.

## Resources

- **StatQuest: Random Forests, Part 1 & 2** (YouTube — search "StatQuest Random Forests"): the clearest visual walkthrough of bagging, feature subsets, and OOB error; watch before or during Project 2.
- **StatQuest: Gradient Boost series** (YouTube — search "StatQuest Gradient Boost"): four short parts building the boosting math one careful step at a time; watch part 1 before Project 3.
- **XGBoost documentation**: https://xgboost.readthedocs.io/ — installation, the scikit-learn-style API you're using, and the full parameter list for tuning.
- **scikit-learn ensemble guide**: https://scikit-learn.org/ (User Guide → Ensembles) — reference for `RandomForestClassifier` parameters when benchmarking your own.

## Skills unlocked

- [ ] I can explain why averaging many overfit models reduces variance, in plain words.
- [ ] I can write bootstrap sampling in one line of NumPy and state what fraction of rows it leaves out.
- [ ] I can explain why random forests restrict features per split, and what goes wrong without it.
- [ ] I built a working `RandomForest` class that lands within a few points of scikit-learn's.
- [ ] I can explain out-of-bag evaluation and why it's "free" test data.
- [ ] I can describe gradient boosting as "each tree fixes the previous trees' errors" and say what the learning rate does.
- [ ] I can tune an XGBoost model with cross-validation and justify the chosen hyperparameters.
- [ ] I know when to reach for boosted trees instead of deep learning — and why.

## Next up

Your trees have conquered tables — next you'll teach classical ML to read, building a spam filter with Naive Bayes in [17 · Text Classification: Naive Bayes Spam Filter (+ SVM)](17-naive-bayes-spam-svm.md).
