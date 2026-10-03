# 14 · Model Evaluation: Build Your Own Metrics Library

**Phase 2 — Classical Machine Learning** · Estimated time: 1 week · Prerequisites: [11 · k-Nearest Neighbors](11-first-model-knn.md), [13 · Logistic Regression](13-logistic-regression.md), [07 · Data Visualization](../phase-0-foundations/07-data-visualization.md)

> A model that scores "99% accuracy" can still be completely useless. This lesson builds one on purpose, just to show how it fools the accuracy metric. Evaluation is the difference between someone who *runs* models and someone who can be *trusted* with them. The lesson writes a tiny metrics library, `mymetrics.py`, proves every function correct against scikit-learn, builds k-fold cross-validation from scratch, and shows how to diagnose sick models with learning curves. Almost every lesson from here on reuses `mymetrics.py`.

## What this lesson builds

- **`mymetrics.py`:** a metrics library written from scratch (accuracy, confusion matrix, precision, recall, F1, ROC/AUC), with every function validated against `sklearn.metrics`. On a fake medical test, it exposes a useless model that scores 99% accuracy.
- **`my_kfold.py`:** k-fold cross-validation written from scratch, used to run a *fair* head-to-head between kNN and logistic regression on Titanic, and validated against sklearn's `cross_val_score`.
- **A diagnosis clinic:** learning-curve plots that visually distinguish an underfitting model from an overfitting one, plus a leak hunt that finds and fixes data leakage in a broken pipeline.

## Concepts covered

- **Why accuracy lies**: on imbalanced data (where one class is rare), a model that always predicts the common class looks great and is worthless.
- **Confusion matrix**: a 2x2 table counting every way a prediction can be right or wrong.
- **Precision, recall, F1**: "when I raise the alarm, am I right?" vs "of the real cases, how many did I catch?", and the single number that balances them.
- **ROC curve and AUC**: how good a model is across *all* possible decision thresholds, not just one.
- **k-fold cross-validation**: testing a model k times on k different slices, so that one lucky split cannot give a misleading result.
- **Bias vs variance**: underfitting (too simple to learn the pattern) vs overfitting (memorizing the training data).
- **Learning curves**: plots of performance vs training-set size that reveal which disease a model has.
- **Data leakage**: the cardinal sin, where information from the test set sneaks into training and produces scores that evaporate in the real world.

## Before starting

- Lesson 13 is finished, and the repo root has a working venv.
- Set up this lesson's folder and packages:

```bash
cd ~/ml/ml-learn
source .venv/bin/activate
pip install numpy pandas matplotlib scikit-learn
mkdir -p work/14-model-evaluation
```

- Save a local copy of the Titanic data (lesson 13 loaded it through seaborn, which caches it outside the work folder; this writes an actual CSV that can be reused):

```bash
pip install seaborn
python3 -c "import seaborn as sns; sns.load_dataset('titanic').to_csv('work/14-model-evaluation/titanic.csv', index=False)"
```
- Project 1 needs no download at all: it generates its own medical dataset with NumPy.

## Project 1 — mymetrics.py

**Goal:** Build a metrics library from scratch, and use a fake medical-screening dataset to see first-hand why accuracy lies on imbalanced data. Every function must match scikit-learn's answer exactly.

**Milestones**

- [ ] Create `work/14-model-evaluation/mymetrics.py` and a separate `test_mymetrics.py` that imports it. Keeping tests separate is a habit worth starting now.
- [ ] Generate the patients: 10,000 people, 1% of them actually sick. Give each patient two measurements (features) drawn from normal distributions, with sick patients shifted so the classes overlap but are mostly separable:

  ```python
  import numpy as np
  rng = np.random.default_rng(42)
  n = 10_000
  y = (rng.random(n) < 0.01).astype(int)          # 1 = sick, ~1% of people
  X = rng.normal(0, 1, size=(n, 2))
  X[y == 1] += 2.0                                 # sick patients measure higher
  ```

  Checkpoint: `y.mean()` is close to 0.01 and `y.sum()` is around 100.
- [ ] Write `accuracy(y_true, y_pred)`: the fraction of predictions that are correct. It takes one line of NumPy. Now create the "Dr. Useless" model: `y_pred = np.zeros(n)` (declares *everyone* healthy). Checkpoint: Dr. Useless scores about **0.99 accuracy** while catching **zero** sick patients. This is why accuracy lies.
- [ ] Write `confusion_matrix(y_true, y_pred)` returning the four counts: **TP** (true positives: sick, predicted sick), **FP** (false positives: healthy, predicted sick), **FN** (false negatives: sick, predicted healthy; the dangerous one here), **TN** (true negatives). Return a 2x2 array in sklearn's layout: `[[TN, FP], [FN, TP]]`. Checkpoint: for Dr. Useless, TP = 0 and FN = the number of sick patients.
- [ ] Write `precision`, `recall` and `f1`. **Precision** = TP / (TP + FP): of the alarms raised, how many were real. **Recall** = TP / (TP + FN): of the real cases, how many were caught. **F1** = 2 · (precision · recall) / (precision + recall): their harmonic mean, low if *either* is low. Handle division by zero by returning 0.0. Checkpoint: Dr. Useless has recall **0.0** and F1 **0.0**. These metrics expose a fraud that accuracy certified.
- [ ] Train a real model. Split the data (80/20, shuffle with a fixed seed), fit sklearn's `LogisticRegression` on the training part, and predict on the test part. Compute all the metrics on the test predictions.
- [ ] Validate everything against sklearn. In `test_mymetrics.py`:

  ```python
  from sklearn import metrics
  assert np.isclose(accuracy(y_te, pred), metrics.accuracy_score(y_te, pred))
  assert np.array_equal(confusion_matrix(y_te, pred), metrics.confusion_matrix(y_te, pred))
  # ...same for precision_score, recall_score, f1_score
  ```

  Checkpoint: every assert passes. If a number disagrees with sklearn, sklearn is right: find the bug.
- [ ] Add the ROC curve. The classifier can output a *probability* (`model.predict_proba(X_te)[:, 1]`), and predicting "sick" means "probability above some threshold". The **ROC curve** plots true-positive rate (= recall) against false-positive rate (FP / (FP + TN)) as that threshold sweeps from 1 down to 0. Write `roc_curve(y_true, scores)`: sort by score descending, walk down the list moving the threshold past one prediction at a time, and record (FPR, TPR) at each step. Plot it with matplotlib. Checkpoint: the curve starts at (0,0), ends at (1,1), and bulges toward the top-left corner.
- [ ] Write `auc(fpr, tpr)`: the **AUC** (area under the ROC curve), computed with the trapezoid rule from the integration intuition in [lesson 09](../phase-1-math/09-calculus-and-gradient-descent.md). AUC = 1.0 is a perfect ranker, and 0.5 is coin-flipping. Checkpoint: the AUC is within 0.01 of `metrics.roc_auc_score(y_te, scores)`, and it is above 0.9 for this dataset.

<details><summary>Hints</summary>

- Every metric is just arithmetic on the four confusion-matrix cells. Write `confusion_matrix` first, then define precision/recall/F1 *in terms of it* so a fix in one place fixes all of them.
- Counting TP without a loop: `np.sum((y_true == 1) & (y_pred == 1))`. The `&` needs the parentheses.
- For the ROC walk: after sorting by score descending, each position i means "the top i predictions are called positive". TP at position i = number of 1-labels among the first i; FP = i minus that. `np.cumsum` on the sorted labels gives every TP count in one line.
- If the AUC is *exactly* 0.5 or the curve is a straight diagonal, the hard 0/1 predictions were probably passed in instead of the probabilities.

</details>

**Definition of done:** All asserts against sklearn pass. Explain in one sentence, out loud, why Dr. Useless gets 99% accuracy and 0 recall, and which metric to demand from a real medical test.

## Project 2 — my_kfold

**Goal:** One train/test split is one roll of the dice: a lucky split flatters a model, and an unlucky one slanders it. Build k-fold cross-validation from scratch and use it to run a fair kNN vs logistic regression match on Titanic.

**Milestones**

- [ ] Prepare Titanic exactly as in lesson 13 (numeric features, missing ages filled, `sex` encoded as 0/1). Put the prep in a function, because it is called several times.
- [ ] Write `my_kfold_indices(n, k, seed)`: shuffle the row indices `0..n-1` with a fixed seed, chop them into k nearly-equal folds, and return a list of `(train_idx, val_idx)` pairs. Each fold takes one turn as the validation set while the other k−1 folds are used for training. Checkpoint: with n=891 and k=5, every row appears in exactly one validation fold, and no `(train, val)` pair shares an index (`assert len(np.intersect1d(tr, va)) == 0`).
- [ ] Write `my_cross_val(model, X, y, k)`: for each fold, fit a *fresh clone* of the model on the training rows, score accuracy on the validation rows (use `mymetrics.accuracy` from Project 1), and return the k scores. Use `sklearn.base.clone(model)` so that folds do not contaminate each other.
- [ ] Run the match: `KNeighborsClassifier(n_neighbors=5)` vs `LogisticRegression(max_iter=1000)`, k=5, the same folds for both (that is what makes it fair). Print each model's scores as `mean ± std` (`np.mean`, `np.std`). Checkpoint: five scores per model, each roughly 0.6-0.85, and they *differ across folds*. That spread is exactly why one split cannot be trusted.
- [ ] Validate against sklearn, making sure both sides apply *identical* preprocessing. For logistic regression, compare against plain `cross_val_score(model, X, y, cv=5)`. For kNN, if the features are scaled inside each fold (as they should be; see Hints), the fair sklearn twin is one that also scales per fold: `cross_val_score(make_pipeline(StandardScaler(), KNeighborsClassifier(5)), X, y, cv=5)` (`make_pipeline` lives in `sklearn.pipeline`). Comparing scaled-per-fold kNN against unscaled kNN would be apples vs oranges. The folds from `my_kfold_indices` are shuffled differently from sklearn's, so the individual scores will not match, but the *means* should agree within about 0.03. Checkpoint: they do, for both models.
- [ ] Write a verdict in a comment at the top of the file: which model wins on Titanic, and is the gap bigger than the standard deviations? If the gap is smaller than the spread, the honest answer is "too close to call". Learning to say that is part of the skill.

<details><summary>Hints</summary>

- `np.array_split(shuffled_indices, k)` takes care of the "nearly equal folds when n isn't divisible by k" annoyance.
- Training indices for fold i = all shuffled indices *not* in fold i. `np.concatenate` the other folds, or use `np.setdiff1d`.
- kNN cares about feature scale (a fare of 512 dwarfs an age of 30). Scale features *inside* each fold: fit the scaler on the fold's training rows only. This ordering may feel fussy, but it matters: Project 3 shows what happens when it is done wrong.

</details>

**Definition of done:** The `my_cross_val` means agree with `cross_val_score` within ~0.03 on both models. Explain in one sentence why cross-validation beats a single split.

## Project 3 — Diagnosis clinic

**Goal:** Learn to read a learning curve the way a doctor reads an X-ray, spotting underfitting and overfitting at a glance. Then hunt down data leakage hiding in a plausible-looking pipeline.

**Milestones**

- [ ] Set up the clinic. Use Titanic. The two patients to examine are already familiar from lesson 11: `KNeighborsClassifier(n_neighbors=1)` (memorizes everything; suspected **overfitter**) and `KNeighborsClassifier(n_neighbors=150)` (averages over almost everyone; suspected **underfitter**).
- [ ] Write `learning_curve(model, X_train, y_train, X_val, y_val, sizes)`: for each size in `[50, 100, 200, 400, 600, all]`, fit a fresh clone on just the first `size` training rows, then record accuracy on (a) those same training rows and (b) the untouched validation set. Return both lists.
- [ ] Plot patient 1 (k=1): training accuracy and validation accuracy vs training size, on one chart, with a legend. Checkpoint: training accuracy is pinned at (or near) **1.0** for every size, while validation accuracy sits far below, leaving a big, persistent **gap**. That gap is **variance**: the model memorized noise that does not transfer. This is **overfitting**.
- [ ] Plot patient 2 (k=150). Checkpoint: *both* curves are low and close together, and adding more data barely helps. That is **bias**: the model is too simple (too smoothed-out) to capture the pattern, no matter how much data it gets. This is **underfitting**.
- [ ] Write the prescription in comments: more data helps a high-variance model but not a high-bias one; a high-bias model needs more capacity or better features. Verify one claim: fit k=1 with double the data and confirm the gap shrinks a little. Checkpoint: it does.
- [ ] Now hunt the leak. Here is a pipeline that looks reasonable but is subtly broken. Type it in and run it:

  ```python
  from sklearn.preprocessing import StandardScaler
  from sklearn.model_selection import train_test_split
  from sklearn.neighbors import KNeighborsClassifier

  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X)          # scale first...
  X_tr, X_te, y_tr, y_te = train_test_split(  # ...split second
      X_scaled, y, test_size=0.2, random_state=0)
  model = KNeighborsClassifier(5).fit(X_tr, y_tr)
  print(model.score(X_te, y_te))
  ```

  Find the sin before reading on. The scaler computed its mean and spread from **all** rows, including the test rows. The test set was supposed to simulate *future patients the model has never seen*, but their statistics leaked into training. That is **data leakage**.
- [ ] Fix it: split first, `fit_transform` the scaler on training rows only, and `transform` (*not* fit) the test rows. Checkpoint: on Titanic the score barely moves; this leak is real but mild. That is the dangerous part: leakage usually does not announce itself.
- [ ] See a loud leak, so that the smell is never forgotten. Add a poisoned feature (a noisy copy of the answer) and watch: `X_leaky = np.column_stack([X, y + rng.normal(0, 0.1, len(y))])`. Retrain with an honest split. Checkpoint: accuracy jumps to ~0.95+, absurdly better than anything from Project 2. In real projects this happens with features computed *after* the outcome was known (e.g. "days until patient discharged" when predicting illness). When a score looks too good to be true, hunt for the leak.

<details><summary>Hints</summary>

- For learning curves, shuffle the training rows once (fixed seed) *before* taking the first `size` rows, or the subsets will not be representative.
- Evaluating on the training rows is not cheating here: the train/validation *gap* is exactly what is being measured.
- Rule of thumb for the leak hunt: anything *learned from data* (scaler means, imputation values, feature selections) must be learned from the training rows only. `fit` on train, `transform` on both.

</details>

**Definition of done:** Two labeled learning-curve plots are ready to show someone and diagnose from, and the scaling leak is fixed. State the fit-on-train-only rule from memory.

## Stretch goals

- Add a **precision-recall curve** to `mymetrics.py` (precision vs recall as the threshold sweeps). It is often more informative than ROC when positives are as rare as in Project 1. Compare with sklearn's `precision_recall_curve`.
- Make `my_kfold` **stratified**: each fold keeps the same class ratio as the whole dataset. Check against sklearn's `StratifiedKFold`. Why does this matter a lot for the 1%-sick data and only a little for Titanic?
- Extend precision/recall/F1 to **multiclass** with macro-averaging (compute per class, then average). Validate with `f1_score(..., average="macro")` on a 3-class toy set.
- Wrap Project 1's threshold choice in a small script: given a cost for each FN and FP (a missed illness costs 100x a false alarm), find the threshold that minimizes total cost.

## Getting unstuck

- **A metric disagrees with sklearn:** print the four confusion-matrix cells from both (`metrics.confusion_matrix(y_true, y_pred)`) and compare cell by cell. The bug is almost always a flipped positive class or a swapped FP/FN.
- **Shapes misbehave:** check them with `print(y_true.shape, y_pred.shape)`. A `(n,1)` column vector meeting a `(n,)` array silently broadcasts into a huge `(n,n)` array, and every count goes wrong.
- **Fold scores look extreme (0.0 or 1.0):** print `len(train_idx), len(val_idx)` and the class balance of each fold. There may be an empty fold, or one with a single class.
- Standing advice: read the error message bottom-up, print shapes and a few actual values, ask an AI assistant for a **hint** rather than a solution, and type every line by hand. The fumbling is the learning.

## Resources

- **StatQuest: "ROC and AUC, Clearly Explained"** (search on YouTube) — the friendliest walkthrough of thresholds, TPR/FPR and the curve; watch it after building one in Project 1.
- **StatQuest: "Machine Learning Fundamentals: Bias and Variance"** (search on YouTube) — pairs well with the learning-curve plots from Project 3.
- **scikit-learn model evaluation guide** — <https://scikit-learn.org/stable/modules/model_evaluation.html> — the reference for every metric built in this lesson and dozens more that come up later; skim the precision/recall section.

## Skills unlocked

- [ ] I can explain why 99% accuracy can describe a worthless model, with a concrete example.
- [ ] I can draw a confusion matrix from memory and label TP, FP, FN, TN correctly.
- [ ] I can compute precision, recall and F1 by hand from a confusion matrix, and say which one a medical screening test should optimize.
- [ ] I can explain what an ROC curve shows and what AUC = 0.5 vs 0.9 means.
- [ ] I can implement k-fold cross-validation from scratch and explain why it beats a single split.
- [ ] I can look at a learning curve and diagnose underfitting vs overfitting from the gap.
- [ ] I can spot the fit-on-everything data-leakage bug in a pipeline and fix it.
- [ ] I have a tested `mymetrics.py` I will reuse in lessons 15, 16, 17 and beyond.

## Next up

Now that any model can be *judged* honestly, the next lesson builds a much smarter one and gives the new metrics library its first real workout: [15 · Decision Trees from Scratch](15-decision-trees.md).
