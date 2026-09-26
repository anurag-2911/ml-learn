# 11 · Your First ML Model: k-Nearest Neighbors from Scratch

**Phase 2 — Classical Machine Learning** · Estimated time: 1 week · Prerequisites: [05 · NumPy](../phase-0-foundations/05-numpy.md), [07 · Data Visualization](../phase-0-foundations/07-data-visualization.md), [08 · Linear Algebra by Code](../phase-1-math/08-linear-algebra-by-code.md)

> This is the week you build your first real machine-learning model — no libraries doing the thinking for you. The algorithm is k-Nearest Neighbors (kNN), and it is beautifully simple: to classify a new thing, look at the k most similar things you have already seen and copy their majority vote. Along the way you will pick up the core vocabulary of the entire field — features, labels, training, testing, accuracy, overfitting — and every later lesson stands on these words. By the end you will have a classifier you wrote yourself that identifies flowers with ~95% accuracy, and you will have personally discovered overfitting instead of reading about it.

## What you will build

- **Project 1 — kNN from scratch on Iris:** a Python file `knn.py` with your own train/test split and a `predict` function built on NumPy distances, hitting ~95% accuracy on flowers it has never seen.
- **Project 2 — The k experiment:** a script and a plot (`k_experiment.png`) showing train vs test accuracy for k = 1..25, plus a 3-sentence written explanation of overfitting in your own words.
- **Project 3 — Same thing in 4 lines:** a script using scikit-learn's `KNeighborsClassifier` that reproduces your from-scratch results and teaches you the fit/predict/score pattern used by every sklearn model.

## Concepts you will learn by doing

This lesson is the vocabulary foundation for everything that follows, so here is your first ML glossary. Skim it now, then watch each term become concrete inside the projects.

| Term | Plain-words definition |
|---|---|
| **Learning from data** | Instead of writing rules by hand ("if petal > 2cm then..."), you give the computer examples and it figures out the pattern itself. |
| **Example (sample)** | One thing in your dataset — here, one measured flower. |
| **Feature** | One measurable property of an example — a flower's petal length is a feature. |
| **Label** | The answer you want the model to predict — this flower's species. |
| **Dataset as a matrix** | All examples stacked into a 2D array `X` (one row per example, one column per feature) plus a vector `y` of labels. |
| **Training set** | The examples the model is allowed to learn from. |
| **Test set** | Examples held back and hidden from the model, used only to measure how well it does on *new* data. |
| **Euclidean distance** | The straight-line distance between two points — here, between two flowers in 4-dimensional feature space. Small distance = similar flowers. |
| **kNN** | Classify a new example by taking a majority vote among its k closest training examples. |
| **Accuracy** | The fraction of predictions that were correct. 27 right out of 30 = 0.9. |
| **Overfitting** | The model memorizes quirks of the training data and does worse on new data. |
| **Underfitting** | The model is too crude to capture the pattern at all, so it does badly everywhere. |

## Before you start

- You have finished the NumPy lesson: you can index arrays, use `np.sqrt`, `np.sum`, `np.argsort`, and you know what `axis=` means.
- You can make a basic matplotlib line plot ([lesson 07](../phase-0-foundations/07-data-visualization.md)).
- Activate your venv and install this week's tools (scikit-learn is the standard Python ML library — you will hand-roll kNN first, then use sklearn's version in Project 3):

```bash
cd ~/ml/ml-learn
source .venv/bin/activate
pip install numpy matplotlib scikit-learn
mkdir -p work/11-first-model-knn
cd work/11-first-model-knn
```

- No dataset download needed: the Iris dataset (150 real flowers measured in 1936, 4 measurements each, 3 species) ships inside scikit-learn. Verify your install:

```bash
python3 -c "from sklearn.datasets import load_iris; print(load_iris().data.shape)"
```

Checkpoint: it prints `(150, 4)` — 150 examples, 4 features. That shape *is* the "dataset as a matrix" idea from the glossary.

## Project 1 — kNN from scratch on Iris

**Goal:** Build a working classifier from nothing but NumPy: split the data honestly, measure distances, vote, and report accuracy. When this runs, you have done machine learning.

**Milestones**

- [ ] Create `knn.py`. Load the data and poke at it until it feels real:

  ```python
  import numpy as np
  from sklearn.datasets import load_iris

  iris = load_iris()
  X = iris.data      # shape (150, 4): rows = flowers, columns = features
  y = iris.target    # shape (150,): labels 0, 1, 2
  print(iris.feature_names)   # what the 4 columns mean
  print(iris.target_names)    # what the 3 labels mean
  print(X[0], y[0])           # one example: 4 features and its label
  ```

  Checkpoint: you can say out loud what row 0 means — "a flower with sepal length 5.1 cm ... labeled setosa".
- [ ] Write your own train/test split. Shuffle the row indices with `np.random.permutation(150)` (seed it first with `np.random.seed(42)` so runs are repeatable), take the first 120 shuffled indices as training, the last 30 as test. Build `X_train, y_train, X_test, y_test` by fancy-indexing. Why shuffle? The iris rows are sorted by species — an unshuffled split would train on two species and test on the third. Checkpoint: `X_train.shape == (120, 4)` and `set(y_test)` contains all three labels.
- [ ] Write `euclidean(a, b)`: the straight-line distance between two 4-feature flowers, `sqrt(sum((a - b) ** 2))`. Each flower is a point in 4-dimensional space; the formula is exactly the Pythagoras you coded in [lesson 08](../phase-1-math/08-linear-algebra-by-code.md), just with 4 coordinates. Checkpoint: `euclidean(X[0], X[1])` is about `0.5385`.
- [ ] Write `predict_one(x, X_train, y_train, k)`: compute the distance from `x` to *every* training flower (a loop is fine; there is a loop-free NumPy way in the hints), find the indices of the k smallest distances with `np.argsort`, take the majority vote of their labels with `np.bincount(...).argmax()`. Checkpoint: `predict_one(X_test[0], X_train, y_train, k=5)` equals `y_test[0]`.
- [ ] Notice what just happened: there was no "training" step. kNN learns by *storing* the training set — all the work happens at prediction time. Most later models are the opposite.
- [ ] Predict every test flower, compare with `y_test`, and compute accuracy — the fraction you got right: `np.mean(predictions == y_test)`. Checkpoint: accuracy is 0.90 or higher with k=5, typically around 0.95–0.97.
- [ ] Print the flowers you got wrong (their features, true label, predicted label). Wrong answers are where the learning is: are they borderline flowers between two similar species?

<details><summary>Hints</summary>

- Distances to *all* training rows without a loop: `np.sqrt(np.sum((X_train - x) ** 2, axis=1))` — broadcasting subtracts `x` from every row at once, and `axis=1` sums across each row's 4 features. Result shape: `(120,)`.
- `np.argsort(distances)[:k]` gives the *indices* of the k nearest flowers; feed those indices into `y_train` to get their labels.
- `np.bincount(labels).argmax()` returns the most common label. Use an odd `k` so votes can't tie 3 ways as easily.
- If accuracy is suspiciously terrible (~0.33), you probably forgot to shuffle, or you indexed `y` with different indices than `X` — train rows and their labels must stay paired.

</details>

**Definition of done:** `python3 knn.py` prints a test accuracy ≥ 0.90 from a model whose every line you wrote and can explain.

## Project 2 — The k experiment

**Goal:** Discover overfitting yourself by watching what happens as k changes — this experiment is the single most important idea in the lesson.

**Milestones**

- [ ] New file `k_experiment.py`, reusing your Project 1 functions (import them or copy them in). Write `accuracy(X_eval, y_eval, X_train, y_train, k)` that predicts every row of `X_eval` and returns the fraction correct.
- [ ] For every k from 1 to 25, compute two numbers: accuracy on the **test set** and accuracy on the **training set itself** (i.e. predict each training flower using the training data). Store both lists.
- [ ] Checkpoint before plotting: at k=1, training accuracy is exactly 1.0. Make sure you see why before moving on: each training flower's nearest neighbor *is itself*, at distance 0 — the model is a pure memorizer, and a memorizer aces questions it has already seen.
- [ ] Plot both curves on one chart — x-axis k, y-axis accuracy, one line per set, with a legend and axis labels ([lesson 07](../phase-0-foundations/07-data-visualization.md) skills). Save it as `k_experiment.png`. Checkpoint: the train curve starts at 1.0 and drifts down; the test curve starts *below* the train curve, is best somewhere in the middle (roughly k=3–15), and sags at large k.
- [ ] Read the gap. Where train accuracy is high but test accuracy is lower (small k), the model has memorized training quirks that don't generalize — **overfitting**. Where both are mediocre (large k — imagine k=120: every prediction is just the overall majority vote), the model is too crude — **underfitting**. The best k balances the two.
- [ ] In a file `overfitting.md`, write 3 sentences in your own words: why k=1 is perfect on training data, why that perfection does not carry to test data, and why testing on training data is therefore a lie. This is why you must never test on training data — it tells you about memory, not learning.

<details><summary>Hints</summary>

- Computing train accuracy: evaluate on `X_train`/`y_train` while also using `X_train`/`y_train` as the neighbors. Yes, each point finds itself — that is the point of the exercise.
- With only 30 test flowers, each one is worth 3.3% accuracy, so the test curve moves in chunky steps. That is normal for small test sets.
- If both curves are flat at 1.0, you are accidentally evaluating on training data both times — check which arrays you pass in.

</details>

**Definition of done:** `k_experiment.png` shows the two curves with the k=1 gap visible, and `overfitting.md` explains it in 3 sentences.

## Project 3 — Same thing in 4 lines

**Goal:** Meet scikit-learn, the library you will use for all of Phase 2, and confirm that its kNN agrees with yours. Every sklearn model uses the same three-verb pattern: `fit` (learn from training data), `predict` (answer for new data), `score` (accuracy in one call).

**Milestones**

- [ ] Skim the [scikit-learn getting started guide](https://scikit-learn.org/stable/getting_started.html) — just the "Fitting and predicting" section, 5 minutes.
- [ ] New file `sklearn_knn.py`. Using *your same* `X_train, y_train, X_test, y_test` (same seed!), build the model:

  ```python
  from sklearn.neighbors import KNeighborsClassifier

  model = KNeighborsClassifier(n_neighbors=5)
  model.fit(X_train, y_train)                # "learn" (for kNN: just store)
  print(model.score(X_test, y_test))         # accuracy
  ```

  Checkpoint: the score matches your Project 1 accuracy for k=5, exactly or within one flower (~0.03).
- [ ] Get per-flower predictions with `model.predict(X_test)` and compare element-wise with your own predictions: `np.mean(sklearn_preds == my_preds)`. Checkpoint: 1.0, or extremely close — see hints if a flower or two disagree.
- [ ] Also try sklearn's official splitter, `train_test_split` from `sklearn.model_selection` — the standard way you will split from now on. Note it shuffles for you.
- [ ] Say the pattern out loud until it sticks: **construct → fit → predict/score.** Linear regression, decision trees, random forests — in sklearn they are all these same three lines with a different first line.

<details><summary>Hints</summary>

- A single disagreement usually means a *tie* in the vote (e.g. 2-2-1 among 5 neighbors, or two neighbors at identical distance). Your tie-break and sklearn's may differ. Both answers are defensible — that is worth understanding, not fixing.
- Make sure both implementations see identical splits: same seed, same indices. Different splits make comparison meaningless.

</details>

**Definition of done:** sklearn's accuracy matches yours, you can explain any disagreement, and you can write the fit/predict/score pattern from memory.

## Stretch goals

- **Feature scaling teaser:** multiply one feature column by 100 (pretend it was measured in different units) and rerun. Accuracy drops — distance is dominated by the big feature. Fix it by standardizing each column (subtract mean, divide by standard deviation). This is why preprocessing exists ([lesson 19](19-feature-engineering-pipelines.md) goes deep).
- **Decision-boundary picture:** using only 2 features (petal length and width), classify every point on a fine grid and color the plane by predicted class with `plt.contourf`. Compare the map for k=1 (jagged islands) vs k=15 (smooth regions) — overfitting made visible.
- **Try another distance:** Manhattan distance (`sum(|a - b|)`) instead of Euclidean. Does the best k change? Does accuracy?
- **Harder dataset:** rerun everything on `sklearn.datasets.load_wine` (13 features, 3 classes). Does scaling matter more there?

## If you get stuck

- **Shapes first, always.** Print `X_train.shape`, `distances.shape`, `predictions.shape`. In this lesson almost every bug is a shape bug: distances should be `(120,)`, not `(120, 4)` — if you got the second, your `axis=` is wrong or missing.
- **Rows and labels must travel together.** If accuracy is near 0.33 (random guessing among 3 classes), you have shuffled `X` and `y` differently. Use one array of shuffled indices to slice both.
- **Test one prediction before a hundred.** Get `predict_one` right on a single flower you can check by hand before looping over the test set.
- Standing advice: read error messages bottom-up (the last line names the actual problem), print intermediate values liberally, ask an AI assistant for a *hint* — not a solution — and type every line of code yourself. Muscle memory is part of the curriculum.

## Resources

- **StatQuest: K-nearest neighbors** — a short, friendly visual explanation of exactly what you are building; search for it on YouTube (channel: StatQuest with Josh Starmer). Watch after Project 1, not before.
- **scikit-learn getting started** — https://scikit-learn.org/stable/getting_started.html — the fit/predict/score pattern from the source; skim for Project 3.
- **KNeighborsClassifier docs** — on scikit-learn.org, search "KNeighborsClassifier" to see every option your 4-line version accepts (weights, metrics, algorithms).

## Skills unlocked

- [ ] I can explain features, labels, examples, and why a dataset is a matrix `X` plus a label vector `y`.
- [ ] I can explain why you must never measure a model on its own training data.
- [ ] I can write a seeded, shuffled train/test split with NumPy.
- [ ] I can compute Euclidean distances between a point and a whole matrix of points, without a loop.
- [ ] I implemented kNN from scratch and reached ≥ 0.90 accuracy on held-out data.
- [ ] I can describe overfitting vs underfitting using my k-experiment plot as evidence.
- [ ] I can train and evaluate a scikit-learn model with fit/predict/score.

## Next up

Now that you can *classify*, learn to *predict numbers* — and this time the model genuinely learns parameters from data: [12 · Linear Regression from Scratch](12-linear-regression.md).
