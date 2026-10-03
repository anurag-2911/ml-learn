# 11 · Your First ML Model: k-Nearest Neighbors from Scratch

**Phase 2 — Classical Machine Learning** · Estimated time: 1 week · Prerequisites: [05 · NumPy](../phase-0-foundations/05-numpy.md), [07 · Data Visualization](../phase-0-foundations/07-data-visualization.md), [08 · Linear Algebra by Code](../phase-1-math/08-linear-algebra-by-code.md)

> This lesson builds a first real machine-learning model, with no libraries doing the thinking. The algorithm is k-Nearest Neighbors (kNN), and it is simple: to classify a new thing, look at the k most similar things already seen and copy their majority vote. Along the way, the lesson introduces the core vocabulary of the entire field (features, labels, training, testing, accuracy, overfitting), and every later lesson builds on these words. The finished classifier, written entirely by hand, identifies flowers with ~95% accuracy, and overfitting is discovered first-hand in an experiment rather than read about.

## What this lesson builds

- **Project 1 — kNN from scratch on Iris:** a Python file `knn.py` with a hand-written train/test split and a `predict` function built on NumPy distances, reaching ~95% accuracy on flowers it has never seen.
- **Project 2 — The k experiment:** a script and a plot (`k_experiment.png`) showing train vs test accuracy for k = 1..25, plus a 3-sentence written explanation of overfitting in the learner's own words.
- **Project 3 — Same thing in 4 lines:** a script using scikit-learn's `KNeighborsClassifier` that reproduces the from-scratch results and introduces the fit/predict/score pattern used by every sklearn model.

## Concepts covered

This lesson is the vocabulary foundation for everything that follows, so here is a first ML glossary. Skim it now, then watch each term become concrete inside the projects.

| Term | Plain-words definition |
|---|---|
| **Learning from data** | Instead of hand-written rules ("if petal > 2cm then..."), the computer gets examples and figures out the pattern itself. |
| **Example (sample)** | One thing in the dataset (here, one measured flower). |
| **Feature** | One measurable property of an example. A flower's petal length is a feature. |
| **Label** | The answer the model should predict (here, the flower's species). |
| **Dataset as a matrix** | All examples stacked into a 2D array `X` (one row per example, one column per feature) plus a vector `y` of labels. |
| **Training set** | The examples the model is allowed to learn from. |
| **Test set** | Examples held back and hidden from the model, used only to measure how well it does on *new* data. |
| **Euclidean distance** | The straight-line distance between two points (here, between two flowers in 4-dimensional feature space). Small distance = similar flowers. |
| **kNN** | Classify a new example by taking a majority vote among its k closest training examples. |
| **Accuracy** | The fraction of predictions that were correct. 27 right out of 30 = 0.9. |
| **Overfitting** | The model memorizes quirks of the training data and does worse on new data. |
| **Underfitting** | The model is too crude to capture the pattern at all, so it does badly everywhere. |

## Before starting

- This lesson assumes the NumPy lesson is finished: indexing arrays, using `np.sqrt`, `np.sum` and `np.argsort`, and knowing what `axis=` means.
- Making a basic matplotlib line plot is also assumed ([lesson 07](../phase-0-foundations/07-data-visualization.md)).
- Activate the venv and install this week's tools (scikit-learn is the standard Python ML library; the lesson builds kNN by hand first, then uses sklearn's version in Project 3):

```bash
cd ~/ml/ml-learn
source .venv/bin/activate
pip install numpy matplotlib scikit-learn
mkdir -p work/11-first-model-knn
cd work/11-first-model-knn
```

- No dataset download is needed: the Iris dataset (150 real flowers measured in 1936, 4 measurements each, 3 species) ships inside scikit-learn. Verify the install:

```bash
python3 -c "from sklearn.datasets import load_iris; print(load_iris().data.shape)"
```

Checkpoint: it prints `(150, 4)`, meaning 150 examples and 4 features. That shape *is* the "dataset as a matrix" idea from the glossary.

## Project 1 — kNN from scratch on Iris

**Goal:** Build a working classifier from nothing but NumPy: split the data honestly, measure distances, vote, and report accuracy. Once this runs, it is real machine learning.

**Milestones**

- [ ] Create `knn.py`. Load the data and explore it until it feels real:

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

  Checkpoint: say out loud what row 0 means ("a flower with sepal length 5.1 cm ... labeled setosa").
- [ ] Write a train/test split by hand. Shuffle the row indices with `np.random.permutation(150)` (seed it first with `np.random.seed(42)` so that runs are repeatable), then take the first 120 shuffled indices as training and the last 30 as test. Build `X_train, y_train, X_test, y_test` by fancy-indexing. Why shuffle? The iris rows are sorted by species, so an unshuffled split would train on two species and test on the third. Checkpoint: `X_train.shape == (120, 4)` and `set(y_test)` contains all three labels.
- [ ] Write `euclidean(a, b)`: the straight-line distance between two 4-feature flowers, `sqrt(sum((a - b) ** 2))`. Each flower is a point in 4-dimensional space; the formula is exactly the Pythagoras coded in [lesson 08](../phase-1-math/08-linear-algebra-by-code.md), just with 4 coordinates. Checkpoint: `euclidean(X[0], X[1])` is about `0.5385`.
- [ ] Write `predict_one(x, X_train, y_train, k)`: compute the distance from `x` to *every* training flower (a loop is fine; there is a loop-free NumPy way in the hints), find the indices of the k smallest distances with `np.argsort`, and take the majority vote of their labels with `np.bincount(...).argmax()`. Checkpoint: `predict_one(X_test[0], X_train, y_train, k=5)` equals `y_test[0]`.
- [ ] Notice what just happened: there was no "training" step. kNN learns by *storing* the training set, and all the work happens at prediction time. Most later models are the opposite.
- [ ] Predict every test flower, compare with `y_test`, and compute accuracy, the fraction of correct predictions: `np.mean(predictions == y_test)`. Checkpoint: accuracy is 0.90 or higher with k=5, typically around 0.95–0.97.
- [ ] Print the flowers the model got wrong (their features, true label and predicted label). Wrong answers are where the learning is: are they borderline flowers between two similar species?

<details><summary>Hints</summary>

- Distances to *all* training rows without a loop: `np.sqrt(np.sum((X_train - x) ** 2, axis=1))`. Broadcasting subtracts `x` from every row at once, and `axis=1` sums across each row's 4 features. Result shape: `(120,)`.
- `np.argsort(distances)[:k]` gives the *indices* of the k nearest flowers; feed those indices into `y_train` to get their labels.
- `np.bincount(labels).argmax()` returns the most common label. Use an odd `k` so that votes cannot tie 3 ways as easily.
- If accuracy is suspiciously bad (~0.33), the data was probably not shuffled, or `y` was indexed with different indices than `X`. Train rows and their labels must stay paired.

</details>

**Definition of done:** `python3 knn.py` prints a test accuracy ≥ 0.90 from a model whose every line was written by hand and can be explained.

## Project 2 — The k experiment

**Goal:** Discover overfitting first-hand by watching what happens as k changes. This experiment is the single most important idea in the lesson.

**Milestones**

- [ ] Create a new file, `k_experiment.py`, that reuses the Project 1 functions (import them or copy them in). Write `accuracy(X_eval, y_eval, X_train, y_train, k)` that predicts every row of `X_eval` and returns the fraction correct.
- [ ] For every k from 1 to 25, compute two numbers: accuracy on the **test set** and accuracy on the **training set itself** (i.e. predict each training flower using the training data). Store both lists.
- [ ] Checkpoint before plotting: at k=1, training accuracy is exactly 1.0. Make sure the reason is clear before moving on: each training flower's nearest neighbor *is itself*, at distance 0. The model is a pure memorizer, and a memorizer gets full marks on questions it has already seen.
- [ ] Plot both curves on one chart: x-axis k, y-axis accuracy, one line per set, with a legend and axis labels (skills from [lesson 07](../phase-0-foundations/07-data-visualization.md)). Save it as `k_experiment.png`. Checkpoint: the train curve starts at 1.0 and drifts down; the test curve starts *below* the train curve, is best somewhere in the middle (roughly k=3–15), and sags at large k.
- [ ] Read the gap. Where train accuracy is high but test accuracy is lower (small k), the model has memorized training quirks that do not generalize: that is **overfitting**. Where both are mediocre (large k; imagine k=120, where every prediction is just the overall majority vote), the model is too crude: that is **underfitting**. The best k balances the two.
- [ ] In a file `overfitting.md`, write 3 original sentences: why k=1 is perfect on training data, why that perfection does not carry to test data, and why testing on training data is therefore a lie. This is why a model must never be tested on its training data: such a test measures memory, not learning.

<details><summary>Hints</summary>

- Computing train accuracy: evaluate on `X_train`/`y_train` while also using `X_train`/`y_train` as the neighbors. Each point does find itself; that is the point of the exercise.
- With only 30 test flowers, each one is worth 3.3% accuracy, so the test curve moves in chunky steps. That is normal for small test sets.
- If both curves are flat at 1.0, the code is accidentally evaluating on training data both times. Check which arrays are passed in.

</details>

**Definition of done:** `k_experiment.png` shows the two curves with the k=1 gap visible, and `overfitting.md` explains it in 3 sentences.

## Project 3 — Same thing in 4 lines

**Goal:** Meet scikit-learn, the library used for all of Phase 2, and confirm that its kNN agrees with the from-scratch version. Every sklearn model uses the same three-verb pattern: `fit` (learn from training data), `predict` (answer for new data), `score` (accuracy in one call).

**Milestones**

- [ ] Skim the [scikit-learn getting started guide](https://scikit-learn.org/stable/getting_started.html), just the "Fitting and predicting" section (5 minutes).
- [ ] Create a new file, `sklearn_knn.py`. Using *the same* `X_train, y_train, X_test, y_test` as in Project 1 (same seed), build the model:

  ```python
  from sklearn.neighbors import KNeighborsClassifier

  model = KNeighborsClassifier(n_neighbors=5)
  model.fit(X_train, y_train)                # "learn" (for kNN: just store)
  print(model.score(X_test, y_test))         # accuracy
  ```

  Checkpoint: the score matches the Project 1 accuracy for k=5, exactly or within one flower (~0.03).
- [ ] Get per-flower predictions with `model.predict(X_test)` and compare them element-wise with the from-scratch predictions: `np.mean(sklearn_preds == my_preds)`. Checkpoint: 1.0, or extremely close (see the hints if a flower or two disagree).
- [ ] Also try sklearn's official splitter, `train_test_split` from `sklearn.model_selection`. It is the standard way to split data from now on. Note that it shuffles the data automatically.
- [ ] Say the pattern out loud until it sticks: **construct → fit → predict/score.** In sklearn, linear regression, decision trees and random forests are all these same three lines with a different first line.

<details><summary>Hints</summary>

- A single disagreement usually means a *tie* in the vote (e.g. 2-2-1 among 5 neighbors, or two neighbors at identical distance). The from-scratch tie-break and sklearn's may differ. Both answers are defensible, and the difference is worth understanding, not fixing.
- Make sure both implementations see identical splits: same seed, same indices. Different splits make comparison meaningless.

</details>

**Definition of done:** sklearn's accuracy matches the from-scratch model's, any disagreement can be explained, and the fit/predict/score pattern can be written from memory.

## Stretch goals

- **Feature scaling teaser:** multiply one feature column by 100 (pretend it was measured in different units) and rerun. Accuracy drops, because the big feature dominates the distance. Fix it by standardizing each column (subtract mean, divide by standard deviation). This is why preprocessing exists ([lesson 19](19-feature-engineering-pipelines.md) covers it in depth).
- **Decision-boundary picture:** using only 2 features (petal length and width), classify every point on a fine grid and color the plane by predicted class with `plt.contourf`. Compare the map for k=1 (jagged islands) with k=15 (smooth regions). The difference is overfitting made visible.
- **Try another distance:** Manhattan distance (`sum(|a - b|)`) instead of Euclidean. Does the best k change? Does accuracy?
- **Harder dataset:** rerun everything on `sklearn.datasets.load_wine` (13 features, 3 classes). Does scaling matter more there?

## Getting unstuck

- **Shapes first, always.** Print `X_train.shape`, `distances.shape` and `predictions.shape`. In this lesson almost every bug is a shape bug: distances should be `(120,)`, not `(120, 4)`. A result of `(120, 4)` means the `axis=` is wrong or missing.
- **Rows and labels must travel together.** If accuracy is near 0.33 (random guessing among 3 classes), `X` and `y` were shuffled differently. Use one array of shuffled indices to slice both.
- **Test one prediction before a hundred.** Get `predict_one` right on a single flower that can be checked by hand before looping over the test set.
- Standing advice: read error messages bottom-up (the last line names the actual problem), print intermediate values liberally, ask an AI assistant for a *hint* (not a solution), and type every line of code by hand. Muscle memory is part of the curriculum.

## Resources

- **StatQuest: K-nearest neighbors** — a short, friendly visual explanation of exactly what this lesson builds; search for it on YouTube (channel: StatQuest with Josh Starmer). Watch after Project 1, not before.
- **scikit-learn getting started** — https://scikit-learn.org/stable/getting_started.html — the fit/predict/score pattern from the source; skim for Project 3.
- **KNeighborsClassifier docs** — on scikit-learn.org, search "KNeighborsClassifier" to see every option the 4-line version accepts (weights, metrics, algorithms).

## Skills unlocked

- [ ] I can explain features, labels, examples, and why a dataset is a matrix `X` plus a label vector `y`.
- [ ] I can explain why a model must never be measured on its own training data.
- [ ] I can write a seeded, shuffled train/test split with NumPy.
- [ ] I can compute Euclidean distances between a point and a whole matrix of points, without a loop.
- [ ] I implemented kNN from scratch and reached ≥ 0.90 accuracy on held-out data.
- [ ] I can describe overfitting vs underfitting using the k-experiment plot as evidence.
- [ ] I can train and evaluate a scikit-learn model with fit/predict/score.

## Next up

The next lesson moves from *classifying* to *predicting numbers*, and this time the model genuinely learns parameters from data: [12 · Linear Regression from Scratch](12-linear-regression.md).
