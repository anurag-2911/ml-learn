# 15 · Decision Trees from Scratch

**Phase 2 — Classical Machine Learning** · Estimated time: 1-2 weeks · Prerequisites: [11 · k-Nearest Neighbors](11-first-model-knn.md), [13 · Logistic Regression](13-logistic-regression.md), [14 · Model Evaluation](14-model-evaluation.md), [06 · Pandas](../phase-0-foundations/06-pandas.md)

> A decision tree is a machine learning model that people can read: a learned flowchart of if/else questions such as "is this passenger female? → is she in first class? → predict survived". This lesson writes the algorithm that *discovers* those questions from data. No formulas are handed down: the code measures how "mixed up" a group of labels is and greedily picks the question that unmixes it most. On the raw Titanic passenger list, the tree rediscovers "women and children first", and when it is allowed to grow too deep, it memorizes noise. Trees are also the building block of random forests and gradient boosting (next lesson), which still win most tabular-data competitions today.

## What this lesson builds

- **Project 1 — Split scorer:** a `gini(labels)` impurity function and a `best_split(X, y)` that scans every feature and threshold to find the single best yes/no question. Together they prove that the Titanic's best first question is about sex.
- **Project 2 — Tree grower:** a recursive `Node`-based tree builder with `max_depth`, a `predict` that walks the tree, and a printer that renders the learned rules as indented text; plus a depth-vs-accuracy experiment that makes overfitting visible.
- **Project 3 — sklearn trees:** a `DecisionTreeClassifier` tuned with cross-validation, a `plot_tree` diagram, and a feature-importance chart.

## Concepts covered

- **Decision tree**: a model that predicts by asking a sequence of learned if/else questions about the features.
- **Gini impurity**: a 0-to-0.5 score for how mixed the labels in a group are (0 = all the same class, 0.5 = a perfect 50/50 mix).
- **Entropy**: an alternative score for how mixed up a group is, from information theory. Either works; Gini is just cheaper to compute.
- **Best split**: the (feature, threshold) question that produces the largest drop in impurity, found by a brute-force search over the candidates.
- **Recursive tree building**: applying best-split to the data, then applying the same function to each half, until a stopping rule fires.
- **Stopping criteria**: max depth, minimum samples, or a pure node. These are the brakes that stop the tree from memorizing.
- **Overfitting and pruning**: deep trees memorize training noise. Limiting or cutting back the tree ("pruning") trades training accuracy for test accuracy.
- **Feature importance**: which features the tree's splits relied on most.
- **No feature scaling needed**: trees only compare values to thresholds, so standardizing the features changes nothing for them, unlike for the models in lessons 12 and 13.

## Before starting

1. Finish [lesson 14](14-model-evaluation.md) first. This lesson reuses cross-validation and the `train_test_split`/accuracy helpers from earlier lessons (or sklearn's).
2. Activate the venv at the repo root and check the packages (all installed in earlier lessons):

```bash
cd ~/ml/ml-learn
source .venv/bin/activate
python3 -c "import numpy, pandas, sklearn, matplotlib, seaborn; print('ok')"
```

If anything is missing (the output shows `ModuleNotFoundError` instead of `ok`), install the packages:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

3. Dataset: the Titanic passenger list. No account is needed, because seaborn downloads it the first time it is loaded:

```python
import seaborn as sns
df = sns.load_dataset("titanic")   # 891 rows; columns include survived, pclass, sex, age, fare, sibsp, parch
```

(If a `titanic.csv` was saved back in lesson 6, reusing it with `pd.read_csv` is fine too.)

4. Create the work folder:

```bash
mkdir -p work/15-decision-trees && cd work/15-decision-trees
```

## Project 1 — Split scorer

**Goal:** Write the two functions at the heart of every tree: `gini(labels)`, which scores how mixed a group is, and `best_split(X, y)`, which finds the question that unmixes it most. Then verify that the data itself says "split on sex first".

**Milestones**

- [ ] Prepare the data in `prepare_data.py`: load Titanic with seaborn, keep the columns `survived, pclass, sex, age, fare, sibsp, parch`, map `sex` to numbers (`male`→0, `female`→1), and drop rows with missing `age` (`df.dropna()`; NaN handling was covered in lesson 6). Split the data into a NumPy feature matrix `X` and a label vector `y = survived`. Checkpoint: `X.shape` is about `(714, 6)` and `y.mean()` is about `0.41` (41% of these passengers survived).
- [ ] Split into train/test (80/20, fixed `random_state`) using the lesson-14 helper or `sklearn.model_selection.train_test_split`. All fitting below uses only the training set.
- [ ] Write `gini(labels)`: for each class, compute its proportion `p`; return `1 - sum(p**2)`. Intuition: the chance that two randomly drawn passengers from this group have *different* labels. Checkpoint: `gini([0,0,0,0]) == 0.0`, `gini([0,0,1,1]) == 0.5`, `gini([0,0,0,1])` is `0.375`.
- [ ] Write `split_score(X_col, y, threshold)`: send samples with `X_col <= threshold` left, the rest right, and return the **weighted** Gini: `(n_left*gini_left + n_right*gini_right) / n`. Weighting matters: a pure group of 2 should not count as much as an impure group of 500. Checkpoint: a threshold that separates classes perfectly on a toy array scores `0.0`.
- [ ] Write `best_split(X, y)`: for every feature column, take the sorted unique values and try the midpoints between consecutive ones as thresholds; return the `(feature_index, threshold, score)` with the lowest weighted Gini. Skip "splits" that put everything on one side.
- [ ] Run it on the Titanic training set and print the winning feature's *name*. Checkpoint: the best first split is on `sex` (threshold 0.5), and the weighted Gini drops from about 0.48 at the root to about 0.33. Print the survival rate on each side: roughly 0.75 for women and 0.20 for men. The code has discovered "women and children first" from the data alone, with no history book.

<details><summary>Hints</summary>

- `gini` should work for any labels, not just 0/1. `np.unique(labels, return_counts=True)` gives the counts; divide them by `len(labels)` for proportions.
- Guard the empty case: `gini([])` on an empty side will divide by zero. Return 0.0 for an empty group (or skip such splits in the caller).
- Boolean masks from lesson 5 do the split in one line: `left = y[X[:, f] <= t]`, `right = y[X[:, f] > t]`.
- Brute force is fine here: 6 features × at most a few hundred thresholds is very little work. Get it correct first; speed later.
</details>

**Definition of done:** `gini` passes the three toy checkpoints, and `best_split` on Titanic training data picks `sex` with a printed impurity drop.

## Project 2 — Grow the tree

**Goal:** Turn `best_split` into a full classifier by calling it recursively (a function that builds a tree by building two smaller trees), then read the learned rules as text and measure how depth controls overfitting.

Recursion (a function that calls itself on smaller pieces of the problem) is the hard part of this lesson. The golden rule: **write the base cases first** (the conditions where the function does *not* call itself), and only then the recursive step. Take the milestones slowly and in order.

**Milestones**

- [ ] In `tree.py`, define a `Node` class (lesson 4 skills). A node is either a **leaf** carrying a `prediction`, or an **internal node** carrying `feature`, `threshold`, `left`, `right`. One class with optional attributes is fine; a `node.is_leaf()` method keeps later code clean.
- [ ] Write `make_leaf(y)`: returns a leaf Node predicting the majority class of `y`. Checkpoint: `make_leaf([1,1,0]).prediction == 1`.
- [ ] Write the skeleton of `build_tree(X, y, depth, max_depth)` with **base cases only**. It returns a leaf when: (1) all labels in `y` are the same, (2) `depth >= max_depth`, (3) there are fewer than `min_samples` rows (start with 2), or (4) `best_split` found no split that beats the current impurity. Temporarily make the "otherwise" branch also return a leaf. Checkpoint: the function runs on the full training set without recursing and predicts 0 (the majority died) for everything: a working but terrible depth-0 tree.
- [ ] Now add the recursive step: call `best_split`, partition rows into left (`<= threshold`) and right, and return an internal Node whose children are `build_tree(X_left, y_left, depth+1, max_depth)` and `build_tree(X_right, y_right, depth+1, max_depth)`. Checkpoint: `build_tree(..., max_depth=1)` returns a root splitting on `sex` with two leaves: women→survived, men→died.
- [ ] Write `predict_one(node, x)`: while the node is not a leaf, go left if `x[node.feature] <= node.threshold`, else right; return the leaf's prediction. Wrap it in `predict(node, X)` for a whole matrix. Checkpoint: the depth-1 tree scores about 0.78 accuracy on the test set, so one question already beats the first attempts in lesson 11.
- [ ] Write `print_tree(node, feature_names, indent=0)`: recursively print `"  " * indent` plus either `predict <class> (n=..., gini=...)` for leaves or `if <feature> <= <threshold>:` for internal nodes. Checkpoint: printing a depth-3 tree shows readable nested rules. The top line mentions `sex`, and inside the male branch there should be a split on `age`, the "children" half of the famous rule.
- [ ] The overfitting experiment: for `max_depth` in `[1, 2, 3, 5, 8, None]` (use e.g. 999 for None), train and record train and test accuracy; print a table and plot both curves against depth (lesson 7 skills). Checkpoint: train accuracy climbs toward ~0.95+ while test accuracy peaks around depth 3-5 (~0.78-0.82) and then flattens or *drops*. The gap between the curves is overfitting, drawn by the from-scratch tree.
- [ ] Look at the unlimited-depth tree with `print_tree`. Checkpoint: the printout shows absurd, hyper-specific rules (e.g. splits on `fare <= 26.1` inside splits on `fare <= 26.3`) that clearly memorize individual passengers. Cutting these branches off is the idea called **pruning**; `max_depth` is pruning-in-advance.

<details><summary>Hints</summary>

- `RecursionError: maximum recursion depth exceeded` means a base case is broken. The usual cause is a "split" that sends zero rows to one side, so the recursive call gets the same data forever. Print `len(y)` and `depth` at the top of `build_tree` and watch the sizes: they must shrink on every call.
- Trust the leap of faith: while writing the recursive step, *assume* `build_tree` already works for the smaller halves. The recursive step only wires together one level; the base cases handle the bottom.
- Debug on tiny data first: 8 hand-made rows, 2 features, `max_depth=2`. That is small enough to trace on paper against `print_tree`'s output.
- `predict_one` can be a loop (`while not node.is_leaf(): ...`) instead of recursion; a loop is often easier to reason about.
</details>

**Definition of done:** the tree trains on Titanic, `print_tree` shows human-readable rules starting with sex, and the depth experiment plot clearly shows the train/test gap growing with depth.

## Project 3 — sklearn trees

**Goal:** Use scikit-learn's production version of the from-scratch tree (the same algorithm, with more engineering), and tune it properly with the cross-validation built in lesson 14.

**Milestones**

- [ ] In `sklearn_tree.py`, fit `DecisionTreeClassifier(max_depth=3, random_state=0)` from `sklearn.tree` on the same train split. Checkpoint: test accuracy is within ~0.03 of the from-scratch depth-3 tree. Both implement the same idea.
- [ ] Visualize it: `from sklearn.tree import plot_tree`, then `plot_tree(clf, feature_names=..., class_names=["died","survived"], filled=True)` inside a matplotlib figure; save it as a PNG. Checkpoint: the diagram's root box says `sex <= 0.5` with gini ≈ 0.48, matching the Project 1 numbers.
- [ ] Tune depth with cross-validation (lesson 14): for `max_depth` in 1-15, run `cross_val_score(clf, X_train, y_train, cv=5)` and plot mean CV accuracy vs depth. Checkpoint: the curve rises, peaks around depth 3-6, then declines or plateaus. Pick the best depth from *this* curve, not from test accuracy (the test set stays untouched until the end).
- [ ] Retrain at the chosen depth on the full training set, evaluate **once** on the test set, and report accuracy plus the confusion matrix from the lesson-14 metrics library. Checkpoint: test accuracy is around 0.78-0.83.
- [ ] Inspect `clf.feature_importances_` (each feature's total share of impurity reduction, summing to 1) and draw a bar chart. Checkpoint: `sex` is the most important feature, with `fare`, `age` and `pclass` behind it.
- [ ] The no-scaling experiment: standardize `X` with `StandardScaler` (lesson 12 made this *necessary*), retrain the tree, and compare predictions. Checkpoint: the predictions are identical. A threshold question like `age <= 6.5` just becomes `scaled_age <= -1.6`, and the tree does not care. Write two sentences in a comment on why scaling mattered for gradient-descent models but not here.

<details><summary>Hints</summary>

- `plot_tree` needs a big figure to be readable: `plt.figure(figsize=(16, 8))` before the call, then `plt.savefig("tree.png", dpi=150)`.
- CV numbers on ~570 training rows will wobble by a couple of points between depths. That is normal: look at the shape of the curve, not at tiny differences.
- sklearn's tree may pick slightly different thresholds than the from-scratch tree at deeper levels (tie-breaking differs). The same top split with small downstream differences is not a bug.
</details>

**Definition of done:** the project has produced `tree.png`, a CV-vs-depth plot, a feature-importance chart, and one final, honest test accuracy from a depth chosen by cross-validation only.

## Stretch goals

- **Entropy option:** add `entropy(labels) = -sum(p * log2(p))` and a `criterion="gini"|"entropy"` switch to the from-scratch tree, then compare the trees they grow (usually nearly identical).
- **More stopping knobs:** implement `min_samples_leaf` (reject splits that create a leaf smaller than N) and redo the depth experiment. Does it tame the unlimited-depth tree?
- **Real pruning:** explore sklearn's cost-complexity pruning via the `ccp_alpha` parameter (see the scikit-learn tree docs below) and plot test accuracy vs alpha.
- **Regression tree:** swap Gini for variance of the target and majority-vote leaves for mean-value leaves, and predict passenger `fare`. The same recursion then becomes a regressor.

## Getting unstuck

- **Infinite recursion / `RecursionError`:** print `depth` and `len(y)` at the top of `build_tree`. If `len(y)` ever stops shrinking, `best_split` is returning a split with an empty side. Forbid such splits.
- **Weird best splits:** unit-test `gini` on toy lists first; one wrong impurity poisons every score above it. Then test `best_split` on a 6-row array where the answer is obvious by eye.
- **Accuracy far below ~0.75:** check that the model was fit on the training set only, that `sex` was encoded as numbers (a string column silently breaks `<=` comparisons), and that `predict_one` uses `<=` on the same side as the builder did.
- Standing advice: read error tracebacks bottom-up; print shapes and intermediate values instead of guessing; when stuck for over 30 minutes, ask an AI assistant for a **hint** ("what's a likely cause of X?"), never for the finished code; and type every line by hand, because that is where the learning happens.

## Resources

- **StatQuest: Decision and Classification Trees** — the clearest visual walkthrough of Gini and splitting; search YouTube for "StatQuest Decision and Classification Trees" and watch it before or during Project 1.
- **scikit-learn tree documentation** — https://scikit-learn.org/stable/modules/tree.html — how the library version works, its parameters, pruning (`ccp_alpha`), and honest notes on trees' weaknesses. Read after Project 2, skim before Project 3.

## Skills unlocked

- [ ] I can explain a decision tree as a learned flowchart and read one's rules aloud from a printout.
- [ ] I can compute Gini impurity by hand for a small group of labels and explain what 0.0 and 0.5 mean.
- [ ] I can find the best split of a dataset by scanning features and thresholds, and justify the weighting in the score.
- [ ] I can write a recursive function with explicit base cases and debug it when the recursion does not terminate.
- [ ] I can demonstrate overfitting with a train-vs-test curve and pick a model size using cross-validation, not the test set.
- [ ] I can use `DecisionTreeClassifier`, visualize it with `plot_tree`, and interpret `feature_importances_`.
- [ ] I can explain why trees need no feature scaling while gradient-descent models do.

## Next up

One tree overfits, but a *crowd* of imperfect trees that vote together is one of the strongest ideas in machine learning, and the next lesson builds such crowds: [16 · Ensembles: Random Forests and Gradient Boosting](16-ensembles-random-forest-boosting.md).
