# 13 · Logistic Regression from Scratch

**Phase 2 — Classical ML** · Estimated time: 1 week · Prerequisites: [12 · Linear Regression from Scratch](12-linear-regression.md), [09 · Calculus You Can Run: Gradient Descent](../phase-1-math/09-calculus-and-gradient-descent.md), [06 · Pandas: Interrogating Real Datasets](../phase-0-foundations/06-pandas.md)

> Last lesson your model predicted *numbers*. But most interesting questions are yes/no: will this passenger survive? Is this email spam? Will this customer leave? This week you bend your linear regression into a **classifier** — a model that answers yes/no questions with a probability attached. You will build it from scratch with just NumPy, watch it draw a literal line between two clouds of points, then use it to predict who survived the Titanic. Along the way you meet two ideas that power every neural network you will ever build: the sigmoid function and cross-entropy loss.

## What you will build

- **Project 1 — 2D playground:** a from-scratch logistic regression (sigmoid + cross-entropy + gradient descent) trained on synthetic point clouds, with a plot of the decision boundary it learned — and a demonstration of where a straight line fails.
- **Project 2 — Titanic survival predictor:** your scratch model trained on real passenger data with features you prepare by hand, beating the "everyone dies" baseline and reaching ~78–80% accuracy.
- **Project 3 — sklearn + interpretation:** the same problem solved in three lines with scikit-learn, plus a plain-English write-up of *which features* pushed survival probability up or down.

## Concepts you will learn by doing

- **Classification** — predicting a category (survived / died) instead of a number.
- **Why a straight line can't output probabilities** — a line outputs −∞ to +∞; a probability must live between 0 and 1.
- **Sigmoid squashing** — a function that smoothly squeezes any number into the range (0, 1).
- **Probability outputs and decision thresholds** — the model says "0.83 chance of survival"; *you* choose where to cut (usually 0.5) to turn that into a yes/no.
- **Binary cross-entropy loss** — a loss that punishes confident wrong answers brutally, and why MSE is the wrong tool here.
- **Decision boundaries** — the line (or curve) in feature space where the model switches from "no" to "yes", made visible with a plot.
- **One-hot encoding** — turning a category like passenger class into 0/1 columns a model can use (first taste; more in [lesson 19](19-feature-engineering-pipelines.md)).
- **Dumb baselines** — always ask "what score would a model with zero intelligence get?" before celebrating your accuracy. This habit starts now and never stops.

## Before you start

Check you can do these (from earlier lessons): implement gradient descent by hand (lesson 09), train a linear regression from scratch (lesson 12), and filter/clean a DataFrame (lesson 06).

Activate your venv and install what's needed:

```bash
cd ~/ml/ml-learn
source .venv/bin/activate
pip install numpy pandas matplotlib seaborn scikit-learn
```

Create your work folder:

```bash
mkdir -p work/13-logistic-regression
cd work/13-logistic-regression
```

No downloads needed: Project 1 generates its own data, and seaborn fetches the Titanic dataset for you the first time you call `sns.load_dataset("titanic")` (needs internet, no account).

## Project 1 — 2D playground: draw the line

**Goal:** Build logistic regression from scratch on toy 2D data where you can *see* everything — the points, the training, and the boundary the model learns.

**Milestones**

- [ ] Make the data. In `playground.py`, generate two labeled clouds of points:

  ```python
  from sklearn.datasets import make_blobs
  X, y = make_blobs(n_samples=200, centers=2, cluster_std=1.5, random_state=42)
  ```

  `X` is a (200, 2) array — each row is a point with two coordinates (think: two features). `y` is 200 labels, 0 or 1. Scatter-plot the points, colored by label (`plt.scatter(X[:,0], X[:,1], c=y)`). Checkpoint: you should see two distinct blobs of different colors.
- [ ] Feel the problem with a straight line. Your lesson-12 linear regression outputs any number: −3.7, 0.4, 812. Try to interpret −3.7 as "the probability of class 1" — it's nonsense. Write one comment in your file explaining, in your own words, why raw linear output can't be a probability.
- [ ] Implement the sigmoid: `sigmoid(z) = 1 / (1 + np.exp(-z))`. Plot it for z from −10 to 10. Checkpoint: an S-shaped curve, flat near 0 on the left, flat near 1 on the right, crossing 0.5 exactly at z = 0. This is the squasher: any score in, valid probability out.
- [ ] Wire up the model. Prediction is two steps: linear score `z = X @ w + b` (exactly lesson 12), then `p = sigmoid(z)`. `p` is the model's probability that each point is class 1. Initialize `w` as zeros of shape (2,) and `b = 0.0`, and confirm `predict_proba(X)` returns 200 numbers, all exactly 0.5 (zero weights → zero score → sigmoid gives 0.5: the model starts out maximally unsure).
- [ ] Implement binary cross-entropy (BCE) loss: the average of `-(y * np.log(p) + (1 - y) * np.log(1 - p))`. Read it slowly: when the true label is 1, the loss is −log(p), which is tiny if p is near 1 and *explodes* toward infinity as p approaches 0. Confident and wrong = massive punishment. MSE would give a wrong answer a gentle, flat penalty (and makes gradient descent's job much harder here — the loss surface gets bumpy); BCE keeps the gradient strong exactly when the model is badly wrong.
- [ ] Train with gradient descent. Beautiful fact: the gradients look identical to linear regression's, just with sigmoid probabilities in place of raw predictions — `dw = X.T @ (p - y) / n` and `db = np.mean(p - y)`. Loop ~1000 steps with learning rate ~0.1, recording the loss each step. Checkpoint: loss starts at about 0.693 (that's −log(0.5), the cost of pure uncertainty) and falls steadily below 0.1.
- [ ] Turn probabilities into decisions with a threshold: `y_pred = (p >= 0.5).astype(int)`. Compute accuracy: `np.mean(y_pred == y)`. Checkpoint: accuracy above 0.95 on the blobs.
- [ ] Plot the decision boundary. Evaluate your model on a fine grid covering the plot area and color each grid point by predicted class, with the training points on top:

  ```python
  xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                       np.linspace(y_min, y_max, 300))
  grid = np.c_[xx.ravel(), yy.ravel()]        # every grid point as a row
  # get your model's 0/1 predictions for `grid`, reshape to xx.shape,
  # then: plt.contourf(xx, yy, preds, alpha=0.3)
  ```

  Checkpoint: a straight color border cleanly separating the two blobs. That border is where p = 0.5.
- [ ] Break it. Regenerate data with `make_moons(n_samples=200, noise=0.15, random_state=42)` — two interleaving crescents — and rerun everything. Checkpoint: accuracy drops to roughly 0.80–0.88 and the plot shows a straight boundary slicing hopelessly through curved data. A straight line cannot bend. **Remember this exact picture** — in [lesson 21](../phase-3-deep-learning/21-neural-networks-intuition.md) neural networks exist precisely to bend it.

<details><summary>Hints</summary>

- `np.log(0)` is −infinity, and your loss will print `nan` if any probability hits exactly 0 or 1. Clip first: `p = np.clip(p, 1e-15, 1 - 1e-15)`.
- If loss *increases* or oscillates, your learning rate is too high — drop it 10×. If loss barely moves, raise it 10×.
- For the boundary plot, get `x_min, x_max` from `X[:,0].min() - 1` and `X[:,0].max() + 1` (same idea for y), so the grid covers all points with a margin.
- Sanity-check gradients the lesson-09 way: nudge one weight by 1e-5, recompute the loss, and confirm (change in loss)/1e-5 ≈ your computed gradient.

</details>

**Definition of done:** one script that trains from scratch on blobs (accuracy > 0.95, boundary plot saved) and on moons (visibly failing straight boundary saved), with a loss curve that decreases smoothly.

## Project 2 — Titanic: who survives?

**Goal:** Point your scratch model at real, messy data. The features won't be ready-made numbers — you will build them by hand and learn why that work matters.

**Milestones**

- [ ] Load and look:

  ```python
  import seaborn as sns
  df = sns.load_dataset("titanic")
  print(df.shape, df["survived"].mean())
  ```

  Checkpoint: 891 rows, and about 0.38 of passengers survived.
- [ ] **Establish the dumb baseline first.** A "model" that predicts *died* for everyone gets `1 - 0.38 ≈ 0.62` accuracy — 62% with zero intelligence. Write this number down. Any model scoring near 62% has learned nothing, whatever its accuracy sounds like. From this lesson on, every project starts by computing the dumbest possible baseline.
- [ ] Prepare four features by hand (no libraries doing it for you — that's lesson 19):
  - `sex` → a 0/1 column (e.g. male=0, female=1). Models eat numbers, not strings.
  - `pclass` (ticket class 1/2/3) → **one-hot encode** it: three 0/1 columns, `class_1`, `class_2`, `class_3`, one per category, exactly one "hot" per row. Why not keep 1/2/3 as-is? That claims third class is "three times" first class on some scale — an ordering and spacing you invented. One-hot makes no such claim. (`pd.get_dummies(df["pclass"], prefix="class")` does it.)
  - `age` → it has missing values (check `df["age"].isna().sum()`); fill them with the median age, and note in a comment that you did this and why.
  - `fare` → scale it as in lesson 12: subtract the mean, divide by the standard deviation. Fares run 0–512 while your other columns are 0/1; unscaled, fare's gradient dwarfs everything else.
- [ ] Assemble `X` (shape (891, 6): sex, three class columns, age scaled too, fare) and `y = df["survived"].values`. Convert with `.astype(float)` — one-hot columns come out as True/False.
- [ ] Split before training: shuffle the row indices with `np.random.default_rng(42)`, take ~80% for training and hold out ~20% as a **test set** the model never sees during training — your honesty check (lesson 12's train/test idea, now a habit).
- [ ] Train your Project 1 model on the training split — same code, just 6 weights instead of 2. Checkpoint: loss starts near 0.693 and drops to roughly 0.45–0.50.
- [ ] Evaluate on the test split at threshold 0.5. Checkpoint: test accuracy around 0.78–0.80 — clearly above the 0.62 baseline. Print both numbers side by side, always.
- [ ] Play with the threshold: at 0.3 you predict "survived" more freely; at 0.7 only when very confident. Print accuracy at thresholds 0.3, 0.5, 0.7. The trade-off you feel here (catching more survivors vs. fewer false alarms) gets proper names — precision and recall — in [lesson 14](14-model-evaluation.md).

<details><summary>Hints</summary>

- `nan` loss? A missing value slipped through. Check `np.isnan(X).sum()` — it should be 0 before training.
- Accuracy stuck at ~0.62 (the baseline)? Your model is predicting one class for everything. Usually the culprit is an unscaled feature (fare or age) drowning the others, or a learning rate so high the weights blew up.
- Scale age and fare using the *training* split's mean/std, then apply those same numbers to the test split — the test set must not leak information into training.
- `pd.concat([...], axis=1)` glues your prepared columns into one DataFrame before `.values`.

</details>

**Definition of done:** a script that prints the baseline (~0.62) and your test accuracy (~0.78+), trained on features you prepared entirely by hand, evaluated only on rows the model never trained on.

## Project 3 — sklearn and the story in the weights

**Goal:** Reproduce your result with scikit-learn in a few lines, then do what separates an ML practitioner from a script-runner: read the model and explain it in plain English.

**Milestones**

- [ ] Same `X` and `y`, same split, then:

  ```python
  from sklearn.linear_model import LogisticRegression
  model = LogisticRegression()
  model.fit(X_train, y_train)
  print(model.score(X_test, y_test))
  ```

  Checkpoint: accuracy within a couple of points of your scratch model (~0.78–0.81). Weeks of understanding, three lines of library — this is the pattern for the rest of the curriculum: scratch first, library after.
- [ ] Extract the learned weights: `model.coef_[0]` (one weight per feature) and `model.intercept_`. Print each next to its feature name.
- [ ] Interpret the signs. A positive weight pushes the survival probability up as that feature grows; negative pushes it down. Checkpoint: `sex` (female=1) has the largest positive weight, and `class_3` is clearly negative.
- [ ] Compare with your scratch model's weights. Checkpoint: every feature has the same *sign* in both models (magnitudes may differ — sklearn applies regularization, a deliberate shrinking of weights you'll meet properly in [lesson 19](19-feature-engineering-pipelines.md)).
- [ ] Write the story. In `titanic_story.md` (5–10 sentences, plain English, no jargon): who was likely to survive the Titanic and why, according to the model? Which feature mattered most? Does it match the history you know ("women and children first", class-segregated deck access)? One sentence on what the model *cannot* tell you: weights show correlation in this data, not causation.

<details><summary>Hints</summary>

- Keep feature names in a list in the same order as your X columns, then `for name, w in zip(names, model.coef_[0]): print(name, round(w, 3))`.
- Comparing weight *magnitudes* across features is only fair because you scaled the numeric ones — a weight on raw fare (0–512) and a weight on sex (0/1) live on incomparable scales. Notice that your hand-scaling made interpretation possible.

</details>

**Definition of done:** sklearn matches your scratch accuracy, and `titanic_story.md` explains the model's reasoning so clearly that a friend who has never heard of ML could follow it.

## Stretch goals

- Add features: `sibsp + parch` (family size aboard), or a 0/1 `is_child` flag for age < 16. Can you push test accuracy past 0.80? Report baseline, before, and after.
- Plot survival probability vs. age for a 3rd-class male vs. a 1st-class female, holding other features fixed — two sigmoid-shaped curves that make the model's beliefs visible.
- On the moons data, add hand-crafted features `x1²`, `x2²`, and `x1·x2` as extra columns and retrain. The boundary is now curved and accuracy jumps above 0.95 — a straight line in a *bigger* feature space bends in the original one. This trick is the soul of kernels (lesson 17) and, in spirit, of neural networks (lesson 21).
- Implement mini-batch gradient descent (update on random chunks of ~32 rows instead of the full dataset each step) and compare the loss curves — a preview of how all deep learning trains.

## If you get stuck

- **Loss is `nan`:** almost always `log(0)`. Clip probabilities (`np.clip(p, 1e-15, 1 - 1e-15)`) and check for `nan` in `X` before training.
- **Loss stuck at 0.693:** the model is outputting 0.5 for everything — weights aren't updating. Print `dw` for a few steps: all zeros means a wiring bug; huge values mean the learning rate or an unscaled feature is exploding things.
- **Shapes:** `(891,)` vs `(891, 1)` mismatches produce silent wrong answers via broadcasting (lesson 05). Print `X.shape, y.shape, p.shape` — keep `y` and `p` both 1-D.
- Standing advice: read error messages bottom-up (the last line names the real problem), print shapes and a few values at every step, ask an AI assistant for a *hint* rather than a solution, and type every line of code yourself — copy-paste teaches your clipboard, not you.

## Resources

- **StatQuest: Logistic Regression** (YouTube — search "StatQuest logistic regression"): the friendliest walkthrough of sigmoid and why this is still "regression". Watch after Project 1 to consolidate.
- **3Blue1Brown: Neural Networks, Chapter 1** (YouTube — search "3Blue1Brown but what is a neural network"): watch the sigmoid appear inside a neural network — your Project 1 model is literally a one-neuron network, which is why this lesson matters so much.
- **scikit-learn LogisticRegression docs** ([scikit-learn.org](https://scikit-learn.org)): reference for the parameters you met in Project 3, including the regularization it applies by default.

## Skills unlocked

- [ ] I can explain why a linear model's raw output cannot be a probability, and what the sigmoid does about it.
- [ ] I can implement logistic regression from scratch: sigmoid, binary cross-entropy, and the gradient updates.
- [ ] I can explain why cross-entropy, not MSE, is the right loss for classification.
- [ ] I can plot a decision boundary and explain what the p = 0.5 line means.
- [ ] I can one-hot encode a categorical feature and say why the raw 1/2/3 encoding lies.
- [ ] I always compute a dumb baseline before judging any model's accuracy.
- [ ] I can read a trained model's weights and tell the story of its predictions in plain English.
- [ ] I know from experience where a linear decision boundary fails — and why that motivates neural networks.

## Next up

Accuracy is one number, and this week you already felt it hide things (the threshold game, the baseline trap) — next you build your own metrics library to judge models properly: [14 · Model Evaluation: Build Your Own Metrics Library](14-model-evaluation.md).
