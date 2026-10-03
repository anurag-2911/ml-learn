# 12 · Linear Regression from Scratch

**Phase 2 — Classical ML** · Estimated time: 1 week · Prerequisites: [09 · Calculus You Can Run: Gradient Descent](../phase-1-math/09-calculus-and-gradient-descent.md), [11 · Your First ML Model: k-Nearest Neighbors from Scratch](11-first-model-knn.md)

> The kNN model in lesson 11 never actually *learned* anything; it only memorized the training data. This lesson builds the first model that learns: it starts with wrong numbers, measures how wrong it is, and nudges the numbers to be less wrong, thousands of times, until it can predict California house prices from census data. This loop of **predict, measure, differentiate, update** is the single most important pattern in all of machine learning. The GPT model in lesson 33 is trained by this *exact same loop*; only the model in the middle is more complex. Here the loop runs on a model simple enough to see through, and everything after this lesson is a variation on it.

## What this lesson builds

- **Project 1:** a one-feature linear regression in pure NumPy that predicts a district's median house value from its median income, plus two plots: the data with the fitted line, and the loss curve of the training run.
- **Project 2:** the vectorized 8-feature version. Training blows up on the raw features, standardization fixes it, and a from-scratch R² function scores the model on a held-out test set.
- **Project 3:** a head-to-head comparison of the from-scratch model against scikit-learn's `LinearRegression`, ending with a `notes.md` of 3 insights about what actually drives house prices.

## Concepts covered

- **Regression vs classification**: predicting a number (a price) vs predicting a category (spam / not spam).
- **A model is a prediction rule with learnable parameters**: here, just a weight `w` and a bias `b`.
- **Mean squared error (MSE)**: one number that says how wrong the predictions are, on average.
- **The MSE gradient**: the derivative that shows which way to nudge `w` and `b` (lesson 09, now put to work).
- **The training loop**: predict → loss → gradient → update, the template that every model in the course follows, up to and including GPT.
- **Reading a loss curve**: what smooth descent, wild spikes, and flat lines each say about training.
- **Feature scaling**: why gradient descent fails when features live on wildly different scales, and how standardization fixes it.
- **R² (R-squared)**: how much better the model is than just guessing the average: 1.0 is perfect, 0.0 is no better than the average, and negative is worse.

## Before starting

**Check prerequisites.** Two earlier ideas should be clear enough to explain out loud: what a derivative says about a function (lesson 09), and why data is split into train and test sets (lesson 11). If either is fuzzy, skim those lessons first.

**Activate the venv and install packages** (from the repo root):

```bash
source .venv/bin/activate
pip install numpy matplotlib scikit-learn
```

**Get the dataset.** This lesson uses the California Housing dataset: ~20,000 census districts, each with 8 numeric features (median income, house age, rooms per household, ...) and a target (the district's median house value, in units of $100,000). No account is needed: scikit-learn downloads it (under 0.5 MB) and caches it in `~/scikit_learn_data` the first time it is loaded. Test it now:

```bash
python3 -c "
from sklearn.datasets import fetch_california_housing
d = fetch_california_housing()
print(d.data.shape, d.target.shape)
print(d.feature_names)
"
```

The command should print `(20640, 8) (20640,)` and the 8 feature names, starting with `MedInc`.

**Create the work folder:**

```bash
mkdir -p work/12-linear-regression
cd work/12-linear-regression
```

## Project 1 — One feature: income predicts house value

**Goal:** Predict a district's median house value from its median income alone, using a straight line `y = w*x + b` whose `w` and `b` are found by gradient descent, with every line of it written by hand in NumPy.

**Milestones**

- [ ] In `linreg1.py`, load the dataset and pull out one feature: `x = data[:, 0]` (that is `MedInc`, median income in tens of thousands of dollars) and `y = target`. Print both shapes. **Checkpoint: both are `(20640,)`.**
- [ ] Scatter-plot `x` vs `y` with matplotlib (use `alpha=0.1`, since there are 20k points). **Checkpoint: a cloud sloping up-and-right, with a suspicious horizontal line of points at y = 5.0.** The dataset caps prices at $500k. Real data has flaws; note it and move on.
- [ ] Write `predict(x, w, b)` returning `w * x + b`. This is the entire model: a prediction rule with two learnable parameters. Start with `w = 0.0, b = 0.0`, the model that predicts $0 for every house.
- [ ] Write `mse(y_pred, y)`: the mean of the squared differences, `mean((y_pred - y)**2)`. Squaring makes every error positive and punishes big misses hardest. **Checkpoint: with w=0, b=0, the loss is about 5.6.** That number is the "maximally clueless" baseline.
- [ ] Write `gradients(x, y, w, b)` returning `dw` and `db`, the derivatives of MSE with respect to `w` and `b`. Derive them on paper first, using two facts from lesson 09: the slope of `u**2` is `2*u`, and the chain rule (outer slope times inner slope) applied to `(w*x + b - y)²`, where the inner part `w*x + b - y` has slope `x` with respect to `w` and `1` with respect to `b`. They come out to means over the data.
- [ ] Trust nothing: verify the analytic gradient numerically. Nudge `w` by `1e-4`, recompute the loss, and check that `(loss_new - loss_old) / 1e-4` is close to `dw` (the same trick as in lesson 09). **Checkpoint: they agree to 3+ significant digits.**
- [ ] Write the training loop, and recognize it as *the* template:

  ```python
  for step in range(200):
      y_pred = predict(x, w, b)        # 1. predict
      loss = mse(y_pred, y)            # 2. measure
      dw, db = gradients(x, y, w, b)   # 3. differentiate
      w = w - lr * dw                  # 4. update
      b = b - lr * db
      losses.append(loss)
  ```

  Use `lr = 0.01` (the learning rate: how big each nudge is). Print the loss every 20 steps. **Checkpoint: the loss falls from ~5.6 and levels off around 0.7; it never rises.**
- [ ] Plot the loss curve (`losses` vs step number). **Checkpoint: a smooth downhill slide that flattens out, the signature of healthy training.** Save it as `loss_curve.png`. This exact plot is drawn for every model trained from here on.
- [ ] Plot the scatter again with the fitted line on top. **Checkpoint: the line cuts through the middle of the cloud; after 200 steps w lands around 0.45 and b around 0.3** (interpretation: each extra $10k of income predicts roughly $45k more house value). b is still creeping upward at this point: with 2000 steps the pair settles at the exact best fit, w ≈ 0.42 and b ≈ 0.45.

<details><summary>Hints</summary>

- The gradient of MSE with respect to `w` works out to `2 * mean((y_pred - y) * x)`, and with respect to `b` to `2 * mean(y_pred - y)`. Derive it before peeking; the derivation *is* the lesson.
- If the loss goes **up** instead of down, the sign is almost certainly flipped: it is `y_pred - y` inside the gradient, and `w - lr*dw` (minus, not plus) in the update.
- If the loss jumps around or shoots to huge numbers, the learning rate is too big. Divide it by 10 and retry. If it is too small, the curve barely moves; multiply it by 10. This trial-and-error tuning is normal, and it never goes away.
- Keep `w` and `b` plain Python floats and let NumPy broadcasting (lesson 05) handle the rest, with no loops over data points anywhere.

</details>

**Definition of done:** the loss curve decreases smoothly to ~0.7, the fitted line visibly matches the data, and the analytic gradients pass the numerical check.

## Project 2 — All 8 features (and the scaling ambush)

**Goal:** Use all 8 features at once with vectorized NumPy, run head-first into the most common gradient-descent failure in practice, and then fix it with feature standardization.

**Milestones**

- [ ] In `linreg8.py`, load the full `X` of shape `(20640, 8)` and `y`. Split into train and test: shuffle indices with `np.random.default_rng(42).permutation(len(y))`, take the first 80% for training, the rest for testing. **Checkpoint: train is `(16512, 8)`, test is `(4128, 8)`.** The test set stays locked away until the final milestone.
- [ ] Upgrade the model: `w` is now a vector of shape `(8,)` (one weight per feature) and prediction is one matrix multiply: `X @ w + b` (lesson 08). The gradients vectorize the same way: `dw` becomes shape `(8,)`, using `X.T` in place of `x`. MSE does not change at all.
- [ ] Re-run the gradient check from Project 1 on `w[0]` (the `MedInc` weight). **Checkpoint: analytic and numeric agree again.** It is cheap insurance, worth repeating every time. (On the raw `Population` weight the check drifts apart, because a `1e-4` nudge is too coarse for a feature whose values run into the thousands. That is a first hint of the scaling problem coming up next.)
- [ ] Now train on the **raw** features with the same `lr = 0.01`. **Checkpoint: the loss explodes to astronomical numbers or `nan` within a few steps.** This is not a bug in the code; it is the lesson. Lower `lr` until training survives, and notice that it now crawls uselessly.
- [ ] Diagnose it: print each feature's mean and standard deviation (a measure of how spread out values are, from lesson 10). **Checkpoint: `Population` has values in the thousands while `AveBedrms` sits near 1.** One learning rate cannot fit both: big-scale features get huge gradients (explosion) while small-scale ones get tiny ones (crawl).
- [ ] Fix it with **standardization**: for each feature column, subtract its mean and divide by its standard deviation, so every feature has mean ~0 and spread ~1. Compute the means and stds **on the training set only**, then apply those same numbers to the test set. The test set must never leak information into training.
- [ ] Retrain on standardized features with `lr = 0.1` for ~1000 steps. **Checkpoint: a smooth loss curve settling near 0.52–0.53.** That is better than Project 1's 0.7, because 8 features carry more information than one.
- [ ] Write `r_squared(y_pred, y)`: `1 - (sum of squared errors) / (sum of squared errors of always predicting the mean)`. It reads as "fraction of the variation the model explains": 1.0 is perfect, 0.0 means no better than guessing the average, and negative means worse than that.
- [ ] Evaluate on the held-out test set (standardized with the *training* stats). **Checkpoint: test R² around 0.6** (anywhere in 0.57–0.61 is right, depending on the shuffle). Sanity-check the R² function: feed it `y_pred = full of the training mean` and confirm that it returns ~0.0.

<details><summary>Hints</summary>

- The vectorized weight gradient is `2/n * X.T @ (y_pred - y)`. Check that its shape is `(8,)` before anything else. Printing shapes is debugging tool #1.
- Getting `(20640, 1)` where `(20640,)` is expected? Mixing those two shapes makes NumPy broadcast a huge `(20640, 20640)` array. Keep `y`, `y_pred`, and errors all 1-D.
- Standardize in one line per set: `Xs = (X - mu) / sigma` with `mu = X_train.mean(axis=0)` and `sigma = X_train.std(axis=0)`. The `axis=0` means "per column".
- Can standardizing break anything? It cannot hurt correctness: it only rescales the hill that gradient descent walks down, from a squashed canyon into a round bowl.

</details>

**Definition of done:** training on raw features exploded, the reason *why* can be explained in one sentence, and the standardized model reaches test R² ≈ 0.6 with a clean loss curve.

## Project 3 — Sanity check against scikit-learn

**Goal:** Prove that the from-scratch implementation is genuinely correct by matching scikit-learn's `LinearRegression`, then use the fitted weights to extract real insight from the data.

**Milestones**

- [ ] In `compare.py`, rebuild what Project 2 used: copy the loading, the seed-42 split, the standardization (name the results `X_train_std` and `X_test_std`), the standardized training loop and `r_squared` from `linreg8.py`, so that `w` and `b` exist here too. Then fit sklearn on **the same standardized training data**:

  ```python
  from sklearn.linear_model import LinearRegression
  sk = LinearRegression().fit(X_train_std, y_train)
  print(sk.coef_, sk.intercept_)
  ```

- [ ] Print sklearn's 8 coefficients and intercept next to the from-scratch `w` and `b`, feature name by feature name. **Checkpoint: each pair matches to about 2 decimal places.** (sklearn solves the problem exactly with linear algebra; gradient descent approached the same answer by walking downhill. Small gaps shrink with longer training.)
- [ ] Compute sklearn's test R² and compare it with the NumPy model's test R². **Checkpoint: they agree to ~2–3 decimals.** The from-scratch code has now independently reproduced a library that millions of people trust.
- [ ] Rank the features by the absolute value of their weights. Because the features are standardized, the weights are directly comparable: each says how much the prediction moves per "one typical unit" of that feature. **Checkpoint: `MedInc` is the biggest positive weight; `Latitude` and `Longitude` are both strongly negative.**
- [ ] Write `notes.md` with 3 insights, in the learner's own words. Questions to think about: Why would latitude and longitude *both* be negative in California? (Where do house prices peak on the map?) `AveBedrms` has a *positive* weight while `AveRooms` is negative: do more bedrooms per house *cause* higher prices, or is a weight not the same thing as a cause? Which feature is the most surprising?

<details><summary>Hints</summary>

- If the coefficients are wildly different, first suspect that sklearn was fitted on raw features while the NumPy model was trained on standardized ones (or vice versa). Same data in, same numbers out.
- If they are close but not 2-decimals close, run the training loop for 5000+ steps. Gradient descent approaches the exact solution asymptotically.
- For the ranking, `np.argsort(np.abs(w))[::-1]` gives feature indices from most to least influential; index into `feature_names` with it.

</details>

**Definition of done:** the NumPy model's coefficients and R² match sklearn's, and `notes.md` contains 3 concrete insights that reference actual weights.

## Stretch goals

- **The closed-form solution:** linear regression is special; the best `w` can be computed directly with the *normal equation*, `w = (XᵀX)⁻¹Xᵀy` (search "normal equation linear regression"). Implement it in 3 lines with `np.linalg` and confirm it matches gradient descent. Then consider: why does deep learning use gradient descent anyway? (Hint: no closed form exists for anything nonlinear.)
- **Learning-rate safari:** train with `lr` in `[1.0, 0.3, 0.1, 0.03, 0.001]` and plot all five loss curves on one chart. The chart shows divergence, the sweet spot, and the crawl: a picture worth keeping for lesson 25.
- **Polynomial features:** add `MedInc²` as a 9th feature (standardize it too). Does test R² improve? This step is manual feature engineering, the whole topic of lesson 19.
- **Mini-batch gradient descent:** instead of the full 16k rows per step, use a random batch of 32. The loss gets noisier but steps get cheap. This is how every deep network is actually trained.

## Getting unstuck

- **Loss is `nan` or astronomically huge:** the learning rate is too high, or standardization was skipped. Shrink `lr` 10× and check feature scales.
- **Loss steadily increases:** the sign is flipped. It must be `y_pred - y` in the gradient and `w - lr*dw` (subtract, not add) in the update.
- **Shape errors, or memory blows up:** print `.shape` of every array in the loop. A `(n,)` vs `(n,1)` mismatch silently broadcasts into an `(n,n)` matrix.
- **Loss fine, line looks wrong:** the plot probably uses unstandardized `x` with weights learned on standardized `x`. Pick one world for each plot.
- Standing advice: read error messages bottom-up (the last line names the real problem), print shapes and a few actual values before theorizing, ask an AI assistant for a *hint* rather than the solution, and type every line of code by hand. Copy-paste teaches the clipboard, not the learner.

## Resources

- **StatQuest: Linear Regression** — search YouTube for "StatQuest Linear Regression"; the friendliest visual walk-through of fitting lines, least squares, and R². Watch it *after* Project 1, when its content will already be familiar.
- **scikit-learn linear models guide** — https://scikit-learn.org/stable/modules/linear_model.html — the reference for `LinearRegression` (the benchmark in Project 3), plus a preview of variants (Ridge, Lasso) that appear in lesson 19.

## Skills unlocked

- [ ] I can explain the difference between regression and classification with an example of each.
- [ ] I can describe a model as a prediction rule with learnable parameters, and name them for linear regression.
- [ ] I can implement MSE and derive its gradient with respect to `w` and `b` by hand.
- [ ] I can write the four-step training loop from memory: predict, measure, differentiate, update.
- [ ] I can read a loss curve and diagnose too-high or too-low learning rates from its shape.
- [ ] I can explain why gradient descent fails on unscaled features and fix it with standardization, using training-set statistics only.
- [ ] I can implement R² and say in plain words what a value of 0.6 means.
- [ ] I can verify a from-scratch model against a trusted library before believing it.

## Next up

The next lesson keeps the same training loop with one small twist, squashing the output through a sigmoid so that the regressor becomes a classifier: [13 · Logistic Regression from Scratch](13-logistic-regression.md).
