# 12 · Linear Regression from Scratch

**Phase 2 — Classical ML** · Estimated time: 1 week · Prerequisites: [09 · Calculus You Can Run: Gradient Descent](../phase-1-math/09-calculus-and-gradient-descent.md), [11 · Your First ML Model: k-Nearest Neighbors from Scratch](11-first-model-knn.md)

> Your kNN model in lesson 11 never actually *learned* anything — it just memorized the training data. This week you build your first model that learns: it starts with wrong numbers, measures how wrong it is, and nudges the numbers to be less wrong, thousands of times, until it can predict California house prices from census data. That nudge-loop — **predict, measure, differentiate, update** — is the single most important pattern in all of machine learning. The GPT you will train in lesson 33 is trained by this *exact same loop*; only the model in the middle gets fancier. Learn the loop here, on a model simple enough to see through, and everything after this is variations on a theme.

## What you will build

- **Project 1:** a one-feature linear regression in pure NumPy that predicts a district's median house value from its median income, plus two plots: the data with your fitted line, and the loss curve of your training run.
- **Project 2:** the vectorized 8-feature version, where you watch training blow up on raw features, fix it with standardization, and score it with your own R² on a held-out test set.
- **Project 3:** a head-to-head comparison of your model against scikit-learn's `LinearRegression`, ending with a `notes.md` of 3 insights about what actually drives house prices.

## Concepts you will learn by doing

- **Regression vs classification** — predicting a number (a price) vs predicting a category (spam / not spam).
- **A model is a prediction rule with learnable parameters** — here just a weight `w` and a bias `b`.
- **Mean squared error (MSE)** — one number that says how wrong your predictions are, on average.
- **The MSE gradient** — the derivative that tells you which way to nudge `w` and `b` (lesson 09, now with a job).
- **The training loop** — predict → loss → gradient → update, the template every model through GPT follows.
- **Reading a loss curve** — what smooth descent, wild spikes, and flat lines each tell you.
- **Feature scaling** — why gradient descent chokes when features live on wildly different scales, and how standardization fixes it.
- **R² (R-squared)** — how much better your model is than just guessing the average, on a 0-to-1 scale.

## Before you start

**Check prerequisites.** You should be able to explain, out loud, what a derivative tells you about a function (lesson 09) and why we split data into train and test sets (lesson 11). If either is fuzzy, skim those lessons first.

**Activate your venv and install packages** (from the repo root):

```bash
source .venv/bin/activate
pip install numpy matplotlib scikit-learn
```

**Get the dataset.** You'll use the California Housing dataset: ~20,000 census districts, each with 8 numeric features (median income, house age, rooms per household, ...) and a target — the district's median house value in units of $100,000. No account needed; scikit-learn downloads it (about 1 MB) and caches it in `~/scikit_learn_data` the first time you call it. Test it now:

```bash
python3 -c "
from sklearn.datasets import fetch_california_housing
d = fetch_california_housing()
print(d.data.shape, d.target.shape)
print(d.feature_names)
"
```

You should see `(20640, 8) (20640,)` and the 8 feature names, starting with `MedInc`.

**Create your work folder:**

```bash
mkdir -p work/12-linear-regression
cd work/12-linear-regression
```

## Project 1 — One feature: income predicts house value

**Goal:** Predict a district's median house value from its median income alone, using a straight line `y = w*x + b` whose `w` and `b` are found by gradient descent — every line of it written by you in NumPy.

**Milestones**

- [ ] In `linreg1.py`, load the dataset and pull out one feature: `x = data[:, 0]` (that's `MedInc`, median income in tens of thousands of dollars) and `y = target`. Print both shapes. **Checkpoint: both are `(20640,)`.**
- [ ] Scatter-plot `x` vs `y` with matplotlib (use `alpha=0.1`, there are 20k points). **Checkpoint: a cloud sloping up-and-right, with a suspicious horizontal line of points at y = 5.0** — the dataset caps prices at $500k. Real data has warts; note it and move on.
- [ ] Write `predict(x, w, b)` returning `w * x + b`. This is your entire model: a prediction rule with two learnable parameters. Start with `w = 0.0, b = 0.0` — the model that predicts $0 for every house.
- [ ] Write `mse(y_pred, y)`: the mean of the squared differences, `mean((y_pred - y)**2)`. Squaring makes every error positive and punishes big misses hardest. **Checkpoint: with w=0, b=0, the loss is about 5.6.** That number is your "maximally clueless" baseline.
- [ ] Write `gradients(x, y, w, b)` returning `dw` and `db` — the derivatives of MSE with respect to `w` and `b`. Derive them on paper first, exactly like you differentiated by hand in lesson 09 (chain rule on `(w*x + b - y)²`). They come out to means over the data.
- [ ] Trust nothing: verify your analytic gradient numerically. Nudge `w` by `1e-4`, recompute the loss, and check `(loss_new - loss_old) / 1e-4` is close to your `dw` (same trick as lesson 09). **Checkpoint: they agree to 3+ significant digits.**
- [ ] Write the training loop — and recognize it as *the* template:

  ```python
  for step in range(200):
      y_pred = predict(x, w, b)        # 1. predict
      loss = mse(y_pred, y)            # 2. measure
      dw, db = gradients(x, y, w, b)   # 3. differentiate
      w = w - lr * dw                  # 4. update
      b = b - lr * db
      losses.append(loss)
  ```

  Use `lr = 0.01` (the learning rate — how big each nudge is). Print the loss every 20 steps. **Checkpoint: loss falls from ~5.6 and levels off around 0.7 — never rises.**
- [ ] Plot the loss curve (`losses` vs step number). **Checkpoint: a smooth downhill slide flattening out — the signature of healthy training.** Save it as `loss_curve.png`. You will draw this exact plot for every model you ever train.
- [ ] Plot the scatter again with your fitted line on top. **Checkpoint: the line cuts through the middle of the cloud; w lands around 0.4 and b around 0.4–0.5** (interpretation: each extra $10k of income predicts roughly $40k more house value).

<details><summary>Hints</summary>

- The gradient of MSE with respect to `w` works out to `2 * mean((y_pred - y) * x)`, and with respect to `b` to `2 * mean(y_pred - y)`. Derive it yourself before peeking — the derivation *is* the lesson.
- If your loss goes **up** instead of down, you almost certainly have the sign flipped: it's `y_pred - y` inside the gradient, and `w - lr*dw` (minus!) in the update.
- If the loss jumps around or shoots to huge numbers, your learning rate is too big. Divide it by 10 and retry. Too small and the curve barely moves — multiply by 10. This dial-twiddling is normal; you'll do it forever.
- Keep `w` and `b` plain Python floats and let NumPy broadcasting (lesson 05) handle the rest — no loops over data points anywhere.

</details>

**Definition of done:** Your loss curve decreases smoothly to ~0.7, your fitted line visibly matches the data, and your analytic gradients pass the numerical check.

## Project 2 — All 8 features (and the scaling ambush)

**Goal:** Use all 8 features at once with vectorized NumPy — and run head-first into the most common gradient-descent failure in practice, then fix it with feature standardization.

**Milestones**

- [ ] In `linreg8.py`, load the full `X` of shape `(20640, 8)` and `y`. Split into train and test: shuffle indices with `np.random.default_rng(42).permutation(len(y))`, take the first 80% for training, the rest for testing. **Checkpoint: train is `(16512, 8)`, test is `(4128, 8)`.** The test set stays locked away until the final milestone.
- [ ] Upgrade the model: `w` is now a vector of shape `(8,)` (one weight per feature) and prediction is one matrix multiply: `X @ w + b` (lesson 08). The gradients vectorize the same way — `dw` becomes shape `(8,)`, using `X.T` in place of `x`. MSE doesn't change at all.
- [ ] Re-run the gradient check from Project 1 on one element of `w`. **Checkpoint: analytic and numeric agree again.** Cheap insurance, every time.
- [ ] Now train on the **raw** features with the same `lr = 0.01`. **Checkpoint: the loss explodes to astronomical numbers or `nan` within a few steps.** This is not a bug in your code — it's the lesson. Lower `lr` until it survives and notice it now crawls uselessly.
- [ ] Diagnose it: print each feature's mean and standard deviation (a measure of how spread out values are, from lesson 10). **Checkpoint: `Population` has values in the thousands while `AveBedrms` sits near 1.** One learning rate cannot fit both: big-scale features get huge gradients (explosion) while small-scale ones get tiny ones (crawl).
- [ ] Fix it with **standardization**: for each feature column, subtract its mean and divide by its standard deviation, so every feature has mean ~0 and spread ~1. Compute the means and stds **on the training set only**, then apply those same numbers to the test set — the test set must never leak information into training.
- [ ] Retrain on standardized features with `lr = 0.1` for ~1000 steps. **Checkpoint: a smooth loss curve settling near 0.52–0.53** — better than Project 1's 0.7, because 8 features carry more information than one.
- [ ] Write `r_squared(y_pred, y)`: `1 - (sum of squared errors) / (sum of squared errors of always predicting the mean)`. It reads as "fraction of the variation my model explains": 1.0 is perfect, 0.0 means no better than guessing the average, negative means worse than that.
- [ ] Evaluate on the held-out test set (standardized with the *training* stats). **Checkpoint: test R² around 0.6** (anywhere in 0.57–0.61 is right, depending on the shuffle). Sanity-check your R² function: feed it `y_pred = full of the training mean` and confirm you get ~0.0.

<details><summary>Hints</summary>

- The vectorized weight gradient is `2/n * X.T @ (y_pred - y)` — check that its shape is `(8,)` before anything else. Shape-printing is debugging superpower #1.
- Getting `(20640, 1)` where you expect `(20640,)`? Mixing those two shapes makes NumPy broadcast a `(20640, 20640)` monster. Keep `y`, `y_pred`, and errors all 1-D.
- Standardize in one line per set: `Xs = (X - mu) / sigma` with `mu = X_train.mean(axis=0)` and `sigma = X_train.std(axis=0)`. The `axis=0` means "per column".
- Wondering if standardizing broke anything? It can't hurt correctness — it only rescales the hill gradient descent walks down, from a squashed canyon into a round bowl.

</details>

**Definition of done:** You watched raw-feature training explode, can explain *why* in one sentence, and your standardized model reaches test R² ≈ 0.6 with a clean loss curve.

## Project 3 — Sanity check against scikit-learn

**Goal:** Prove your from-scratch implementation is genuinely correct by matching scikit-learn's `LinearRegression`, then use the fitted weights to extract real insight from the data.

**Milestones**

- [ ] In `compare.py`, fit sklearn on **the same standardized training data** you used in Project 2:

  ```python
  from sklearn.linear_model import LinearRegression
  sk = LinearRegression().fit(X_train_std, y_train)
  print(sk.coef_, sk.intercept_)
  ```

- [ ] Print sklearn's 8 coefficients and intercept next to your `w` and `b`, feature name by feature name. **Checkpoint: every number matches yours to about 2 decimal places.** (sklearn solves the problem exactly with linear algebra; you approached the same answer by walking downhill. Small gaps shrink if you train longer.)
- [ ] Compute sklearn's test R² and compare with yours. **Checkpoint: they agree to ~2–3 decimals.** You have now independently reproduced a library that millions of people trust. Let that land.
- [ ] Rank the features by the absolute value of their weights. Because the features are standardized, the weights are directly comparable — each says how much the prediction moves per "one typical unit" of that feature. **Checkpoint: `MedInc` is the biggest positive weight; `Latitude` and `Longitude` are both strongly negative.**
- [ ] Write `notes.md` with 3 insights in your own words. Prompts to chew on: Why would latitude and longitude *both* be negative in California — where do house prices peak on the map? `AveBedrms` has a *positive* weight while `AveRooms` is negative — does more bedrooms per house *cause* higher prices, or is a weight not the same thing as a cause? Which feature surprised you most?

<details><summary>Hints</summary>

- If coefficients are wildly different, first suspect you fitted sklearn on raw features but trained yours on standardized ones (or vice versa). Same data in, same numbers out.
- If they're close but not 2-decimals close, run your training loop for 5000+ steps — gradient descent approaches the exact solution asymptotically.
- For the ranking, `np.argsort(np.abs(w))[::-1]` gives feature indices from most to least influential; index into `feature_names` with it.

</details>

**Definition of done:** Your coefficients and R² match sklearn's, and `notes.md` contains 3 concrete insights that reference actual weights.

## Stretch goals

- **The closed-form solution:** linear regression is special — the best `w` can be computed directly with the *normal equation*, `w = (XᵀX)⁻¹Xᵀy` (search "normal equation linear regression"). Implement it in 3 lines with `np.linalg` and confirm it matches gradient descent. Then ponder: why does deep learning use gradient descent anyway? (Hint: no closed form exists for anything nonlinear.)
- **Learning-rate safari:** train with `lr` in `[1.0, 0.3, 0.1, 0.03, 0.001]` and plot all five loss curves on one chart. You'll see divergence, the sweet spot, and the crawl — a picture worth keeping for lesson 25.
- **Polynomial features:** add `MedInc²` as a 9th feature (standardize it too). Does test R² improve? You've just done manual feature engineering — lesson 19's whole topic.
- **Mini-batch gradient descent:** instead of the full 16k rows per step, use a random batch of 32. Loss gets noisier but steps get cheap — this is how every deep network is actually trained.

## If you get stuck

- **Loss is `nan` or astronomically huge:** learning rate too high, or you skipped standardization. Shrink `lr` 10× and check feature scales.
- **Loss steadily increases:** flipped sign — it must be `y_pred - y` in the gradient and `w - lr*dw` (subtract!) in the update.
- **Shape errors, or memory blows up:** print `.shape` of every array in the loop. A `(n,)` vs `(n,1)` mismatch silently broadcasts into an `(n,n)` matrix.
- **Loss fine, line looks wrong:** you're probably plotting against unstandardized `x` with weights learned on standardized `x`. Pick one world for each plot.
- Standing advice: read error messages bottom-up (the last line names the real problem), print shapes and a few actual values before theorizing, ask your AI assistant for a *hint* rather than the solution, and type every line of code yourself — copy-paste teaches your clipboard, not you.

## Resources

- **StatQuest: Linear Regression** — search YouTube for "StatQuest Linear Regression"; the friendliest visual walk-through of fitting lines, least squares, and R². Watch it *after* Project 1 and enjoy already knowing it.
- **scikit-learn linear models guide** — https://scikit-learn.org/stable/modules/linear_model.html — the reference for `LinearRegression` you race against in Project 3, plus a preview of variants (Ridge, Lasso) you'll meet in lesson 19.

## Skills unlocked

- [ ] I can explain the difference between regression and classification with an example of each.
- [ ] I can describe a model as a prediction rule with learnable parameters, and name them for linear regression.
- [ ] I can implement MSE and derive its gradient with respect to `w` and `b` by hand.
- [ ] I can write the four-step training loop from memory: predict, measure, differentiate, update.
- [ ] I can read a loss curve and diagnose too-high or too-low learning rates from its shape.
- [ ] I can explain why gradient descent fails on unscaled features and fix it with standardization — using training-set statistics only.
- [ ] I can implement R² and say in plain words what a value of 0.6 means.
- [ ] I can verify a from-scratch model against a trusted library before believing it.

## Next up

The same training loop, one small twist — squash the output through a sigmoid and your regressor becomes a classifier: [13 · Logistic Regression from Scratch](13-logistic-regression.md).
