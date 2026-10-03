# 09 · Calculus You Can Run: Gradient Descent

**Phase 1 — Math** · Estimated time: 1 week · Prerequisites: [05 · NumPy](../phase-0-foundations/05-numpy.md), [07 · Data Visualization](../phase-0-foundations/07-data-visualization.md), [08 · Linear Algebra by Code](08-linear-algebra-by-code.md)

> Every neural network, from a toy digit classifier to GPT, is trained by one algorithm: gradient descent. It works by asking, over and over, "which way is downhill?" and taking a small step that way. The math behind "which way is downhill" is the derivative. There are no derivative rules to memorize: this lesson *computes* derivatives numerically with three lines of Python, shows gradient descent rolling into a valley on a plot, and then uses it to fit a first model to noisy data. Later, when deep networks are trained, every failure mode caused on purpose this week will look familiar.

## What this lesson builds

- **Project 1 — Numerical differentiator:** a `deriv(f, x)` function that measures the slope of any Python function, verified against known answers, plus a plot of a curve with its tangent line.
- **Project 2 — Gradient descent visualized:** a from-scratch optimizer that minimizes a 1D function (with a plot of its path), then a 2D "bowl" (with the descent path drawn on a contour plot), plus experiments that break it on purpose.
- **Project 3 — Fit a line the hard way:** a tiny training loop that fits `y = w*x + b` to noisy points by gradient-descending the mean squared error: a first trained model.

## Concepts covered

- **Derivative**: the local slope of a function at a point, meaning how much the output changes per tiny nudge of the input.
- **Numerical differentiation**: estimating that slope as `(f(x+h) - f(x)) / h` with a tiny `h`; no algebra is needed.
- **Partial derivative**: the slope with respect to one input of a multi-input function, holding the others fixed.
- **Gradient**: the vector of all partial derivatives; it points in the direction of steepest *ascent*.
- **Chain rule**: how slopes of nested functions multiply together (verified numerically, not proved on paper).
- **Gradient descent**: repeatedly stepping a little bit in the *negative* gradient direction to find a minimum.
- **Learning rate**: the step-size knob. If it is too big, descent diverges; if it is too small, descent crawls.
- **Loss surface**: the landscape that gradient descent moves down. It is the loss (a badness score) as a function of the model's parameters.
- **Local minimum**: a valley that is lowest nearby but maybe not lowest overall.

## Before starting

1. Check the prerequisites: creating NumPy arrays and doing arithmetic on them (lesson 05), and making line, scatter and contour-like plots with matplotlib (lesson 07). No calculus background is needed; that is the point.
2. Activate the venv at the repo root and confirm that the tools are there (both were installed in earlier lessons):

```bash
cd ~/ml/ml-learn
source .venv/bin/activate
python3 -c "import numpy, matplotlib; print('ready')"
```

If that prints an error, install them: `pip install numpy matplotlib`.

3. Create the work folder:

```bash
mkdir -p work/09-calculus-and-gradient-descent
cd work/09-calculus-and-gradient-descent
```

There are no datasets to download; the code generates every number.

## Project 1 — Numerical differentiator

**Goal:** Build `deriv(f, x)`, a function that returns the slope of any function `f` at any point `x`, and check that it agrees with the textbook answers without having to learn them.

**Milestones**

- [ ] Create `deriv.py`. Write `deriv(f, x, h=1e-5)` returning `(f(x + h) - f(x)) / h`. That is the whole idea of a derivative: nudge the input by a tiny amount `h`, see how much the output moved, and divide. It is the slope of the line through two nearby points on the curve.
- [ ] Test it on `f(x) = x**2`. The true derivative of `x**2` is `2x` (trust this one fact; the lesson verifies it rather than deriving it). Check `deriv(lambda x: x**2, 3.0)`. Checkpoint: the result is roughly `6.0` (something like `6.00001`).
- [ ] Test on `math.sin`: the true derivative of `sin(x)` is `cos(x)`. Loop over a few points (`0, 0.5, 1.0, 2.0`) and print `deriv(math.sin, x)` next to `math.cos(x)`. Checkpoint: they match to 4-5 decimal places at every point.
- [ ] Experiment with `h`: try `h = 0.1`, `1e-5`, and `1e-13` on `x**2` at `x=3`. Checkpoint: `0.1` is visibly off (about 6.1), `1e-5` is excellent, and `1e-13` gets *worse* again. Floating-point numbers only carry about 16 digits, so subtracting two nearly-equal values destroys precision. Note the sweet spot; the rest of the lesson reuses it.
- [ ] Optional upgrade: the *centered* difference `(f(x+h) - f(x-h)) / (2*h)` is more accurate for the same `h`. Compare its error against the one-sided version.
- [ ] Tangent line plot: pick `f(x) = x**3 - 2*x` and a point like `x0 = 1.5`. Plot `f` over `[-2.5, 2.5]`, then draw the tangent line at `x0`: a straight line through `(x0, f(x0))` with slope `deriv(f, x0)`, that is, `y = f(x0) + slope * (x - x0)`. Checkpoint: the line touches the curve at `x0` and matches its tilt there, instead of crossing through at an angle. Try a couple of different `x0` values.
- [ ] Chain rule, numerically: composite functions multiply their slopes. For `g(x) = sin(x**2)`, the chain rule says slope = `cos(x**2) * 2*x` (outer slope times inner slope). At `x = 1.3`, print `deriv(lambda x: math.sin(x**2), 1.3)` next to `math.cos(1.3**2) * 2 * 1.3`. Checkpoint: they agree to several decimals. Keep this result in mind: backpropagation is this rule, applied thousands of times.

<details><summary>Hints</summary>

- `deriv` should accept any callable, so that `math.sin`, a `lambda` or a `def`-ed function can be passed interchangeably.
- For the tangent line, compute the line's y-values with NumPy over the same x-array used for the curve: `y_tan = f(x0) + slope * (xs - x0)`.
- If the sin/cos check is off by a lot, make sure the code works in radians (Python's default). Do not convert to degrees anywhere.
- Weird `h=1e-13` results are not a bug in the code; print `f(x+h) - f(x)` on its own to see how tiny (and noisy) the difference gets.

</details>

**Definition of done:** `deriv` matches `2x` and `cos(x)` to at least 4 decimal places with `h=1e-5`, the tangent-line plot looks right at any chosen point, and the chain-rule check passes.

## Project 2 — Gradient descent visualized

**Goal:** Implement gradient descent from scratch, watch it find the minimum of a function, then break it in the two classic ways, so that both failures are easy to recognize in all later ML work.

**Milestones**

- [ ] Create `descent.py`. Target: `f(x) = (x - 3)**2 + 2`, a parabola whose lowest point is at `x = 3` (minimum value 2). Plot it over `[-2, 8]` first. Checkpoint: the plot shows a U-shape with its bottom at x=3.
- [ ] Write the descent loop: start at `x = -1.0`, and repeat 50 times: `x = x - lr * deriv(f, x)` with learning rate `lr = 0.1`. The derivative gives the uphill direction, so subtracting it moves `x` downhill; `lr` scales how big each step is. Record every `x` in a list. Checkpoint: the printed `x` values march from -1 toward 3 and settle around `2.9999...`.
- [ ] Visualize the path: scatter the points `(x_i, f(x_i))` on top of the curve, connected by a line (use a colormap or fading alpha so that the order is visible). Checkpoint: the dots start high on the left slope, take big steps at first, then bunch up tightly at the bottom. The steps shrink automatically because the slope itself shrinks near the minimum.
- [ ] Failure mode 1 (learning rate 10x too big): rerun with `lr = 1.1`. Checkpoint: `x` overshoots the minimum, bounces to the other side farther away each time, and after about 20 steps the values explode toward huge numbers. This is *divergence*. In deep learning it shows up as the loss shooting to `inf` or `NaN`.
- [ ] Failure mode 2 (learning rate 100x too small): rerun with `lr = 0.001`. Checkpoint: after 50 steps `x` has barely left the starting area (it is still below 0). It *would* get there eventually; this is the "training is crawling" failure. Plot `f(x_i)` versus iteration number for all three learning rates on one chart to compare them.
- [ ] Go 2D: define a bowl `f(x, y) = x**2 + 3*y**2` (an oval-shaped valley with its bottom at `(0, 0)`). Now there are two slopes: the **partial derivative** with respect to `x` (nudge `x`, hold `y` fixed) and with respect to `y` (nudge `y`, hold `x` fixed). Write `grad(f, x, y)` that returns both, using the same numerical trick twice. The pair is the **gradient**: the arrow pointing steepest uphill. Checkpoint: at `(1, 1)`, `grad` returns roughly `(2.0, 6.0)`.
- [ ] Descend the bowl from `(-4, 3)` with `lr = 0.1` for 60 steps, updating `x` and `y` together (each minus `lr` times its own partial). Record the path. Checkpoint: the point ends within 0.01 of `(0, 0)`.
- [ ] Contour plot: use `np.meshgrid` and `plt.contour` to draw the bowl's level curves (rings of equal height, like a topographic hiking map), then draw the descent path on top with `plt.plot(xs, ys, 'o-')`. Checkpoint: the path cuts across the rings roughly perpendicular to them (the gradient is always perpendicular to a level curve), zig-zagging slightly in the narrow `y` direction before settling at the center.

<details><summary>Hints</summary>

- Reuse `deriv` from Project 1 (`from deriv import deriv` if the file is in the same folder).
- For partials of `f(x, y)`: freeze one variable with a lambda, e.g. the x-partial is `deriv(lambda v: f(v, y), x)`.
- `meshgrid` recipe: `X, Y = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-4, 4, 100))`, then `plt.contour(X, Y, f(X, Y), levels=20)`. This works because `f` uses only array-friendly operations.
- If the divergence run crashes with an overflow error, that *is* the lesson. Cap the loop with a check like `if abs(x) > 1e6: break`, so that the plot can still show what happened.

</details>

**Definition of done:** 1D descent converges to x≈3 and the three-learning-rates comparison plot is finished; 2D descent reaches (0,0) and the contour plot shows the path. Explain in one sentence each what too-big and too-small learning rates look like.

## Project 3 — Fit a line the hard way

**Goal:** Use gradient descent to fit a line `y = w*x + b` to noisy data by minimizing mean squared error. It is the same loop, but now the "position" being optimized is the model's parameters. This is training.

**Milestones**

- [ ] Create `fit_line.py`. Generate fake data with a known answer: pick `true_w = 2.5`, `true_b = -1.0`, then `xs = np.linspace(0, 10, 50)` and `ys = true_w * xs + true_b + np.random.normal(0, 2, size=50)` (that last term is noise: random wobble like real measurements have). Scatter-plot it. Checkpoint: a cloud of points rising to the right.
- [ ] Write the loss function: `mse(w, b)` = the **mean squared error**, the average of `(prediction - actual)**2` over all points, where prediction is `w * xs + b`. Squaring makes every miss positive and punishes big misses hardest. One line with NumPy: `np.mean((w * xs + b - ys)**2)`. Checkpoint: `mse(true_w, true_b)` is small (around 4, since noise had standard deviation 2) while `mse(0, 0)` is much larger (hundreds).
- [ ] Pause and take stock: `mse` is a function of two numbers, `w` and `b`, exactly like the bowl in Project 2. This is a **loss surface**: the landscape of how wrong the model is over all possible parameter values. Training a model means descending this surface. Optional: contour-plot `mse` over a grid of `w` in `[0, 5]`, `b` in `[-5, 3]` to see the valley.
- [ ] Descend it: start at `w, b = 0.0, 0.0`, compute both partials of `mse` numerically with the same `grad` approach, and update both with `lr = 0.01` for 200 steps. Record the loss every step. Checkpoint: the printed loss drops fast, then flattens near the noise floor (around 4).
- [ ] Check the recovered parameters. Checkpoint: `w` is within about 0.2 of 2.5 and `b` within about 0.7 of -1.0 (the noise means the fit will not be exact; rerun with different noise and watch the answers wobble slightly).
- [ ] Two plots: (1) loss versus iteration, a first *training curve* (a shape that comes up throughout the rest of this curriculum); (2) the scatter of points with the fitted line drawn through it and, for comparison, the true line. Checkpoint: the fitted and true lines nearly overlap.
- [ ] Experiment: raise `lr` until training diverges (the loss grows instead of shrinking). Note the threshold. Real training runs fail in exactly this way.

<details><summary>Hints</summary>

- Set `np.random.seed(42)` at the top so that runs are reproducible while debugging.
- Partials of a two-argument function, same trick as before: `dw = deriv(lambda v: mse(v, b), w)` and `db = deriv(lambda v: mse(w, v), b)`. Compute both *before* updating either: updating `w` first would change the landscape that `b`'s partial was measured on.
- If the loss decreases and then plateaus far above 4, run more steps or nudge `lr` up a little; `b` typically converges more slowly than `w` here.
- If the loss is `nan`, the learning rate is too big. This failure is familiar by now.

</details>

**Definition of done:** the training curve descends and flattens, the recovered `w` and `b` are close to 2.5 and -1.0, and the fitted line visually matches the data. The result is a trained model; lesson 12 does this same task with exact gradients and proper vocabulary.

## Stretch goals

- **Momentum:** keep a running velocity `v = 0.9*v - lr*slope; x = x + v` and re-run the 2D bowl. Compare the path plots: momentum smooths the zig-zag. This is a real optimizer trick that comes back in PyTorch.
- **Non-convex terrain:** minimize `f(x) = x**4 - 3*x**2 + x`, which has two valleys. Start descent from `x = -2` and from `x = +2`; show each run getting trapped in a different **local minimum**. Plot both paths on the curve.
- **Fit a parabola:** extend Project 3 to `y = a*x**2 + b*x + c` (three parameters, three partials). Generate matching fake data and recover all three.
- **3D surface plot:** render the Project 2 bowl with `plot_surface` (search matplotlib's docs for "mplot3d") and draw the descent path in 3D.

## Getting unstuck

- **Derivative checks failing?** Print `h`, `f(x+h)`, `f(x)`, and the raw difference. Nine times out of ten it is a too-big or too-small `h`, or a mix-up between degrees and radians.
- **Descent going the wrong way (uphill)?** The code is probably *adding* the gradient instead of subtracting it, or there is a sign error in `f`. Print the first three `(x, slope, new_x)` triples and sanity-check them by hand: a negative slope should move `x` right.
- **`nan` or `inf` anywhere?** The learning rate is too big. Cut it by 10x and retry before hunting for other bugs.
- **Plots look wrong?** Plot fewer things: the curve alone, then the points alone, then together.
- Standing advice: read error tracebacks from the bottom up, because the last line names the actual problem; print intermediate values and array shapes liberally; when truly stuck, ask an AI assistant for a **hint, not a solution** ("why might my gradient descent oscillate?", not "write it for me"); and type every line of code by hand, because typing the code is part of the learning.

## Resources

- **3Blue1Brown — Essence of Calculus** (search YouTube for "3Blue1Brown Essence of Calculus"): a very visual explanation of derivatives and the chain rule; watch episodes 1-4 alongside Project 1.
- **Khan Academy — Differential calculus** (free): https://www.khanacademy.org/math/differential-calculus — optional practice problems to solidify the slope idea; skim them rather than working through them all, since this curriculum stays numeric.

## Skills unlocked

- [ ] I can explain what a derivative is in one sentence, using the word "slope" and no formulas.
- [ ] I can compute the derivative of any Python function numerically and know how to pick `h`.
- [ ] I can compute a gradient (partial derivatives) of a multi-input function and say which direction it points.
- [ ] I verified the chain rule numerically and can state it in plain words.
- [ ] I can implement gradient descent from scratch in 1D and 2D and plot its path.
- [ ] I can recognize a too-big learning rate (divergence, nan) and a too-small one (crawling) from a loss curve.
- [ ] I can explain what a loss surface is and what "training a model" means geometrically.
- [ ] I fit `y = w*x + b` to noisy data with my own training loop and recovered the true parameters.

## Next up

Before the first real ML model comes one more math pillar, randomness, which the next lesson explores with dice, coins and Monte Carlo instead of formula sheets: **[10 · Probability and Statistics by Simulation](10-probability-statistics.md)**.

One last thing to carry forward: backpropagation, the famous algorithm inside every deep-learning framework (built from scratch in [lesson 22](../phase-3-deep-learning/22-micrograd-backpropagation.md)), is just the chain rule organized cleverly. This lesson has already covered its core idea.
