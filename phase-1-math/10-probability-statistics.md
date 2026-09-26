# 10 · Probability and Statistics by Simulation

**Phase 1 — Math** · Estimated time: 1-2 weeks · Prerequisites: [05 · NumPy](../phase-0-foundations/05-numpy.md), [07 · Data Visualization](../phase-0-foundations/07-data-visualization.md)

> Machine learning is the science of making good decisions from noisy, random data — so probability is its native language. But you won't learn it the textbook way, with formulas and Greek letters first. You'll learn it the casino way: write a simulation, run it a million times, *see* what happens, and only then meet the formula that predicted it. By the end you'll know in your bones why more data means more certainty, why a 99%-accurate medical test can still be wrong 91% of the time, and why A/B tests with small samples lie to you. These intuitions sit underneath every model you'll ever train.

## What you will build

- **Project 1 — Casino simulator**: a script that flips coins and rolls dice by the million, plots the running average converging, estimates π by throwing random darts, and confirms the birthday paradox.
- **Project 2 — CLT machine**: a program that takes *any* weird distribution you feed it and shows a bell curve emerging from its sample means.
- **Project 3 — Bayes disease detector**: a simulation of 1,000,000 patients that reveals the shocking truth about "99% accurate" tests, verified against Bayes' formula.
- **Project 4 — A/B test simulator**: an experiment-runner that shows how often the *worse* of two buttons wins by pure luck, at different sample sizes.

## Concepts you will learn by doing

- **Probability as long-run frequency** — the probability of an event is the fraction of times it happens if you repeat the experiment forever (we'll settle for a million).
- **Random variable** — a number whose value depends on chance, like "the result of a die roll".
- **Expectation** — the long-run average value of a random variable.
- **Variance and standard deviation** — how spread out the values are around that average.
- **Distributions** — the shape of randomness: uniform (all outcomes equal), binomial (count of successes), normal (the bell curve). You'll see each one as a histogram, not a formula.
- **Law of large numbers** — averages of more samples wobble less and settle on the truth.
- **Central Limit Theorem** — averages of almost *anything* random form a bell curve. This is why the normal distribution is everywhere.
- **Bayes' rule** — how to correctly update a belief when new evidence arrives.
- **Sampling error** — why small samples lie, and how much data you need before you can trust a result.

## Before you start

You should be comfortable with NumPy arrays (lesson 05) and matplotlib histograms and line plots (lesson 07). No new math is assumed — that's the point.

Activate your venv at the repo root and make sure the tools are there (both should already be installed from earlier lessons):

```bash
cd ~/ml/ml-learn
source .venv/bin/activate
pip install numpy matplotlib
```

Create your work folder:

```bash
mkdir -p work/10-probability-statistics
cd work/10-probability-statistics
```

No datasets to download — you'll *generate* all your data with NumPy's random number generator. One habit to start now: always create a seeded generator so your "random" results are reproducible (you get the same numbers every run, which makes debugging possible):

```python
import numpy as np
rng = np.random.default_rng(seed=42)
print(rng.integers(0, 2, size=10))   # ten coin flips: 0 = tails, 1 = heads
```

## Project 1 — Casino simulator

**Goal:** Build `casino.py`, a collection of small simulations that make the law of large numbers visible: coin flips, dice rolls, Monte Carlo π, and the birthday paradox.

**Milestones**

- [ ] Simulate 100,000 coin flips as an array of 0s and 1s using `rng.integers(0, 2, size=100_000)`. Compute the fraction of heads. Checkpoint: you should see a value close to 0.5 but *not exactly* 0.5 — randomness has texture.
- [ ] Compute the **running average**: after flip 1, after flip 2, ... after flip 100,000. (Hint: `np.cumsum` gives cumulative sums; divide by `np.arange(1, n+1)`.) Plot it as a line with the x-axis on a log scale (`plt.xscale("log")`). Checkpoint: the line thrashes wildly at the start, then calms and hugs 0.5 — you have just *watched* the law of large numbers.
- [ ] Repeat the running-average plot for 100,000 rolls of a fair six-sided die. Before running it, write down your guess for where the line will settle. Checkpoint: it converges to 3.5 — the **expectation** of a die roll, the average of 1..6. Note that 3.5 is a value the die can never actually show; expectation is a long-run average, not a typical outcome.
- [ ] Plot a histogram of the die rolls themselves. Checkpoint: six roughly equal bars — this flat shape is the **uniform distribution**.
- [ ] Now count heads in groups: flip 100 coins, count the heads, repeat 10,000 times, histogram the counts. Checkpoint: a hump centered on 50, mostly between 40 and 60. This shape is the **binomial distribution** — the distribution of "number of successes in n tries".
- [ ] **Monte Carlo π**: throw random darts at the unit square by generating `x` and `y` arrays with `rng.random(1_000_000)`. A dart lands inside the quarter-circle when `x**2 + y**2 <= 1`. The fraction that land inside ≈ π/4 (ratio of the areas). Estimate π. Checkpoint: with 1,000,000 darts you should see roughly 3.14, typically within ±0.01 of the true value.
- [ ] Plot the π estimate as a function of the number of darts (running average again). Checkpoint: same converging shape as the coin plot — "more samples, less wobble" is a universal law, and it's why ML models want more data.
- [ ] **Birthday paradox**: write a function `has_shared_birthday(k)` that draws `k` random birthdays (`rng.integers(0, 365, size=k)`) and returns whether any two match (compare `len(np.unique(days))` to `k`). Run it 10,000 times for k=23 and take the mean. Checkpoint: about 0.507 — in a room of just 23 people, a shared birthday is more likely than not.
- [ ] Plot the shared-birthday probability for k = 2 to 60. Checkpoint: an S-shaped curve crossing 0.5 near k=23 and reaching ~0.99 by k=57. Your simulation just answered a question most people's intuition gets badly wrong.

<details><summary>Hints</summary>

- Running average in one line: `np.cumsum(flips) / np.arange(1, len(flips) + 1)`. Print the first five values by hand-checking against the flips to convince yourself it's right.
- For the binomial histogram, `rng.integers(0, 2, size=(10_000, 100))` gives 10,000 experiments of 100 flips at once; sum along `axis=1`. No Python loops needed.
- If your π estimate is near 0.785 instead of 3.14, you forgot to multiply by 4.
- For the birthday paradox you *do* need a Python loop over the 10,000 trials (each trial needs its own `unique` check) — that's fine, it still runs in a second or two.

</details>

**Definition of done:** `casino.py` runs top to bottom, produces the four plots (coin convergence, die histogram, π convergence, birthday curve), and each checkpoint number matches.

## Project 2 — CLT machine

**Goal:** Build `clt.py`, a program that demonstrates the Central Limit Theorem: sample means from *any* distribution — no matter how lumpy — pile up into a bell curve.

**Milestones**

- [ ] Invent a deliberately weird distribution. Ideas: `rng.random(n)**3` (squashed toward 0), or a mixture — 80% of values near 1 and 20% near 10 (use `np.where(rng.random(n) < 0.8, ...)`). Draw 100,000 values and histogram them. Checkpoint: the histogram looks nothing like a bell — lopsided, maybe two humps. That's the point.
- [ ] Write a function `sample_means(dist_fn, n, trials=10_000)` that: draws `n` values from your weird distribution, averages them, and repeats `trials` times, returning an array of 10,000 means. (Vectorization hint: draw a `(trials, n)` array and take `.mean(axis=1)`.)
- [ ] Plot histograms of `sample_means` for n = 1, 2, 5, 30 in a 2×2 grid of subplots (`plt.subplots(2, 2)`), all with the same x-limits so shapes are comparable. Checkpoint: n=1 is your weird shape; n=2 is less weird; n=5 has a hump; **n=30 is a clean, symmetric bell curve — regardless of what monstrosity you started from.** This is the Central Limit Theorem.
- [ ] Overlay reality on theory: on the n=30 histogram (use `density=True`), plot the **normal distribution** curve with mean = your distribution's mean and standard deviation = your distribution's std divided by √30. The formula for the bell curve is `np.exp(-(x - mu)**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))`. Checkpoint: the curve traces the top of your histogram bars. You simulated first; the formula merely *predicted* what you already saw.
- [ ] Measure the shrinkage: print the standard deviation of the sample means for each n. Checkpoint: it shrinks roughly like 1/√n — quadruple the sample size, halve the noise. Remember this ratio; it is the price tag of certainty in all of statistics.
- [ ] Swap in a second, completely different weird distribution and rerun everything. Checkpoint: same bell at n=30. The CLT doesn't care what you feed it — which is exactly why the normal distribution shows up all over nature and ML.

<details><summary>Hints</summary>

- A distribution function you can swap out easily: define `def weird(size): return rng.random(size)**3` and pass the *function* into `sample_means`, calling `dist_fn((trials, n))` inside.
- If all four histograms look identical, check that you're averaging over `axis=1` (within each trial), not `axis=0`.
- Use `bins=50` and shared `range=` in all four histograms; mismatched bins can fake or hide the effect.

</details>

**Definition of done:** `clt.py` produces the 2×2 grid showing the bell emerge, the theory curve matches the n=30 histogram, and it works for at least two different source distributions.

## Project 3 — Bayes disease detector

**Goal:** Build `bayes.py`: simulate a rare disease and a good test, discover that most positive results are false alarms, then verify the surprise with Bayes' rule.

The setup: a disease affects **1 in 1,000** people. A test is **99% accurate** both ways (99% of sick people test positive, 99% of healthy people test negative). You test positive. How worried should you be? Almost everyone — including many doctors — guesses "99% chance I'm sick". Simulate it and find out.

**Milestones**

- [ ] Simulate 1,000,000 people as a boolean array: `sick = rng.random(1_000_000) < 0.001`. Checkpoint: `sick.sum()` is around 1,000.
- [ ] Simulate the test: sick people test positive with probability 0.99, healthy people test positive with probability 0.01 (the 1% false-alarm rate). Build a boolean `positive` array using `np.where` or boolean logic with two fresh `rng.random` draws.
- [ ] Count the four groups: sick+positive (true positives), healthy+positive (false positives), sick+negative, healthy+negative. Print them in a little 2×2 table. Checkpoint: roughly 990 true positives and roughly 9,990 false positives — the false alarms **outnumber** the real cases about 10 to 1.
- [ ] Answer the question: among people who tested *positive*, what fraction are actually sick? Compute `(sick & positive).sum() / positive.sum()`. Checkpoint: about **0.09 — just 9%**. A positive result on a "99% accurate" test means you are still probably healthy. Sit with that for a minute.
- [ ] Explain it in one comment in your code, using **natural frequencies** (counts of people, not probabilities): "Out of a million people, ~1,000 are sick and ~999,000 are healthy. The test flags ~990 of the sick — but also ~9,990 of the healthy, because 1% of a huge number is still a big number. So of ~10,980 positives, only ~990 are real."
- [ ] Now the formula. **Bayes' rule** computes P(sick | positive) — the probability of being sick *given* a positive test: `P(sick|pos) = P(pos|sick) * P(sick) / P(pos)`, where `P(pos) = P(pos|sick)*P(sick) + P(pos|healthy)*P(healthy)`. Plug in 0.99, 0.001, 0.01, 0.999 and compute it in code. Checkpoint: 0.0902... — matching your simulation to two decimal places. Simulation first, formula second: the formula is just the simulation done with algebra.
- [ ] Explore: wrap it in a function of `disease_rate` and plot P(sick | positive) for disease rates from 1-in-10 to 1-in-100,000 (log x-axis). Checkpoint: the curve shows the answer depends enormously on how rare the disease is — the **prior**, your belief before seeing the evidence, matters as much as the test.

<details><summary>Hints</summary>

- One clean way to build the test: `positive = np.where(sick, rng.random(n) < 0.99, rng.random(n) < 0.01)`.
- If your answer comes out near 99%, you probably computed the fraction of *sick* people who test positive (that IS 99%) instead of the fraction of *positive* people who are sick. The whole lesson lives in that difference of direction.
- Sanity check: the four group counts must sum to exactly 1,000,000.

</details>

**Definition of done:** `bayes.py` prints the 2×2 table, the simulated ~9% figure, and the Bayes-formula figure, and they agree. You can explain the result out loud using counts of people. (This exact reasoning powers the spam filter you'll build in [lesson 17](../phase-2-classical-ml/17-naive-bayes-spam-svm.md).)

## Project 4 — A/B test simulator

**Goal:** Build `ab_test.py`: two versions of a signup button, where B is *truly* better (11% click rate vs 10%) — and discover how often A wins the experiment anyway when the sample is small.

**Milestones**

- [ ] Write `run_experiment(n)`: show button A to `n` visitors (each clicks with probability 0.10) and button B to `n` visitors (probability 0.11), using `rng.random(n) < rate`. Return both observed click rates. Checkpoint: run it once with n=100 — the observed rates will often be nowhere near 10% and 11% (you might see 6% vs 14%, or A ahead of B).
- [ ] Write `prob_wrong_winner(n, trials=2_000)`: run the experiment `trials` times and return the fraction where A's observed rate is ≥ B's — i.e., where the truly-worse button ties or wins. Checkpoint: for n=100, the worse button wins or ties roughly **40% of the time**. An A/B test with 100 visitors per side is barely better than a coin flip.
- [ ] Evaluate n = 100, 500, 1,000, 5,000, 10,000, 50,000 and plot wrong-winner probability vs n (log x-axis). Checkpoint: a falling curve — around 40% at n=100, roughly 20-25% at n=1,000, about **1% at n=10,000**. You need tens of thousands of visitors to reliably detect a 1-percentage-point improvement.
- [ ] Connect it to the CLT: each observed click rate is a *sample mean* (mean of 0/1 clicks), so by Project 2 it's bell-shaped with spread shrinking like 1/√n. Small n → wide bells → the two bells overlap → the worse button often lands higher by luck. Visualize this: histogram A's and B's observed rates over 2,000 trials, on the same axes, for n=100 and again for n=10,000. Checkpoint: at n=100 the two histograms overlap almost completely; at n=10,000 they are two nearly-separate spikes.
- [ ] Explore the effect gap: repeat the wrong-winner curve with B's true rate at 0.15 instead of 0.11. Checkpoint: far fewer samples needed — big effects are easy to detect, tiny effects need huge samples. This trade-off (effect size vs sample size) is exactly what you'll wrestle with when comparing *models* in [lesson 14](../phase-2-classical-ml/14-model-evaluation.md): a model that scores 1% better on a small test set may not be better at all.

<details><summary>Hints</summary>

- Vectorize the trials: `clicks_a = (rng.random((trials, n)) < 0.10).mean(axis=1)` gives all 2,000 observed rates for A in one line. This is fine up to about n=10,000 (a 2,000 × 10,000 float array is ~160 MB), but at n=50,000 it balloons to ~800 MB. There, either loop over chunks of ~200 trials, or — better — let NumPy count the clicks directly: `rng.binomial(n, 0.10, size=trials) / n` draws all 2,000 observed rates with negligible memory. Same distribution, no giant array.
- "Wins or ties" is `(rates_a >= rates_b).mean()`. Decide the tie rule and keep it consistent, or your numbers will drift from the checkpoints.
- Your exact percentages will differ by a point or two from the checkpoints — that's sampling error, the very thing you're studying. Rerun with a different seed and watch them wiggle.

</details>

**Definition of done:** `ab_test.py` produces the falling wrong-winner curve and the overlapping-histograms plot, and you can explain to an imaginary product manager why "B won our 200-visitor test" means almost nothing.

## Stretch goals

- **Gambler's ruin**: a gambler starts with $50 and bets $1 on a fair coin flip until reaching $0 or $100. Simulate thousands of gamblers; plot a few of their bankroll paths and the distribution of how long they survive.
- **Monty Hall**: simulate the famous three-doors game-show problem and confirm that switching doors wins 2/3 of the time. Prepare to distrust your intuition again.
- **Bootstrap confidence intervals**: given one sample of 100 values, resample from it with replacement 10,000 times (`rng.choice`) to estimate how uncertain its mean is — a simulation trick real statisticians use daily, and no formula required.
- **Estimate anything**: use Monte Carlo to answer a question with no clean formula, e.g. "if I roll 10 dice, what's the probability the sum exceeds 45?" Then try to verify it a second, independent way.

## If you get stuck

- **Your numbers don't exactly match the checkpoints.** They shouldn't — every checkpoint here is a long-run value and your run is finite. Close (a percent or two) is right; wildly off means a bug. Increase the sample size: real effects stabilize, bugs don't.
- **Results change every run.** You forgot the seed. Create `rng = np.random.default_rng(seed=42)` once at the top and use only that `rng` everywhere — never `np.random.something` directly.
- **A probability comes out above 1 or below 0.** You divided by the wrong count. Print the numerator and denominator separately and ask "counts of *what*?" — most Bayes bugs are a right number divided by the wrong total.
- **Boolean array confusion.** `sick & positive` needs `&` (not `and`) and parentheses around comparisons: `(x > 0) & (y > 0)`. `.sum()` on booleans counts the `True`s — that one idiom powers half this lesson.
- Standing advice: read error messages bottom-up (the last line names the actual problem); print `.shape`, `.dtype` and a few values of every array you're unsure about; ask an AI assistant for a **hint**, not a solution; and type all code yourself — muscle memory is how this sticks.

## Resources

- [Seeing Theory](https://seeing-theory.brown.edu) — a beautiful interactive visual introduction; the Basic Probability and Central Limit Theorem chapters mirror Projects 1 and 2. Explore it *after* building your own versions.
- **StatQuest with Josh Starmer** — search YouTube for this channel: short, extremely clear videos. Watch "The Central Limit Theorem" after Project 2 and "Bayes' Theorem" after Project 3 to hear the ideas in a second voice.
- [NumPy random generator docs](https://numpy.org) — search the site for "random Generator" for everything `rng` can draw: integers, uniforms, normals, choices, shuffles.

## Skills unlocked

- [ ] I can simulate a random process in NumPy and use the long-run frequency to estimate a probability.
- [ ] I can explain expectation and variance as "long-run average" and "spread", and compute both from simulated data.
- [ ] I can recognize uniform, binomial and normal distributions by their histogram shapes.
- [ ] I can explain the law of large numbers and the Central Limit Theorem from plots I made myself, including why sample-mean noise shrinks like 1/√n.
- [ ] I can walk through Bayes' rule using natural frequencies and explain why a positive result on an accurate test for a rare condition is usually a false alarm.
- [ ] I can explain why a small A/B test (or a small ML test set) often crowns the wrong winner, and roughly how sample size fixes it.
- [ ] I seed my random generators and treat "results wiggle between runs" as information, not annoyance.

## Next up

You now have all the math you need to train your first real model — in the next lesson you build a classifier from scratch and *measure* how well it works: [11 · Your First ML Model: k-Nearest Neighbors from Scratch](../phase-2-classical-ml/11-first-model-knn.md).
