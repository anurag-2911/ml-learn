# 22 · Build micrograd: Backpropagation from Scratch (Karpathy 1)

**Phase 3 — Deep Learning** · Estimated time: 1-2 weeks · Prerequisites: [21 · Neural Networks: The Intuition](21-neural-networks-intuition.md), [09 · Calculus You Can Run: Gradient Descent](../phase-1-math/09-calculus-and-gradient-descent.md), [05 · NumPy](../phase-0-foundations/05-numpy.md)

> This is THE pivotal lesson of the whole curriculum. Everything before it was preparation; everything after it is application. You will build **micrograd**: a tiny *autograd engine* — a program that automatically computes gradients for any math expression — and then train a real neural network with it. This is exactly what PyTorch and TensorFlow do at their core, and you will write it in about 150 lines of Python. Once you have built backpropagation with your own hands, it stops being magic forever, and every deep learning tool you meet afterwards will feel familiar instead of mysterious.

## What you will build

- **Project 1:** Your own `micrograd` engine — a `Value` class with automatic backpropagation and an `MLP` (multi-layer perceptron) built on top of it — typed line by line alongside Karpathy's video.
- **Project 2:** An extended engine with `exp`, `log`, `sigmoid`, `relu` and division, every operation verified against numerical gradients by a `grad_check.py` script you write.
- **Project 3:** A trained micrograd MLP that solves XOR and the `make_moons` dataset, with a plot of the decision boundary — learned by an engine you built from nothing.

## Concepts you will learn by doing

- **Computational graph** — every math expression is secretly a graph of small operations feeding into each other.
- **The `Value` object** — one number (`data`), the gradient of the final output with respect to it (`grad`), and a little function that knows how to pass gradients backward (`_backward`).
- **The chain rule, operationally** — every node's gradient is just *local gradient × upstream gradient*. That one sentence is all of backpropagation.
- **Topological sort** — ordering the graph so `backward()` visits every node only after everything that depends on it.
- **Neurons, layers and MLPs** — a neural network is nothing more than `Value` objects composed into `Neuron` → `Layer` → `MLP`.
- **The training loop** — forward pass, loss, `backward()`, nudge weights, repeat — now running on YOUR engine.
- **Gradient checking** — using the numerical derivatives from lesson 09 to prove your analytic gradients are correct.

## Before you start

- You should be comfortable with: classes and `__init__` (lesson 04), what a derivative measures and the gradient descent loop (lesson 09), and why XOR needs a hidden layer (lesson 21). If any of those feel shaky, skim that lesson first — this one leans on all three.
- Activate your venv and install what you need:

```bash
cd ~/ml/ml-learn
source .venv/bin/activate
pip install numpy matplotlib scikit-learn
```

- Optional but nice: `graphviz` lets you draw the computational graph like Karpathy does in the video. Skip it if it gives you trouble — it is cosmetic.

```bash
sudo apt install graphviz
pip install graphviz
```

- Create your work folder:

```bash
mkdir -p work/22-micrograd-backpropagation
cd work/22-micrograd-backpropagation
```

- Queue up the video: it is the **first video** of the [Karpathy Zero to Hero playlist](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ), titled "The spelled-out intro to neural networks and backpropagation: building micrograd". It is about 2.5 hours long. **Do not watch it in one sitting.** Budget 3-5 sessions across several days, and expect each hour of video to take you 2-3 hours with pausing and typing.

**The one rule of this lesson: pause the video, type every line yourself, never copy-paste.** Your fingers are part of your memory. When your version breaks and the video's works, finding the difference IS the learning.

## Project 1 — Watch + build along: micrograd with Karpathy

**Goal:** Follow video 1 end to end and leave with a working `engine.py` (the `Value` class) and `nn.py` (`Neuron`, `Layer`, `MLP`) that can train a tiny neural network. Work in a Python file or a Jupyter notebook — whichever you prefer, but type everything.

**Milestones**

- [ ] **Warm-up: derivatives numerically.** Follow the video's opening: define `f(x) = 3*x**2 - 4*x + 5` and estimate its slope at `x = 3.0` by computing `(f(x + h) - f(x)) / h` with a tiny `h`. This is lesson 09 again, on purpose. Checkpoint: you should see a slope of about **14.0**.
- [ ] **A bare `Value`.** Create `engine.py` with a `Value` class holding `self.data`, plus `__repr__` so printing shows the number, plus `__add__` and `__mul__` so `Value(2.0) + Value(3.0)` and `a * b` work. Each result is a new `Value`. Checkpoint: `Value(2.0) * Value(-3.0) + Value(10.0)` prints a Value with `data=4.0`.
- [ ] **Remember the graph.** Give each `Value` a `_prev` (the child Values that produced it) and `_op` (the operation name). Now every result remembers where it came from — this chain of parents IS the computational graph. If you installed graphviz, build the little `draw_dot` helper from the video and look at your expression as an actual picture.
- [ ] **Backprop by hand, once.** Build the video's example: `a=2.0, b=-3.0, c=10.0; e=a*b; d=e+c; f=-2.0; L=d*f`. Add a `grad` field (start at 0) and fill in every node's gradient *manually*, using local gradient × upstream gradient at each step. A `+` node passes the upstream gradient through unchanged; a `*` node multiplies it by the *other* input's data. Checkpoint: `a.grad == 6.0` and `b.grad == -4.0`. Verify one of them by nudging the input by `h=0.001` and watching how much `L` moves.
- [ ] **`_backward` closures.** Now automate the previous step: inside `__add__` and `__mul__`, define a small function `_backward()` that pushes the output's `grad` into the inputs' `grad`s, and store it on the result. Two rules that will save you hours: always **`+=` into grads, never `=`** (a Value used twice must accumulate both contributions — try `b = a + a` to see why), and gradients flow from output to inputs.
- [ ] **`tanh` and a real neuron.** Add a `tanh()` method — you learned in lesson 21 why we need a squashing function. Its local gradient is `1 - t**2` where `t` is the output. Rebuild the video's two-input neuron `(x1*w1 + x2*w2 + b).tanh()` and backprop through it by calling the `_backward`s in reverse order by hand. Checkpoint: with the video's numbers, `w2.grad` is **0.0** (because `x2` is 0 — a weight on a dead input gets no gradient).
- [ ] **The real `backward()`.** Write the topological sort: a recursive walk that lists every node so each one appears after all nodes that depend on it, then set `self.grad = 1.0` and call each node's `_backward()` in reverse topological order. This is the moment micrograd becomes an autograd engine. Checkpoint: `L.backward()` reproduces every gradient from your by-hand pass, without you touching anything.
- [ ] **Python plumbing.** Make the engine pleasant: let `Value + 3` work (wrap plain numbers in `Value` inside `__add__`), and add `__radd__`/`__rmul__` so `3 * Value(2.0)` works too (Python calls these when the left operand is a plain number). Follow the video's segment on this.
- [ ] **`Neuron`, `Layer`, `MLP`.** In `nn.py`, build the three classes from the video: a `Neuron` holds random weight `Value`s and does `w·x + b` then `tanh`; a `Layer` is a list of Neurons; an `MLP` is a list of Layers. Checkpoint: `MLP(3, [4, 4, 1])` called on `[2.0, 3.0, -1.0]` returns a single `Value` strictly between -1 and 1.
- [ ] **Train it.** Use the video's tiny dataset — 4 input examples, targets `[1.0, -1.0, -1.0, 1.0]` — with mean squared error loss (from lesson 12: average of squared prediction errors). Write the loop you already know from lesson 09: forward → loss → **zero all grads** → `loss.backward()` → `p.data -= lr * p.grad` for every parameter. Checkpoint: loss falls below **0.05** within a few hundred steps, and the four predictions are close to their targets.
- [ ] **The zero-grad bug.** Karpathy deliberately shows the most common bug in deep learning: forgetting to reset grads to zero each step, so gradients pile up across iterations. Break your loop on purpose, watch the loss misbehave, fix it. You will meet this bug again in PyTorch (`optimizer.zero_grad()`) — now you know exactly what it is.

<details><summary>Hints</summary>

- If gradients come out doubled or wrong when a variable is reused (like `b = a + a`), you wrote `=` instead of `+=` somewhere in a `_backward`.
- If `backward()` misses some nodes, your topological sort probably isn't marking nodes as visited *before* recursing into children — trace it on a 3-node graph on paper.
- `tanh` backward in one line: `self.grad += (1 - t**2) * out.grad`. If your loss won't drop, print a few `.grad`s after `backward()` — all zeros means the graph is disconnected somewhere (often a plain float snuck in where a `Value` should be).
- Training diverges (loss grows)? Your learning rate is too big. Try 0.05, then adjust. You did this dance in lesson 09.

</details>

**Definition of done:** `python3 train_tiny.py` trains your MLP on the 4-example dataset, prints a falling loss ending below 0.05, and you wrote every line yourself.

## Project 2 — Make it YOURS: extend micrograd

**Goal:** Add `exp`, `log`, `sigmoid`, `relu` and division to your engine, each with a correct backward pass, and *prove* each one right with numerical gradient checking. Copying along with a video is one skill; extending the code alone is the real test of understanding — this project is where micrograd becomes yours.

**Milestones**

- [ ] **Write `grad_check.py` first.** A function that takes any scalar function built from `Value`s, computes gradients two ways — your `backward()`, and the centered numerical estimate `(f(x+h) - f(x-h)) / (2h)` from lesson 09 with `h=1e-6` — and reports the relative error. Starter shape:

```python
def numeric_grad(f, x, h=1e-6):
    """f: a plain-float function. Returns df/dx at x, estimated numerically."""
    return (f(x + h) - f(x - h)) / (2 * h)
```

- [ ] **`exp` and `log`.** Two classic derivatives: `d/dx e^x = e^x` and `d/dx ln(x) = 1/x`. Use `math.exp` and `math.log` for the forward pass. Checkpoint: gradient check passes with relative error below **1e-4** for each, at a few different input values.
- [ ] **`__pow__`, `__neg__`, `__sub__`, `__truediv__`.** Support `a ** 2` (power rule from lesson 09: `n * x**(n-1)`), then get subtraction and division nearly free: `a - b` is `a + (-b)`, and `a / b` is `a * b**-1`. This composition trick — new ops out of old ops — is how real frameworks stay small. Checkpoint: `(Value(4.0) / Value(2.0)).data == 2.0` and the gradient check passes.
- [ ] **`sigmoid`.** The squashing function from lesson 13 (logistic regression): `1 / (1 + e^-x)`, local gradient `s * (1 - s)`. Implement it directly OR compose it from `exp` and division and let backprop handle it automatically — do both and confirm they match. That comparison is the whole point of an autograd engine.
- [ ] **`relu`.** The simplest activation in deep learning: `max(0, x)` — it passes positives through and zeroes negatives. Gradient: 1 if the input was positive, else 0. Checkpoint: gradient check passes at `x=2.0` and `x=-2.0` (skip `x=0.0` — the derivative doesn't exist exactly there, and that's fine).
- [ ] **Torture test.** Gradient-check one ugly compound expression using ALL your new ops in one graph, with multiple inputs, including an input used twice. Checkpoint: every input's relative error below **1e-4**.

<details><summary>Hints</summary>

- Pattern for every new op: compute `out`, then define `_backward` as `self.grad += <local derivative> * out.grad`, then store it on `out`. If you can say the local derivative out loud, you can write the op.
- For ops where the output shows up in its own derivative (`exp`, `sigmoid`, `tanh`), use `out.data` inside `_backward` instead of recomputing.
- Relative error is `abs(analytic - numeric) / max(abs(analytic), abs(numeric), 1e-8)` — the `1e-8` avoids dividing by zero when both are ~0.
- Gradient check fails for `log` or division? Check your test inputs: `log` needs positive inputs, and division blows up near zero. Test at safely non-zero points like 0.5 or 3.0.

</details>

**Definition of done:** `python3 grad_check.py` checks every operation in your engine — old and new — and prints a pass (relative error < 1e-4) for each.

## Project 3 — Train on real shapes: XOR and moons

**Goal:** Use YOUR engine to solve the two classic problems: XOR from lesson 21 (which a single neuron provably cannot solve) and scikit-learn's `make_moons` (two interleaved crescent shapes that no straight line can separate). Then draw the curved decision boundary your engine learned.

**Milestones**

- [ ] **XOR, finally conquered.** Four inputs `[0,0],[0,1],[1,0],[1,1]`, targets `[-1, 1, 1, -1]` (tanh outputs live in -1..1, so use ±1 targets). Train an `MLP(2, [4, 1])` with MSE loss. Checkpoint: all four predictions have the **correct sign**, and loss is below 0.05. In lesson 21 this failed with a single perceptron — the hidden layer plus backprop is what fixed it.
- [ ] **Load moons.** In a new script, generate the dataset and look at it before training — always look at your data (lesson 07):

```python
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
X, y = make_moons(n_samples=100, noise=0.1, random_state=42)
y = y * 2 - 1   # convert labels {0,1} -> {-1,1} for tanh
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm"); plt.savefig("moons.png")
```

- [ ] **Train on moons.** `MLP(2, [16, 16, 1])`, MSE loss over all 100 points per step. Fair warning: your engine builds a Python object per number, so one step takes a second or two — that slowness is a *lesson*, and it is why lesson 23 moves to NumPy and lesson 24 to PyTorch. Print loss every 10 steps. Checkpoint: loss drops steadily; after ~100-300 steps, **accuracy above 0.90** (a prediction is correct when `sign(output) == label`).
- [ ] **Plot the decision boundary.** Build a grid of points covering the plot area (`np.meshgrid` from lesson 05, ~50×50 is plenty), run each grid point through your trained MLP, color by predicted sign, then scatter the real data on top. Checkpoint: the plot shows a **curved boundary snaking between the two crescents** — a shape learned, not programmed, by ~300 lines of your own Python.
- [ ] **Victory lap.** NOW go read [karpathy/micrograd](https://github.com/karpathy/micrograd) on GitHub — the actual repo, after building, not before. Compare his `engine.py` with yours. Note what he did differently (he uses `relu`, his engine is ~100 lines). Reading someone else's solution *after* writing your own is one of the fastest ways to level up.

<details><summary>Hints</summary>

- micrograd wants plain Python floats: convert each NumPy row with `[float(v) for v in row]` before feeding the MLP, or you'll get confusing NumPy-type errors.
- Moons stuck at ~0.85 accuracy? Train longer, try learning rate 0.1 with decay (e.g. cut it in half every 100 steps), or check that `noise` isn't set too high.
- For the boundary plot: `xx, yy = np.meshgrid(np.linspace(-2, 3, 50), np.linspace(-1.5, 2, 50))` gives the grid; predict each `(xx[i,j], yy[i,j])` in a double loop and store signs in a 2D array; draw with `plt.contourf(xx, yy, preds, alpha=0.3)` before the scatter.
- If XOR sometimes fails to train: your random init landed badly. Re-run with a different seed, or use a slightly larger hidden layer. Small nets on tiny data are seed-sensitive — that's normal and worth knowing.

</details>

**Definition of done:** Two saved plots — `moons.png` (raw data) and `boundary.png` (learned decision boundary over the data) — plus a training run reaching >0.90 accuracy on moons and correct signs on XOR, all powered by your engine.

## Stretch goals

- **Batching and speed:** time your moons training with `time python3 train_moons.py`, then find and optimize the slowest part (Python's `cProfile` can show you). How fast can you make pure-Python micrograd?
- **Better loss:** replace MSE with max-margin (hinge) loss like the micrograd repo's demo notebook uses, and add L2 regularization (a small penalty on weight sizes, from lesson 13). Does the boundary get smoother?
- **A real optimizer:** implement momentum (lesson 09's stretch goal) as an update rule on your parameters. Compare steps-to-convergence on moons with and without it.
- **Draw the graph of a whole MLP:** if you got graphviz working, render the full computational graph of a 2-neuron network's loss. Marvel at how big it already is — then remember GPT is the same idea with billions of nodes.

## If you get stuck

- **Wrong gradients are silent** — the code runs, the loss just won't drop. This is exactly why you built gradient checking; when training misbehaves, run `grad_check` on your ops before staring at the training loop.
- **Print the graph.** For small expressions, print each node's `data`, `grad` and `_op` after `backward()`. All-zero grads = disconnected graph (a float where a `Value` should be). Doubled grads = `=` instead of `+=`.
- **Shrink the problem.** Debug on `y = a * b + c`, not on a 3-layer MLP. Every bug you can have is already reproducible in a 3-node graph.
- Read error messages bottom-up — the last line names the error, the lines above name the place.
- Ask an AI assistant for a **hint, not a solution** ("why might all my grads be zero after backward()?"), and type every line yourself — the struggle is the point, doubly so in this lesson.

## Resources

- [Karpathy Zero to Hero playlist](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) — video 1, "The spelled-out intro to neural networks and backpropagation", is the spine of Project 1. The video description links his own notebooks.
- [karpathy/micrograd](https://github.com/karpathy/micrograd) — the reference implementation. Look ONLY after Project 3's victory lap; peeking early robs you of the lesson.

## Skills unlocked

- [ ] I can explain what a computational graph is and draw one for a small expression.
- [ ] I can state the chain rule as "local gradient × upstream gradient" and apply it by hand at a `+`, `*` and `tanh` node.
- [ ] I can explain why `backward()` needs a topological sort and why gradients must accumulate with `+=`.
- [ ] I can add a brand-new operation to an autograd engine and write its backward pass unaided.
- [ ] I can verify any gradient with numerical gradient checking and interpret the relative error.
- [ ] I can build a neuron, layer and MLP out of scalar Values and train them with a loop I wrote.
- [ ] I know what `zero_grad` is for, because I've watched training break without it.
- [ ] Backpropagation is no longer magic to me.

## Next up

Scalar-by-scalar autograd is beautiful but slow — next you'll rebuild the same ideas with fast NumPy matrices and read handwritten digits: [23 · Neural Network in Pure NumPy: MNIST Digits](23-mlp-numpy-mnist.md).
