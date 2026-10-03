# 22 · Build micrograd: Backpropagation from Scratch (Karpathy 1)

**Phase 3 — Deep Learning** · Estimated time: 1-2 weeks · Prerequisites: [21 · Neural Networks: The Intuition](21-neural-networks-intuition.md), [09 · Calculus You Can Run: Gradient Descent](../phase-1-math/09-calculus-and-gradient-descent.md), [05 · NumPy](../phase-0-foundations/05-numpy.md)

> This is the pivotal lesson of the whole curriculum: everything before it is preparation, and everything after it is application. It builds **micrograd**, a tiny *autograd engine* (a program that automatically computes gradients for any math expression), and then uses it to train a real neural network. PyTorch and TensorFlow do exactly this at their core, and micrograd does it in about 150 lines of Python. Once backpropagation has been built by hand, it is never a black box again, and every deep learning tool that comes later feels familiar instead of mysterious.

## What this lesson builds

- **Project 1:** A `micrograd` engine, typed line by line alongside Karpathy's video. It consists of a `Value` class with automatic backpropagation and an `MLP` (multi-layer perceptron) built on top of it.
- **Project 2:** An extended engine with `log`, `sigmoid` and `relu`, plus a `grad_check.py` script that verifies every operation against numerical gradients.
- **Project 3:** A trained micrograd MLP that solves XOR and the `make_moons` dataset, with a plot of the decision boundary, learned by an engine built from nothing.

## Concepts covered

- **Computational graph**: every math expression is really a graph of small operations feeding into each other.
- **The `Value` object**: one number (`data`), the gradient of the final output with respect to it (`grad`), and a little function that knows how to pass gradients backward (`_backward`).
- **The chain rule, operationally**: every node's gradient is just *local gradient × upstream gradient*. That one sentence is all of backpropagation.
- **Topological sort**: ordering the graph so that `backward()` visits every node only after everything that depends on it.
- **Neurons, layers and MLPs**: a neural network is nothing more than `Value` objects composed into `Neuron` → `Layer` → `MLP`.
- **The training loop**: forward pass, loss, `backward()`, nudge weights, repeat. Now it runs on the engine built in this lesson.
- **Gradient checking**: using the numerical derivatives from lesson 09 to prove that the engine's analytic gradients are correct.

## Before starting

- Make sure these three earlier topics are solid: classes and `__init__` (lesson 04), what a derivative measures and the gradient descent loop (lesson 09), and why XOR needs a hidden layer (lesson 21). If any of them feels shaky, skim that lesson first, because this one leans on all three.
- Activate the venv and install the required packages:

```bash
cd ~/ml/ml-learn
source .venv/bin/activate
pip install numpy matplotlib scikit-learn
```

- Optional but useful: `graphviz` draws the computational graph the way Karpathy does in the video. It is cosmetic, so skip it if it causes trouble. It comes in two parts: the Graphviz program, which does the drawing, and a Python package that talks to it. Install the program first. **macOS:** on an Apple Silicon Mac with Homebrew, run `brew install graphviz`. Macs without Homebrew (Intel Macs, and macOS 14 or older) skip graphviz altogether; nothing else in this lesson needs it. **Windows (WSL2) and Linux:** run `sudo apt install -y graphviz` (on Fedora, `sudo dnf install -y graphviz`). Then install the Python package:

```bash
pip install graphviz
```

- Create the work folder for this lesson:

```bash
mkdir -p work/22-micrograd-backpropagation
cd work/22-micrograd-backpropagation
```

- Queue up the video: it is the **first video** of the [Karpathy Zero to Hero playlist](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ), titled "The spelled-out intro to neural networks and backpropagation: building micrograd". It is about 2.5 hours long. **Do not watch it in one sitting.** Budget 3-5 sessions across several days, and expect each hour of video to take 2-3 hours with pausing and typing.

**The one rule of this lesson: pause the video, type every line by hand, never copy-paste.** Typing the code helps it stick in memory. When the typed version breaks and the video's version works, finding the difference *is* the learning.

## Project 1 — Watch + build along: micrograd with Karpathy

**Goal:** Follow video 1 end to end and finish with a working `engine.py` (the `Value` class) and `nn.py` (`Neuron`, `Layer`, `MLP`) that can train a tiny neural network. Work in a Python file or a Jupyter notebook (either is fine), but type everything.

**Milestones**

- [ ] **Warm-up: derivatives numerically.** Follow the video's opening: define `f(x) = 3*x**2 - 4*x + 5` and estimate its slope at `x = 3.0` by computing `(f(x + h) - f(x)) / h` with a tiny `h`. This is lesson 09 again, on purpose. Checkpoint: the slope comes out at about **14.0**.
- [ ] **A bare `Value`.** Create `engine.py` with a `Value` class holding `self.data`, plus `__repr__` so printing shows the number, plus `__add__` and `__mul__` so `Value(2.0) + Value(3.0)` and `a * b` work. Each result is a new `Value`. Checkpoint: `Value(2.0) * Value(-3.0) + Value(10.0)` prints a Value with `data=4.0`.
- [ ] **Remember the graph.** Give each `Value` a `_prev` (the child Values that produced it) and `_op` (the operation name). Now every result remembers where it came from. This chain of parents *is* the computational graph. If graphviz is installed, build the little `draw_dot` helper from the video and look at the expression as an actual picture. A notebook displays the returned graph by itself; in a plain `.py` file, save it with `draw_dot(L).render('graph', format='svg')` and open the resulting `graph.svg`.
- [ ] **Backprop by hand, once.** Build the video's example: `a=2.0, b=-3.0, c=10.0; e=a*b; d=e+c; f=-2.0; L=d*f`. Add a `grad` field (start at 0) and fill in every node's gradient *manually*, using local gradient × upstream gradient at each step. A `+` node passes the upstream gradient through unchanged; a `*` node multiplies it by the *other* input's data. Checkpoint: `a.grad == 6.0` and `b.grad == -4.0`. Verify one of them by nudging the input by `h=0.001` and watching how much `L` moves.
- [ ] **`_backward` closures.** Now automate the previous step: inside `__add__` and `__mul__`, define a small function `_backward()` that pushes the output's `grad` into the inputs' `grad`s, and store it on the result. Two rules that save hours: always **`+=` into grads, never `=`** (a Value used twice must accumulate both contributions; try `b = a + a` to see why), and gradients flow from output to inputs.
- [ ] **`tanh` and a real neuron.** Add a `tanh()` method. Lesson 21 showed why a squashing function is needed. Its local gradient is `1 - t**2`, where `t` is the output. Rebuild the video's two-input neuron `(x1*w1 + x2*w2 + b).tanh()` and backprop through it by calling the `_backward`s in reverse order by hand. Checkpoint: with the video's numbers, `w2.grad` is **0.0** (because `x2` is 0, and a weight on a dead input gets no gradient).
- [ ] **The real `backward()`.** Write the topological sort: a recursive walk that lists every node so each one appears after all the nodes it depends on (its inputs), then set `self.grad = 1.0` and call each node's `_backward()` in reverse topological order, which starts at the output. This is the moment micrograd becomes an autograd engine. Checkpoint: `L.backward()` reproduces every gradient from the by-hand pass, with no manual steps.
- [ ] **Python plumbing.** Make the engine pleasant to use: let `Value + 3` work (wrap plain numbers in `Value` inside `__add__`), and add `__radd__`/`__rmul__` so `3 * Value(2.0)` works too (Python calls these when the left operand is a plain number). Follow the video's segment on this.
- [ ] **`Neuron`, `Layer`, `MLP`.** In `nn.py`, build the three classes from the video: a `Neuron` holds random weight `Value`s and does `w·x + b` then `tanh`; a `Layer` is a list of Neurons; an `MLP` is a list of Layers. Checkpoint: `MLP(3, [4, 4, 1])` called on `[2.0, 3.0, -1.0]` returns a single `Value` strictly between -1 and 1.
- [ ] **Train it.** Create `train_tiny.py`, which imports `MLP` from `nn.py`, and use the video's tiny dataset (4 input examples, targets `[1.0, -1.0, -1.0, 1.0]`) with mean squared error loss (from lesson 12: the average of squared prediction errors). Write the familiar loop from lesson 09: forward → loss → **zero all grads** → `loss.backward()` → `p.data -= lr * p.grad` for every parameter. Checkpoint: loss falls below **0.05** within a few hundred steps, and the four predictions are close to their targets.
- [ ] **The zero-grad bug.** Near the end of the video, Karpathy finds a bug he made by accident, and it is the most common bug in deep learning: forgetting to reset grads to zero each step, so gradients pile up across iterations. Break the loop on purpose and compare it with the fixed loop. On this tiny dataset the buggy loop usually still drives the loss down, often faster, because the piled-up gradients act like a huge learning rate; the bug hides on easy problems and breaks training on harder ones. Then fix it. The same bug comes back in PyTorch (`optimizer.zero_grad()`), and this step shows exactly what it is.

<details><summary>Hints</summary>

- If gradients come out doubled or wrong when a variable is reused (like `b = a + a`), a `_backward` somewhere uses `=` instead of `+=`.
- If inputs end up with zero grads after `backward()`, check the topological sort: `topo.append(v)` must come *after* the loop that recurses into `v._prev`, so each node lands after its inputs. If grads come out wrong (usually too large) on a graph that reuses a node, the `visited` check is missing and that node is processed twice. Trace it on a 3-node graph on paper.
- `tanh` backward in one line: `self.grad += (1 - t**2) * out.grad`. If the loss will not drop, print a few `.grad`s after `backward()`. All zeros means the graph is disconnected somewhere (often a plain float slipped in where a `Value` should be).
- Training diverges (loss grows)? The learning rate is too big. Try 0.05, then adjust, as in lesson 09.

</details>

**Definition of done:** `python3 train_tiny.py` trains the MLP on the 4-example dataset and prints a falling loss ending below 0.05. Every line of the code was typed by hand.

## Project 2 — Own the engine: extend micrograd

**Goal:** Add `log`, `sigmoid` and `relu` to the engine, each with a correct backward pass, and *prove* every operation right with numerical gradient checking. That includes `exp`, `**`, subtraction and division, which the video already built in Project 1: for those, re-derive each backward pass without looking at the video and let the gradient check confirm it. Copying along with a video is one skill; extending the code alone is the real test of understanding. This project is where micrograd becomes the learner's own.

**Milestones**

- [ ] **Write `grad_check.py` first.** A function that takes any scalar function built from `Value`s, computes gradients two ways (the engine's `backward()`, and the centered numerical estimate `(f(x+h) - f(x-h)) / (2h)` from lesson 09 with `h=1e-6`), and reports the relative error. Starter shape:

```python
def numeric_grad(f, x, h=1e-6):
    """f: a plain-float function. Returns df/dx at x, estimated numerically."""
    return (f(x + h) - f(x - h)) / (2 * h)
```

- [ ] **`exp` and `log`.** Two classic derivatives: `d/dx e^x = e^x` and `d/dx ln(x) = 1/x`. Use `math.exp` and `math.log` for the forward pass. Checkpoint: the gradient check passes with relative error below **1e-4** for each, at a few different input values.
- [ ] **`__pow__`, `__neg__`, `__sub__`, `__truediv__`.** Support `a ** 2` (power rule: the derivative of `x**n` is `n * x**(n-1)`; for `n = 2` that is the `2x` checked in lesson 09), then get subtraction and division nearly free: `a - b` is `a + (-b)`, and `a / b` is `a * b**-1`. This composition trick (new ops out of old ops) is how real frameworks stay small. Checkpoint: `(Value(4.0) / Value(2.0)).data == 2.0` and the gradient check passes.
- [ ] **`sigmoid`.** The squashing function from lesson 13 (logistic regression): `1 / (1 + e^-x)`, local gradient `s * (1 - s)`. Implement it directly, or compose it from `exp` and division and let backprop handle it automatically. Do both and confirm that they match. That comparison is the whole point of an autograd engine.
- [ ] **`relu`.** The simplest activation in deep learning: `max(0, x)`, which passes positives through and zeroes negatives. Gradient: 1 if the input was positive, else 0. Checkpoint: the gradient check passes at `x=2.0` and `x=-2.0` (skip `x=0.0`: the derivative does not exist exactly there, and that is fine).
- [ ] **Torture test.** Gradient-check one ugly compound expression that uses *all* the new ops in one graph, with multiple inputs, including an input used twice. Checkpoint: every input's relative error is below **1e-4**.

<details><summary>Hints</summary>

- Pattern for every new op: compute `out`, then define `_backward` as `self.grad += <local derivative> * out.grad`, then store it on `out`. Being able to say the local derivative out loud is enough to write the op.
- For ops where the output shows up in its own derivative (`exp`, `sigmoid`, `tanh`), use `out.data` inside `_backward` instead of recomputing.
- Relative error is `abs(analytic - numeric) / max(abs(analytic), abs(numeric), 1e-8)`. The `1e-8` avoids dividing by zero when both are ~0.
- Gradient check fails for `log` or division? Check the test inputs: `log` needs positive inputs, and division blows up near zero. Test at safely non-zero points like 0.5 or 3.0.

</details>

**Definition of done:** `python3 grad_check.py` checks every operation in the engine, old and new, and prints a pass (relative error < 1e-4) for each.

## Project 3 — Train on real shapes: XOR and moons

**Goal:** Use the home-built engine to solve the two classic problems: XOR from lesson 21 (which a single neuron provably cannot solve) and scikit-learn's `make_moons` (two interleaved crescent shapes that no straight line can separate). Then draw the curved decision boundary that the engine learned.

**Milestones**

- [ ] **XOR, now with backprop.** Four inputs `[0,0],[0,1],[1,0],[1,1]`, targets `[-1, 1, 1, -1]` (tanh outputs live in -1..1, so use ±1 targets). Train an `MLP(2, [4, 1])` with MSE loss. Checkpoint: all four predictions have the **correct sign**, and loss is below 0.05. In lesson 21 a single perceptron failed at this, and the hidden-layer network that solved it needed slow numeric gradients, one nudge per parameter; here one `backward()` call computes every gradient.
- [ ] **Load moons.** In a new script, `train_moons.py`, generate the dataset and look at it before training; always look at the data (lesson 07):

```python
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
X, y = make_moons(n_samples=100, noise=0.1, random_state=42)
y = y * 2 - 1   # convert labels {0,1} -> {-1,1} for tanh
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm"); plt.savefig("moons.png")
```

- [ ] **Train on moons.** `MLP(2, [16, 16, 1])`, MSE loss over all 100 points per step. A warning about speed: the engine builds a Python object per number, so one step takes a second or two. That slowness is a *lesson*, and it is why lesson 23 moves to NumPy and lesson 24 to PyTorch. Print loss every 10 steps. Checkpoint: loss drops steadily; after ~100-300 steps, **accuracy above 0.90** (a prediction is correct when `sign(output) == label`).
- [ ] **Plot the decision boundary.** Build a grid of points covering the plot area (`np.meshgrid`, as in the decision-boundary plots of lessons 13 and 21; ~50×50 is plenty), run each grid point through the trained MLP, color by predicted sign, then scatter the real data on top. Checkpoint: the plot shows a **curved boundary snaking between the two crescents**: a shape learned, not programmed, by ~300 lines of hand-written Python.
- [ ] **Last step: read the original.** Now, after building and not before, go to [karpathy/micrograd](https://github.com/karpathy/micrograd) on GitHub and read the actual repo. Compare his `engine.py` with the home-built one. Note what he did differently (he uses `relu`, his engine is ~100 lines). Reading someone else's solution *after* solving the problem is one of the fastest ways to improve.

<details><summary>Hints</summary>

- micrograd wants plain Python floats: convert each NumPy row with `[float(v) for v in row]` before feeding the MLP. Otherwise, confusing NumPy-type errors appear.
- Moons stuck at ~0.85 accuracy? Train longer, try learning rate 0.1 with decay (e.g. cut it in half every 100 steps), or check that `noise` is not set too high.
- For the boundary plot: `xx, yy = np.meshgrid(np.linspace(-2, 3, 50), np.linspace(-1.5, 2, 50))` gives the grid; predict each `(xx[i,j], yy[i,j])` in a double loop and store signs in a 2D array; draw with `plt.contourf(xx, yy, preds, alpha=0.3)` before the scatter.
- If XOR sometimes fails to train, the random init landed badly. Re-run with a different seed, or use a slightly larger hidden layer. Small nets on tiny data are seed-sensitive; that is normal and worth knowing.

</details>

**Definition of done:** Two saved plots, `moons.png` (raw data) and `boundary.png` (learned decision boundary over the data), plus a training run reaching >0.90 accuracy on moons and correct signs on XOR, all powered by the home-built engine.

## Stretch goals

- **Batching and speed:** time the moons training with `time python3 train_moons.py`, then find and optimize the slowest part (Python's `cProfile` can show it). How fast can pure-Python micrograd get?
- **Better loss:** replace MSE with max-margin (hinge) loss like the micrograd repo's demo notebook uses, and add L2 regularization (a small penalty on weight sizes, the idea behind Ridge in lesson 19). Does the boundary get smoother?
- **A real optimizer:** implement momentum (lesson 09's stretch goal) as an update rule on the parameters. Compare steps-to-convergence on moons with and without it.
- **Draw the graph of a whole MLP:** if graphviz is working, render the full computational graph of a 2-neuron network's loss. Notice how big it already is, and remember that GPT is the same idea with billions of nodes.

## Getting unstuck

- **Wrong gradients are silent.** The code runs; the loss just will not drop. This is exactly what gradient checking is for: when training misbehaves, run `grad_check` on the ops before staring at the training loop.
- **Print the graph.** For small expressions, print each node's `data`, `grad` and `_op` after `backward()`. All-zero grads = disconnected graph (a float where a `Value` should be). Doubled grads = `=` instead of `+=`.
- **Shrink the problem.** Debug on `y = a * b + c`, not on a 3-layer MLP. Every possible bug is already reproducible in a 3-node graph.
- Read error messages bottom-up: the last line names the error, and the lines above name the place.
- Ask an AI assistant for a **hint, not a solution** ("why might all my grads be zero after backward()?"), and type every line by hand. The struggle is the point, doubly so in this lesson.

## Resources

- [Karpathy Zero to Hero playlist](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) — video 1, "The spelled-out intro to neural networks and backpropagation", is the spine of Project 1. The video description links his own notebooks.
- [karpathy/micrograd](https://github.com/karpathy/micrograd) — the reference implementation. Look at it only after reaching the last step of Project 3; peeking early spoils the lesson.

## Skills unlocked

- [ ] I can explain what a computational graph is and draw one for a small expression.
- [ ] I can state the chain rule as "local gradient × upstream gradient" and apply it by hand at a `+`, `*` and `tanh` node.
- [ ] I can explain why `backward()` needs a topological sort and why gradients must accumulate with `+=`.
- [ ] I can add a brand-new operation to an autograd engine and write its backward pass unaided.
- [ ] I can verify any gradient with numerical gradient checking and interpret the relative error.
- [ ] I can build a neuron, layer and MLP out of scalar Values and train them with a loop I wrote.
- [ ] I know what `zero_grad` is for, because I have seen what happens without it.
- [ ] Backpropagation is no longer a mystery to me.

## Next up

Scalar-by-scalar autograd is elegant but slow, so the next lesson rebuilds the same ideas with fast NumPy matrices and uses them to read handwritten digits: [23 · Neural Network in Pure NumPy: MNIST Digits](23-mlp-numpy-mnist.md).
