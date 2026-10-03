# 24 · PyTorch Fundamentals

**Phase 3 — Deep Learning** · Estimated time: 1 week · Prerequisites: [22 · Build micrograd](22-micrograd-backpropagation.md), [23 · Neural Network in Pure NumPy](23-mlp-numpy-mnist.md)

> Lesson 22 built backpropagation by hand, and lesson 23 trained a real digit classifier in pure NumPy: hundreds of lines of gradients, loops and shape bugs. PyTorch is the industry-standard library that automates all of that. Every line of PyTorch code in this lesson maps to something already built in those two lessons, so nothing in the framework is a black box. The lesson introduces tensors, autograd, layers, data loaders and optimizers, and uses them to re-fit a line and to rebuild the MNIST network in a fraction of the code. It ends with a clean, reusable training script that the next nine lessons build on.

## What this lesson builds

- **Project 1 — Autograd feels familiar:** a script that recreates the lesson-22 expression graph in PyTorch, proves that the gradients match micrograd, then fits `y = wx + b` using nothing but tensors and `.backward()`.
- **Project 2 — MNIST, take two:** the lesson-23 digit classifier rebuilt in PyTorch with `nn.Module`, `DataLoader` and Adam, matching or beating the NumPy accuracy, then saved to disk and reloaded to predict.
- **Project 3 — A training template:** a clean `train.py` with reusable train/eval functions, loss and accuracy curves, and best-model checkpointing: the skeleton that lessons 25–33 reuse.

## Concepts covered

- A **tensor** is a NumPy array that also knows how to track gradients and run on a GPU.
- `requires_grad=True` and `.backward()`: PyTorch's autograd, the same engine written by hand in micrograd.
- `nn.Module`, `nn.Linear` and `forward()`: how PyTorch packages layers and their parameters.
- `Dataset` and `DataLoader`: objects that serve the data in shuffled mini-batches automatically.
- **Optimizers** (`SGD`, `Adam`) and why `zero_grad()` must be called every step.
- The canonical five-line PyTorch training loop that appears in essentially every deep learning codebase.
- Saving and loading models with `state_dict`, a dictionary of all learned weights.
- Running on a GPU with `.to(device)` (`cuda` for NVIDIA cards, `mps` for Apple Silicon Macs), and free GPU options for a laptop that has none.

## The secret decoder table

Each PyTorch name below is a renamed version of something already built. Keep this table open all week:

| Built in lessons 22–23 | PyTorch calls it |
|---|---|
| `Value` class in micrograd | `torch.Tensor` with `requires_grad=True` |
| `value.grad` accumulated by hand | `tensor.grad` |
| `loss.backward()` written by hand (topo sort + chain rule) | `loss.backward()` (the same name: Karpathy copied the API on purpose) |
| Resetting grads to zero each step | `optimizer.zero_grad()` |
| The NumPy weight matrices `W1, b1, W2, b2` | `nn.Linear(in_features, out_features)` |
| The `forward(x)` function of matmuls + ReLU | a `nn.Module`'s `forward()` method |
| The softmax + cross-entropy loss code | `nn.CrossEntropyLoss()` |
| `W -= lr * dW` update loop | `optimizer.step()` |
| Hand-rolled mini-batch slicing/shuffling | `DataLoader(dataset, batch_size=..., shuffle=True)` |

## Before starting

Finish lessons 22 and 23 first. This lesson only makes sense once the pain it removes is familiar. Keep the micrograd code and the NumPy MNIST script nearby; the projects compare against both.

Activate the venv at the repo root and install PyTorch. On **macOS**, the command below installs the regular Mac build, which can also use an Apple Silicon Mac's built-in GPU. Current PyTorch does not install on Intel Macs at all, so on an Intel Mac do this lesson in a free [Google Colab](https://colab.research.google.com) notebook instead, as lesson 01 suggested (PyTorch comes preinstalled there). On **Windows (WSL2)** and **Linux** without an NVIDIA GPU, the same command installs the smaller CPU-only build:

```bash
cd ~/ml/ml-learn
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

If the computer does have an NVIDIA GPU set up in WSL2 or on Linux, get the right command from the "Get Started" selector on [pytorch.org](https://pytorch.org) instead. Either way, verify the install:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.backends.mps.is_available())"
```

It prints the version, then `True` or `False` for an NVIDIA GPU (`cuda`) and for an Apple Silicon Mac's GPU (`mps`, macOS 14 or newer). A CPU is completely fine for this whole lesson, since MNIST is small. When later lessons need more computing power, Google Colab and Kaggle notebooks (kaggle.com, account required) provide free GPU time in the browser.

Create the work folder for this lesson:

```bash
mkdir -p work/24-pytorch-fundamentals && cd work/24-pytorch-fundamentals
```

## Project 1 — Autograd feels familiar

**Goal:** Prove that `torch.Tensor` is an industrial-strength version of the micrograd `Value` class. Recreate the lesson-22 expression graph, confirm that the gradients match, then fit a line using raw autograd (no `nn.Module`, no optimizer).

**Milestones**

- [ ] In `autograd_hello.py`, create scalar tensors with gradient tracking on: `a = torch.tensor(2.0, requires_grad=True)`. `requires_grad=True` tells PyTorch "record every operation on this tensor so gradients can flow back to it". That is exactly what the micrograd `Value` class did automatically.
- [ ] Rebuild the exact tiny expression graph from lesson 22. If that was Karpathy's graph, it is `e = a*b`, `d = e + c`, `L = d*f` with `a=2, b=-3, c=10, f=-2`. Print `L`. Checkpoint: `L` is `-8.0`, the same number micrograd gave.
- [ ] Call `L.backward()` and print `a.grad`, `b.grad`, `c.grad`, `f.grad`. Checkpoint: the gradients match the micrograd run exactly (for the example above: `a.grad=6.0`, `b.grad=-4.0`, `c.grad=-2.0`, `f.grad=4.0`). What happened inside that call is exactly what the hand-written micrograd code does.
- [ ] Run `L.backward()` a second time in a fresh script but *without* resetting, using `retain_graph=True`, and print `a.grad` again. Checkpoint: the gradient has **doubled**. PyTorch accumulates gradients into `.grad` instead of overwriting them, which is why every training loop must zero them out each step.
- [ ] In a new script, `fit_line.py`, generate fake data: 100 points of `y = 3x + 2` plus a little noise, using `torch.randn` (lesson 12 did the same with NumPy).
- [ ] Create `w` and `b` as tensors with `requires_grad=True`, both starting at 0. Write a loop of about 200 steps: compute predictions `y_pred = w*x + b`, compute the mean squared error loss, call `loss.backward()`, then update the parameters by hand inside a `with torch.no_grad():` block (`w -= lr * w.grad`), and finally zero both grads with `w.grad.zero_()` and `b.grad.zero_()`. The `no_grad` block means "do not record these operations": updating weights is bookkeeping, not part of the math being differentiated.
- [ ] Print the loss every 20 steps. Checkpoint: the loss falls steadily, and `w` ends near `3.0` and `b` near `2.0` (within about ±0.2). This is lesson 12 again, but this time autograd computed every derivative.

<details><summary>Hints</summary>

- `RuntimeError: element 0 of tensors does not require grad` means the loss was built from tensors that are not connected to `w` and `b`. Check that no step accidentally converted to NumPy or used `.detach()`.
- Trying to backward twice through the same graph raises an error by design. Each forward pass builds a fresh graph, so the loop recomputes `y_pred` and `loss` every iteration.
- If the loss explodes to `inf`, the learning rate is too high. Try `0.01` or `0.001`.
- `.item()` turns a one-element tensor into a plain Python number for printing.

</details>

**Definition of done:** gradients on the expression graph match micrograd exactly, and the line fit recovers `w≈3`, `b≈2` with autograd doing all differentiation.

## Project 2 — MNIST, take two

**Goal:** Rebuild the lesson-23 MNIST network in PyTorch and watch a week of NumPy code shrink to about 50 lines that train faster and score at least as well.

**Milestones**

- [ ] In `mnist_torch.py`, load MNIST via torchvision, which downloads it automatically (no account needed): `datasets.MNIST("data", train=True, download=True, transform=transforms.ToTensor())`. `ToTensor` converts each image to a float tensor scaled to 0–1, the same normalization that lesson 23 did by hand.
- [ ] Wrap the train and test sets in `DataLoader`s (`batch_size=64`, `shuffle=True` for train only). A `DataLoader` is an iterator that hands out shuffled mini-batches: the slicing-and-shuffling code from lesson 23, automated. Checkpoint: looping `for images, labels in train_loader:` once and printing shapes gives `torch.Size([64, 1, 28, 28])` and `torch.Size([64])`.
- [ ] Define the same architecture as lesson 23 using `nn.Sequential`: `nn.Flatten()`, then `nn.Linear(784, 128)`, `nn.ReLU()`, `nn.Linear(128, 10)`. `nn.Linear` creates the weight matrix and bias automatically, randomly initialized and with `requires_grad` already on. Print the model, and count parameters with `sum(p.numel() for p in model.parameters())`. Checkpoint: 101,770 parameters (or the matching count, if the lesson-23 network used different layer sizes).
- [ ] Set up `loss_fn = nn.CrossEntropyLoss()` (softmax + cross-entropy fused, the lesson-23 loss in one object; it takes raw scores, so the model has no softmax layer) and `optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)`. Adam is a smarter cousin of SGD that adapts the step size per weight; lesson 25 explains why.
- [ ] **Overfit one batch first.** Take a single batch and train on only that batch for ~100 steps with the canonical loop:

  ```python
  optimizer.zero_grad()          # reset grads (they accumulate, as Project 1 showed)
  logits = model(images)         # forward pass — calls forward() automatically
  loss = loss_fn(logits, labels) # how wrong are we?
  loss.backward()                # backprop — the lesson 22 code, industrialized
  optimizer.step()               # update every parameter
  ```

  Checkpoint: the loss on that batch drops below 0.01. This sanity check (*can the model memorize 64 examples?*) catches most wiring bugs and is a habit that professionals never skip.
- [ ] Now train on the full training set for 5 epochs (an epoch is one full pass through the data), printing the average loss per epoch. Checkpoint: the first-batch loss starts near 2.3 (that is −ln(1/10), a random 10-way guess, as lesson 23 explained) and the epoch loss falls below 0.1.
- [ ] Write an evaluation pass over the test loader inside `with torch.no_grad():`, using `model.eval()` first. Take the arg-max of the 10 scores as the prediction and compute accuracy. Checkpoint: test accuracy ≥ 0.97, matching or beating the NumPy network in far less code and time.
- [ ] Save the trained weights: `torch.save(model.state_dict(), "mnist.pt")`. A `state_dict` is a plain dictionary that maps parameter names to tensors; print its keys to see them.
- [ ] In a separate script `predict.py`, rebuild the same architecture, load the weights with `model.load_state_dict(torch.load("mnist.pt"))`, and predict 8 test images, showing each image with its predicted and true label in a matplotlib figure saved as `predictions.png` (lesson 07). Checkpoint: the reloaded model gets most or all of the 8 predictions right.
- [ ] Add device support at the top: `device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"`, then `model.to(device)` and move each batch with `images, labels = images.to(device), labels.to(device)`. `cuda` means an NVIDIA GPU and `mps` an Apple Silicon Mac's built-in GPU. On a CPU it changes nothing; the point is that the script is now GPU-ready for free. Pasted into a Colab or Kaggle GPU notebook, it speeds up with zero edits.

<details><summary>Hints</summary>

- `CrossEntropyLoss` wants raw scores ("logits") and integer class labels. If a softmax layer was added or the labels were one-hot encoded, remove that.
- Read shape errors bottom-up: the message names the two mismatched shapes. Print `images.shape` right before the failing line and walk through the layer sizes.
- Missing `zero_grad()`? The loss falls, then goes haywire: gradients from every step are piling up.
- `model.train()` / `model.eval()` toggle training-only behaviors. They are harmless for this model, but build the habit now: lesson 25 introduces layers where it matters.

</details>

**Definition of done:** ≥ 0.97 test accuracy, model saved and reloaded in a separate script that predicts correctly, and the five loop lines can be recited from memory.

## Project 3 — A training template

**Goal:** Refactor Project 2 into a clean, reusable `train.py`. Lessons 25 through 33 all start from this file, so an hour of tidying now pays off for months.

**Milestones**

- [ ] Create `train.py` with this skeleton and fill in the function bodies:

  ```python
  def train_one_epoch(model, loader, loss_fn, optimizer, device):
      """One pass over the data. Returns (avg_loss, accuracy)."""

  def evaluate(model, loader, loss_fn, device):
      """No-grad pass over the data. Returns (avg_loss, accuracy)."""

  def main():
      # config (lr, batch_size, epochs) as variables at the top
      # data -> model -> loss -> optimizer -> epoch loop -> plot
  ```

- [ ] Both functions return average loss *and* accuracy, so one function serves any classification task. Guard the entry point with `if __name__ == "__main__": main()` (from lesson 04).
- [ ] In the epoch loop, call both functions, print a one-line summary per epoch (`epoch 3: train loss 0.081 acc 0.976 | test loss 0.092 acc 0.972`), and append all four numbers to history lists.
- [ ] Checkpoint the best model: whenever test accuracy beats the best so far, `torch.save` the `state_dict` to `best_model.pt` and print a message saying so. This way, the best weights are never lost to a bad epoch late in training.
- [ ] After training, plot two side-by-side charts with matplotlib (lesson 07): loss curves and accuracy curves, each with train and test lines, labeled, saved as `curves.png`. Checkpoint: both training curves improve smoothly, and the test curves track them closely. A widening gap between train and test is overfitting, the problem that lesson 25 tackles.
- [ ] Prove reusability: change *only* the model definition (say, hidden size 128 → 256) and rerun. Checkpoint: everything works untouched and `best_model.pt` updates only if the new run actually wins.
- [ ] Commit the template to git with a clear message, because lesson 25 starts from a copy of this file.

<details><summary>Hints</summary>

- Accumulate `loss.item() * batch_size` and correct-prediction counts inside the loop; divide by total samples at the end for true averages.
- `evaluate` is `train_one_epoch` minus `backward`/`step`/`zero_grad`, wrapped in `torch.no_grad()`, with `model.eval()` first (and `model.train()` back in the train function).
- Keep every tunable number (learning rate, batch size, epochs, hidden size) in one config block at the top, never buried in function bodies.

</details>

**Definition of done:** `python3 train.py` trains MNIST, prints per-epoch metrics, saves `best_model.pt` and `curves.png`, and swapping the model requires touching only one place.

## Stretch goals

- Write a custom `Dataset` class (implement `__len__` and `__getitem__`) that serves MNIST from the raw files used in lesson 23, and confirm that `DataLoader` consumes it without complaint.
- Race SGD vs Adam: same model, 5 epochs each, both loss curves on one plot. Which wins on MNIST, and by how much?
- Upload the script to a free GPU notebook (Colab, or kaggle.com with an account) and time one epoch on the GPU vs the computer's own CPU (on an Apple Silicon Mac, set `device = "cpu"` by hand for that run, and time `"mps"` too). The lessons from 26 onward lean on that speedup.
- Rewrite the model as an explicit `nn.Module` subclass with `__init__` and `forward()` instead of `nn.Sequential`. That form is needed once architectures stop being straight lines.

## Getting unstuck

- **Read errors bottom-up.** PyTorch stack traces are long, but the last few lines name the actual problem: usually two mismatched shapes, both spelled out.
- **Print shapes relentlessly.** `print(x.shape)` before the failing line solves most tensor bugs. Remember that loaders yield `[batch, 1, 28, 28]`; that is what `nn.Flatten()` is for.
- **Loss stuck at 2.30?** The model is guessing randomly: the learning rate is too low, gradients are not flowing, or `optimizer.step()` is missing. Rerun the overfit-one-batch test: it isolates wiring bugs from data bugs.
- **`grad` is `None`?** Gradients only land on *leaf* tensors created with `requires_grad=True`; results of operations expose grads differently.
- When lost, ask an AI assistant for a **hint, not a solution**: "why might CrossEntropyLoss complain about my target shape?" beats "write my training loop".
- Type every line by hand. This matters most for the five-line loop, which must be in muscle memory by the end of this week.

## Resources

- [PyTorch: Learn the Basics](https://pytorch.org/tutorials/beginner/basics/intro.html) — the official beginner path (tensors → datasets → autograd → training); it mirrors this lesson and works well as a second angle.
- [Deep Learning with PyTorch: A 60 Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) — the classic fast tour of autograd and nn; skim it after Project 1, when everything in it should look familiar.
- [pytorch.org](https://pytorch.org) — installation selector and API docs; look up any `nn.` layer or `torch.` function here along the way.

## Skills unlocked

- [ ] I can create tensors with `requires_grad` and explain what `.backward()` does, because I once wrote it by hand.
- [ ] I can explain why gradients must be zeroed every training step, and what happens if they are not.
- [ ] I can build a network with `nn.Sequential` or `nn.Module` and feed it batches from a `DataLoader`.
- [ ] I can write the canonical five-line training loop from memory.
- [ ] I use overfit-one-batch as the first sanity check on any new model.
- [ ] I can save a `state_dict`, reload it in a fresh script, and predict with it.
- [ ] I can move a model and data to a GPU with `.to(device)`, and I know where to find a free one.
- [ ] I have a clean `train.py` template with train/eval functions, curves, and best-model checkpointing, committed to git.

## Next up

With a network that trains, the next lesson looks at why training sometimes *fails* and how to diagnose and tame deep nets: [25 · The Dark Arts of Training Deep Networks](25-training-deep-nets.md).
