# 23 · Neural Network in Pure NumPy: MNIST Digits

**Phase 3 — Deep Learning** · Estimated time: 1-2 weeks · Prerequisites: [05 · NumPy](../phase-0-foundations/05-numpy.md), [09 · Calculus and Gradient Descent](../phase-1-math/09-calculus-and-gradient-descent.md), [14 · Model Evaluation](../phase-2-classical-ml/14-model-evaluation.md), [22 · Build micrograd](22-micrograd-backpropagation.md)

> In lesson 22 you built backpropagation one scalar at a time — beautiful for understanding, hopeless for speed. Real networks process thousands of numbers at once using matrix multiplication, and that vectorized version is exactly what PyTorch and every other framework does under the hood. In this lesson you build it yourself: a full neural network in raw NumPy that reads handwritten digits with 95%+ accuracy. When you finish, frameworks will never feel like magic again — you will know precisely what every line of PyTorch is doing, because you wrote it first.

## What you will build

- **Project 1 — Load and look:** a script that downloads MNIST (70,000 handwritten digits), plots a labeled grid of them, and prepares clean normalized train/validation/test arrays.
- **Project 2 — The network:** a 784-128-10 neural network in pure NumPy — vectorized forward pass, softmax + cross-entropy, matrix-form backprop, minibatch SGD — trained to 95%+ test accuracy.
- **Project 3 — Look inside:** visualizations of what your network learned — first-layer weights as images, a confusion matrix built with your lesson-14 code, and a gallery of its 25 most confidently-wrong predictions.

## Concepts you will learn by doing

- **Vectorized forward pass** — computing the network's output for a whole batch of images with one matrix multiply, instead of looping over examples.
- **Softmax** — a function that turns 10 raw scores into 10 probabilities that sum to 1.
- **Cross-entropy loss** — the multi-class loss: punish the network by how little probability it gave the correct class.
- **One-hot labels** — encoding the label "3" as a vector `[0,0,0,1,0,0,0,0,0,0]` so it can be compared to the 10 output probabilities.
- **Backprop in matrix form** — the same chain rule as micrograd, but on whole matrices; you derive it by making shapes match, not by pushing symbols.
- **Minibatch stochastic gradient descent (SGD)** — updating weights from small random batches of data instead of the whole dataset at once.
- **Weight initialization** — why starting all weights at zero kills a network, and why small random numbers work.
- **Train/validation monitoring and early stopping** — watching two curves and stopping before the network memorizes the training set.

## Before you start

Check you can do these; if not, revisit the linked lessons:

- Multiply matrices in NumPy and predict the result's shape (lesson 05: `(64,784) @ (784,128)` → `(64,128)`).
- Explain what a gradient is and write a basic gradient descent loop (lesson 09).
- Explain what backprop does, from building micrograd (lesson 22).
- You have your own accuracy + confusion-matrix functions from lesson 14.

Install what you need (from the repo root, inside your venv):

```bash
cd ~/ml/ml-learn
source .venv/bin/activate
pip install numpy matplotlib scikit-learn
```

scikit-learn is used **only** to download MNIST — every model line will be pure NumPy.

Create your work folder:

```bash
mkdir -p work/23-mlp-numpy-mnist
cd work/23-mlp-numpy-mnist
```

## Project 1 — Load and look

**Goal:** Get MNIST onto your disk, understand that each image is just a 784-number vector (lesson 05 thinking), and produce clean, normalized, split arrays that Project 2 can load instantly.

**Milestones**

- [ ] Write `download_mnist.py` that fetches the dataset once and caches it locally, so you never re-download. Starter:

  ```python
  from sklearn.datasets import fetch_openml
  import numpy as np

  mnist = fetch_openml('mnist_784', version=1, as_frame=False)
  X = mnist.data.astype(np.float32)      # shape (70000, 784)
  y = mnist.target.astype(np.int64)      # shape (70000,)
  np.savez_compressed('mnist.npz', X=X, y=y)
  ```

  The first run downloads ~50 MB and may take a few minutes. Checkpoint: `mnist.npz` exists and `X.shape` prints `(70000, 784)`.
- [ ] In a new script `explore.py`, load the `.npz`, take one row of `X`, and print its min, max, and shape. Each image is a flat vector of 784 pixel brightnesses (0–255). Reshape one to `(28, 28)` and print a few rows — you can almost see the digit in the numbers.
- [ ] Plot a 5×5 grid of digits with `plt.imshow(img, cmap='gray')`, each subplot titled with its label. Checkpoint: you see 25 handwritten digits and every title matches the digit you see.
- [ ] Normalize pixels to the range 0–1 by dividing by 255.0. This matters: gradient descent behaves badly when inputs are large — you saw the same thing with feature scaling in lesson 19.
- [ ] Split the data: the standard MNIST split is first 60,000 for training, last 10,000 for **test** (touched only once, at the very end). Carve the last 10,000 of the training portion off as your **validation set** (used every epoch to monitor progress). You end with 50,000 train / 10,000 val / 10,000 test.
- [ ] Write a `one_hot(y, num_classes=10)` function that turns labels into one-hot vectors. Checkpoint: `one_hot(y_train).shape` is `(50000, 10)` and every row sums to exactly 1.
- [ ] Save all six arrays to `mnist_prepared.npz` so Project 2 starts with one `np.load`.

<details><summary>Hints</summary>

- If `fetch_openml` is slow or flaky, just let it retry — it caches under `~/scikit_learn_data/` so the pain is once. (Alternative: `torchvision.datasets.MNIST` also works if you already have PyTorch, but you don't need it yet.)
- For one-hot, avoid a Python loop: create a zeros array of shape `(n, 10)` and use fancy indexing — `arr[np.arange(n), y] = 1` — straight from lesson 05.
- `imshow` looking inverted or weird? Check you reshaped to `(28, 28)` and not `(784,)`, and that you didn't normalize twice.

</details>

**Definition of done:** `mnist_prepared.npz` exists with correctly-shaped, 0–1-normalized train/val/test arrays, and you can display any digit with its label in one line.

## Project 2 — The network

**Goal:** Implement a 784-128-10 network — every forward and backward line yours — and train it with minibatch SGD to 95%+ accuracy on the test set. This is the flagship project of the phase.

**Milestones**

- [ ] Create `network.py`. Initialize parameters: `W1` shape `(784, 128)`, `b1` shape `(128,)`, `W2` shape `(128, 10)`, `b2` shape `(10,)`. Use small random values: `np.random.randn(784, 128) * 0.01` (biases can be zeros).
- [ ] Before moving on, convince yourself why **all-zero weights fail**: every hidden neuron would compute the identical output and receive the identical gradient, so all 128 neurons stay clones forever — the network can never use its width. This is called the symmetry problem. Random init breaks the tie.
- [ ] Write the vectorized forward pass for a whole batch `X` of shape `(B, 784)`:
  - `Z1 = X @ W1 + b1` → shape `(B, 128)` (broadcasting adds `b1` to every row — lesson 05)
  - `A1 = relu(Z1)` — ReLU is just `np.maximum(0, Z1)`
  - `Z2 = A1 @ W2 + b2` → shape `(B, 10)` — these raw scores are called **logits**
  - `probs = softmax(Z2)` — exponentiate each logit, divide by the row's sum.
  Checkpoint: on a batch of 64 images, `probs.shape` is `(64, 10)` and every row sums to 1.0 (allow tiny float error).
- [ ] Write `cross_entropy(probs, Y_onehot)`: for each example, take `-log` of the probability assigned to the true class, then average over the batch. Checkpoint: on the **untrained** network the loss is about **2.30** — that is `-log(1/10)`, the loss of pure guessing among 10 classes. If you see 2.30, your forward pass and loss are almost certainly correct.
- [ ] Derive the backward pass **by shapes**. You need `dW1, db1, dW2, db2` — each the same shape as its parameter. The two facts that unlock everything:
  1. Softmax + cross-entropy together have a famously clean gradient: `dZ2 = (probs - Y_onehot) / B`, shape `(B, 10)`.
  2. For any layer `Z = A @ W + b`: `dW = A.T @ dZ`, `db = dZ.sum(axis=0)`, `dA = dZ @ W.T`. Check it: `dW2` needs shape `(128, 10)`; `A1.T @ dZ2` is `(128, B) @ (B, 10)` = `(128, 10)`. The shapes only fit together one way — that is the whole trick.
- [ ] Finish the chain: propagate `dA1` back through ReLU (gradient passes where `Z1 > 0`, is zero elsewhere), then apply the same layer rule to get `dW1, db1`. Checkpoint: print all four gradient shapes; each matches its parameter exactly.
- [ ] **The sanity check you will use for the rest of your life:** take a fixed batch of just 32 images and train on only those, full-batch, a few hundred steps. A correct implementation must drive loss below 0.01 and hit 100% accuracy on those 32 — it is memorizing them, which is exactly the point. If it can't overfit 32 examples, there is a bug, and you just saved yourself hours of debugging a full training run. (Karpathy's training recipe preaches this check for a reason.)
- [ ] Now the real training loop in `train.py`:
  - Each **epoch** (one full pass over the data): shuffle the 50,000 training examples (`np.random.permutation` — shuffle X and Y with the *same* permutation!), slice into minibatches of 64, and for each: forward → loss → backward → update every parameter with `param -= lr * grad`. Start with `lr = 0.1`.
  - After each epoch, compute and print train loss, train accuracy, and **validation** accuracy.
- [ ] Checkpoint after epoch 1: validation accuracy already above 0.90. MNIST is friendly like that.
- [ ] Add simple **early stopping**: keep the parameters from the epoch with the best validation accuracy (save with `np.savez('model.npz', ...)`), and stop when validation accuracy hasn't improved for ~5 epochs.
- [ ] Final evaluation — load the best saved model, run the test set **once**: Checkpoint: **test accuracy ≥ 0.95.** With this architecture and a little learning-rate tuning, 0.97+ is reachable.

<details><summary>Hints</summary>

- **Softmax overflows:** `np.exp(1000)` is `inf`, and MNIST logits can get big. Subtract each row's max logit before exponentiating — this changes nothing mathematically (it cancels in the division) but keeps everything finite. If you see `nan` loss, this is the first suspect.
- **Loss stuck at 2.30?** Gradients are probably not flowing: check you're actually updating the parameters (not copies of them), and that your learning rate isn't `0.0001`-tiny.
- **Loss explodes / oscillates?** Learning rate too high, or you forgot the `/ B` in `dZ2`, making gradients 64× too big.
- **Not sure your backward pass is right?** Do a numerical gradient check (lesson 09 trick): nudge one weight by `1e-5`, recompute the loss, and compare `(loss2 - loss1) / 1e-5` to your analytic gradient for that weight. They should agree to several decimal places.
- Structure suggestion: a function `forward(params, X)` returning the cache `(Z1, A1, probs)`, and `backward(params, cache, X, Y)` returning grads — you'll recognize this exact shape in PyTorch next lesson.

</details>

**Definition of done:** `train.py` trains from scratch, prints per-epoch train/val metrics, early-stops, and the saved model scores ≥ 95% on the untouched test set — with zero ML libraries in the model code.

## Project 3 — Look inside

**Goal:** A trained network is not a black box you built — open it up. See what the neurons learned, where the model fails, and why the failures make sense.

**Milestones**

- [ ] Load `model.npz`. Each **column** of `W1` (784 numbers) is one hidden neuron's weights — and 784 numbers can be reshaped to `28×28` and shown as an image. Plot the first 64 neurons in an 8×8 grid (`cmap='gray'` or `'RdBu'`). Checkpoint: the images are not pure noise — many look like blobs, edges, and stroke fragments. Your network invented its own pen-stroke detectors, and nobody told it to.
- [ ] Compute predictions on the full test set and build a **confusion matrix** — using YOUR functions from [lesson 14](../phase-2-classical-ml/14-model-evaluation.md), not sklearn's. Plot it with `plt.imshow` plus the counts written in each cell. Checkpoint: the diagonal is heavy; the biggest off-diagonal cells are believable pairs like 4↔9, 3↔5, 7↔2.
- [ ] Find the **25 most confidently-wrong** test predictions: among all misclassified images, the ones where the network gave its (wrong) answer the highest probability. Plot them in a 5×5 grid titled `true→predicted (confidence)`.
- [ ] Study that gallery and write 5 sentences in a `notes.md`: Which mistakes would a human also make? Are any labels arguably wrong in the dataset itself? (Some are!) What does that imply about ever reaching 100%?

<details><summary>Hints</summary>

- Confidence of a prediction = `probs.max(axis=1)`. Wrong-mask = `preds != y_test`. Combine them, then `np.argsort` the confidences of the wrong ones and take the last 25.
- Weight images look washed out? Give each subplot its own color scale, or normalize each column to its own min/max before plotting.

</details>

**Definition of done:** Three saved figures — weight grid, confusion matrix, wrong-gallery — plus `notes.md` with your reading of the failures.

## Stretch goals

- **Go deeper:** add a second hidden layer (784-128-64-10). The layer rule from Project 2 repeats mechanically — that regularity is exactly why frameworks can automate it. Does accuracy improve?
- **Momentum:** upgrade SGD with velocity: `v = 0.9*v - lr*grad; param += v`. Compare epochs-to-95% against plain SGD.
- **Init showdown:** train three times — zeros, `randn*0.01`, and He initialization (`randn * sqrt(2/fan_in)`, designed for ReLU) — and plot the three loss curves on one chart. Zeros should flatline; He should win.
- **Fashion-MNIST:** the drop-in harder sibling (`fetch_openml('Fashion-MNIST')` — clothing items, same shapes). Your code should run unchanged. How much does accuracy drop?

## If you get stuck

- **Print shapes first, values second.** Ninety percent of NumPy neural-net bugs are shape bugs, and most of the rest are silent broadcasting doing the wrong thing. Annotate every line of forward/backward with its expected shape in a comment, then assert a few of them.
- **`nan` loss** → softmax overflow (subtract the row max) or `log(0)` (clip probs with `np.clip(probs, 1e-12, 1.0)` before logging).
- **Trains but plateaus around 90%** → check normalization actually happened (print `X_train.max()` — it should be 1.0, not 255), and try `lr` of 0.05 / 0.1 / 0.5 for one epoch each.
- **Suspect gradients** → numerical gradient check on a tiny network (say 10-5-3 with fake data). Small nets make bugs obvious.
- Standing advice: read the error traceback bottom-up; print intermediate values; when truly stuck, ask an AI assistant for a **hint** ("what could make cross-entropy go nan?") and never for the solution; and type every line yourself — copy-paste teaches nothing.

## Resources

- [3Blue1Brown — Neural networks series, chapters 2–4](https://www.3blue1brown.com) — gradient descent and backprop as animations; watch chapter 4 right before deriving your backward pass (also on YouTube: search "3Blue1Brown backpropagation").
- [Michael Nielsen — *Neural Networks and Deep Learning*](http://neuralnetworksanddeeplearning.com/) — a free book that builds a NumPy MNIST network much like yours; chapters 1–2 are the perfect companion (peek only after you've attempted each piece).
- [NumPy broadcasting docs](https://numpy.org/doc/stable/user/basics.broadcasting.html) — the reference for why `+ b1` works across a whole batch.
- Andrej Karpathy — "A Recipe for Training Neural Networks" (blog post; search for it) — the overfit-one-batch check and other lifelong habits, from the source.

## Skills unlocked

- [ ] I can explain why an image is "just" a 784-dimensional vector, and normalize and split a real dataset.
- [ ] I can write a vectorized forward pass and predict the shape of every intermediate matrix before running it.
- [ ] I can explain softmax, cross-entropy, and one-hot labels in plain words, and why untrained loss is ~2.30 for 10 classes.
- [ ] I can derive matrix-form backprop by matching shapes, and verify it with a numerical gradient check.
- [ ] I can write a minibatch SGD loop with shuffling, and explain why all-zero initialization fails.
- [ ] I overfit a tiny batch as a sanity check before every full training run — habit installed.
- [ ] I can read training vs. validation curves and apply early stopping.
- [ ] My pure-NumPy network scores 95%+ on MNIST, and I can show what its neurons learned.

## Next up

You have now built by hand everything a deep learning framework automates — time to claim the reward: [24 · PyTorch Fundamentals](24-pytorch-fundamentals.md), where tensors, autograd, and `nn.Module` will feel like old friends with better engines.
