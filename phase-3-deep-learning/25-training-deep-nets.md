# 25 · The Dark Arts of Training Deep Networks (Karpathy 3-4)

**Phase 3 — Deep Learning** · Estimated time: 1-2 weeks · Prerequisites: [22 · Build micrograd](22-micrograd-backpropagation.md), [23 · Neural Network in Pure NumPy](23-mlp-numpy-mnist.md), [24 · PyTorch Fundamentals](24-pytorch-fundamentals.md)

> You can now build and train a neural network. But sooner or later you will meet a network that simply *refuses* to learn — the loss sits flat, or explodes to NaN, and nothing in the code looks wrong. This lesson teaches you the difference between "ran a tutorial" and "can debug a net that won't train": you will look *inside* a training network at its activation and gradient statistics, deliberately break training in controlled ways, and learn the standard toolkit that fixes it — initialization, batch normalization, better optimizers, schedules, and regularization. Read Karpathy's "A Recipe for Training Neural Networks" blog post alongside this lesson; it is the professional checklist version of everything you are about to experience firsthand.

## What you will build

- **Project 1 — Watch the statistics:** a build-along of Karpathy's makemore Part 3, then a diagnostic notebook for *your* lesson-24 MNIST network showing per-layer histograms of activations and gradients under bad vs good initialization.
- **Project 2 — The great training bake-off:** a controlled 8-run experiment on FashionMNIST — {bad init, good init} × {SGD, Adam} × {with/without BatchNorm} — producing one chart with all loss curves, a results table, and 5 written conclusions.
- **Project 3 — Backprop ninja (stretch but recommended):** makemore Part 4 — manual backpropagation through an entire network, layer by layer, verified tensor-by-tensor against PyTorch's autograd.

## Concepts you will learn by doing

- **Vanishing/exploding activations and gradients** — signals shrinking to zero or blowing up as they pass through layers, which stalls or destroys learning.
- **Why initialization scale matters** — the Kaiming idea: pick starting weight sizes so each layer's output has roughly the same spread as its input.
- **Dead ReLUs and saturated tanh** — neurons stuck outputting zero (ReLU) or pinned at ±1 (tanh), diagnosed with histograms.
- **Batch normalization / layer norm** — layers that re-center and re-scale activations during training so the network stays in a healthy range.
- **Adam vs SGD+momentum** — optimizers that adapt the step size per weight vs plain gradient steps with velocity.
- **Learning-rate schedules and warmup** — changing the learning rate over training instead of keeping it fixed.
- **Dropout and weight decay** — two standard ways to fight overfitting.
- **Karpathy's debugging recipe** — first make the net overfit one tiny batch (prove it *can* learn), then regularize.

## Before you start

1. You need your working PyTorch setup and the MNIST MLP from [lesson 24](24-pytorch-fundamentals.md). Activate the venv at the repo root:

   ```bash
   cd ~/ml/ml-learn
   source .venv/bin/activate
   python3 -c "import torch, torchvision, matplotlib; print(torch.__version__)"
   ```

   If that import fails, install inside the venv: `pip install torch torchvision matplotlib`.

2. Download FashionMNIST (grayscale 28×28 images of clothing — a drop-in replacement for MNIST, no account needed):

   ```bash
   python3 -c "from torchvision import datasets; datasets.FashionMNIST('data', train=True, download=True); datasets.FashionMNIST('data', train=False, download=True)"
   ```

3. Create your work folder and download the names dataset that makemore trains on (32k first names, one per line — the same file you will meet properly in lesson 28):

   ```bash
   mkdir -p work/25-training-deep-nets
   wget https://raw.githubusercontent.com/karpathy/makemore/master/names.txt -P work/25-training-deep-nets/
   ```

4. A heads-up for Project 1: makemore Part 3 continues code built in makemore Parts 1-2, which this curriculum covers later ([lesson 28](../phase-4-nlp-transformers/28-language-models-makemore.md)). You do *not* need those videos first — the first ~15 minutes of Part 3 walk through the complete starter code: loading `names.txt`, building the character vocabulary (`stoi`/`itos`), the context windows, the embedding table `C`, and the train/dev/test split. Pause there and type *all* of it, not just the new material — every checkpoint in Project 1 depends on it.

5. Read [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/) once now (30 min). You won't absorb it all yet — read it again after Project 2 and it will feel like a summary of your own experience.

## Project 1 — Watch the statistics

**Goal:** Learn to *see* what is happening inside a training network by following Karpathy's makemore Part 3 video (activations, gradients, BatchNorm), then turn the same diagnostic plots on your own lesson-24 MNIST network.

**Milestones**

- [ ] Open [the Karpathy playlist](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) and start Part 3, "Building makemore Part 3: Activations & Gradients, BatchNorm". Note: the video continues a character-level language model (a net that predicts the next letter of a name) built in earlier videos you'll do properly in lesson 28. That's fine — type the starter code in full during the first ~15 minutes (see "Before you start", step 4), and from there on treat the model as "an MLP with an embedding table in front". Type every line yourself into `work/25-training-deep-nets/makemore3.py` (or a notebook); never copy-paste.
- [ ] First lesson from the video — *check your loss at initialization*. Before any training, a classifier that knows nothing should predict all classes equally, so its expected loss is `-ln(1/num_classes)`. Checkpoint: for the 27 characters in makemore you compute ≈ 3.29; the naive net in the video starts far higher (≈ 27), because it is *confidently wrong* at init. Fix it as Karpathy does (shrink the final layer's weights) and watch the starting loss drop to ≈ 3.3.
- [ ] Reproduce the tanh-saturation histogram: plot a histogram (a bar chart of how often each value occurs) of the hidden layer's tanh outputs. Checkpoint: with too-large initialization the histogram is two spikes at −1 and +1 (saturated — the neuron's gradient there is ~0, so it stops learning); after scaling the weights down, you see a healthy spread across (−1, 1).
- [ ] Follow the Kaiming initialization discussion: multiplying inputs by a weight matrix changes their spread (standard deviation), so we scale initial weights by `gain / sqrt(fan_in)` — `fan_in` is the number of inputs to the layer — to keep the spread constant layer after layer. Checkpoint: you can say in one sentence why deeper nets make this *more* important (small errors in scale compound at every layer — shrink×shrink×shrink → vanish, grow×grow×grow → explode).
- [ ] Add BatchNorm as in the video: a layer that normalizes each batch of activations to mean 0 and spread 1, then lets the network re-scale them with two learned parameters. Note its weirdness: behaves differently in training vs evaluation (`model.train()` vs `model.eval()` in PyTorch — remember this forever, it is a top-3 source of real bugs).
- [ ] Now your own experiment. Copy your lesson-24 MNIST MLP into `diagnose_mnist.py` and give it 4-5 hidden layers with `tanh` activations. Capture every layer's output during a forward pass. In PyTorch, forward hooks do this — a hook is a function PyTorch calls for you whenever a layer runs:

  ```python
  acts = {}
  def make_hook(name):
      def hook(module, inp, out):
          acts[name] = out.detach()
      return hook
  for name, layer in model.named_modules():
      if isinstance(layer, torch.nn.Tanh):
          layer.register_forward_hook(make_hook(name))
  ```

- [ ] Make a figure with one activation histogram per layer, in two versions. Bad init: initialize all `Linear` weights with `torch.nn.init.normal_(w, std=1.0)`. Good init: PyTorch's default (Kaiming-style). Checkpoint: bad init shows saturation piling up at ±1 that gets *worse* with depth; good init shows similar healthy spreads at every layer.
- [ ] Do the same for gradients: after `loss.backward()`, plot a histogram of `layer.weight.grad` per layer, and print each one's standard deviation. Checkpoint: with bad init the gradient spreads differ by orders of magnitude across layers (vanishing/exploding gradients — same disease as activations, seen on the backward pass); with good init they are within roughly one order of magnitude of each other.
- [ ] Swap `tanh` for `ReLU` and repeat with bad init (try `std=0.05` too). A ReLU outputs `max(0, x)`, so a neuron whose input is always negative outputs 0 forever and gets zero gradient — a **dead ReLU**. Checkpoint: you can compute the fraction of exactly-zero activations per layer, and with bad small init it is far above the healthy ~50%.

<details><summary>Hints</summary>

- The video is 1h45m. Budget 2-3 sessions, pausing to type and rerun. Speed 1.25× works; 2× does not (for learning).
- For the per-layer figure, `fig, axes = plt.subplots(1, n_layers, figsize=(16, 3))` then `axes[i].hist(a.flatten().numpy(), bins=50)` gives you the strip-of-histograms look from the video.
- If your hooks fire multiple times, you are calling `model(x)` more than once — clear `acts` before each forward pass.
- Dead-ReLU fraction: `(a == 0).float().mean()` on the post-ReLU activation tensor.

</details>

**Definition of done:** you have the makemore Part 3 code working with final validation loss ≈ 2.1 (matching the video), plus two saved figures for your MNIST net (activations, gradients; bad vs good init) and 3 written sentences interpreting them.

## Project 2 — The great training bake-off

**Goal:** Run a *controlled experiment* — change one thing at a time, measure, compare — across the three biggest training levers: initialization, optimizer, and BatchNorm.

**Milestones**

- [ ] In `work/25-training-deep-nets/bakeoff.py`, build a configurable MLP for FashionMNIST (784 → a few hidden layers → 10 classes) where a small config dict controls: `init` ("bad" = normal with std 1.0, "good" = PyTorch default), `optimizer` ("sgd" or "adam"), `batchnorm` (True/False). One `build_model(cfg)` and one `train(cfg)` function — no copy-pasted variants.
- [ ] Sanity checks first, straight from the Recipe blog post. (1) Initial loss: with good init, the first-batch loss should be ≈ `-ln(1/10)` ≈ 2.30. (2) **Overfit one batch**: train on a single batch of 64 images only, for a few hundred steps. Checkpoint: loss falls below 0.01. If a network cannot memorize 64 examples, something is broken — fix that *before* running any experiment.
- [ ] About optimizers: **SGD** takes a plain step downhill (`weight -= lr * grad`), usually with **momentum** — a running velocity that smooths the steps. **Adam** additionally adapts the step size for each individual weight based on the history of its gradients, which makes it far less sensitive to your learning-rate choice. They need different learning rates: start with SGD at `lr=0.1, momentum=0.9` and Adam at `lr=1e-3`.
- [ ] Run the full grid: 2 inits × 2 optimizers × 2 BatchNorm settings = 8 runs. Fix everything else (same seed via `torch.manual_seed(42)`, same batch size, same number of epochs — 3-5 epochs is plenty). Record the training loss every ~50 steps and the final test accuracy for each run. Save the recorded curves to disk (e.g. one `.npy` or CSV per run) so a plotting mistake doesn't force a re-train.
- [ ] Plot all 8 loss curves on ONE chart, labeled, with a legend. Log-scale on the y-axis (`plt.yscale('log')`) makes the differences readable. Save as `bakeoff.png`. Checkpoint: the curves visibly separate into a fast group and a slow/stuck group.
- [ ] Make a results table (in a `RESULTS.md` in your work folder): one row per run, columns for config and final test accuracy. Checkpoint: your best configuration reaches test accuracy above 0.86; bad-init + SGD + no BatchNorm is clearly the worst — likely stuck near 2.30 loss (i.e. still guessing) or far behind.
- [ ] Study the interactions, not just the winners: does BatchNorm *rescue* bad init? (It largely should — that robustness is a big reason it's everywhere.) Does Adam close the gap on its own? Which lever mattered most?
- [ ] Add one more run: your best config plus a learning-rate schedule. **Warmup** means starting with a tiny learning rate for the first few hundred steps and ramping up (protects the net while it's still in a random, fragile state); a **decay schedule** (e.g. cosine) then shrinks the rate over training so the net can settle into a minimum. Try `torch.optim.lr_scheduler.CosineAnnealingLR`. Checkpoint: the end of its loss curve is smoother (less noisy) than the unscheduled version.
- [ ] Write **5 conclusions** at the bottom of `RESULTS.md`. Each must cite your own numbers ("Adam with bad init reached X% vs SGD's Y%, so ..."), not folklore.

<details><summary>Hints</summary>

- Loop over configs with `itertools.product([...], [...], [...])` and store results in a dict keyed by the config tuple — resist writing eight blocks of code.
- To apply "bad" init after building: `for m in model.modules(): if isinstance(m, torch.nn.Linear): torch.nn.init.normal_(m.weight, std=1.0)`.
- BatchNorm for an MLP is `torch.nn.BatchNorm1d(hidden_size)`, placed between the Linear layer and the activation. And call `model.eval()` before computing test accuracy — with BatchNorm this is not optional.
- 8 CPU runs at 3 epochs each is roughly 10-25 minutes total. If it drags, subsample the training set to ~20k images — the *comparison* is the point, not the absolute score.

</details>

**Definition of done:** `bakeoff.png` (8 labeled curves), a results table, and 5 numbered conclusions backed by your own numbers.

## Project 3 — Backprop ninja (stretch but recommended)

**Goal:** The final exam of "I truly understand backprop": follow makemore Part 4 and compute every gradient in a real network *by hand* — through cross-entropy loss, a linear layer, tanh, and BatchNorm — verifying each tensor against PyTorch's autograd.

**Milestones**

- [ ] Watch "Building makemore Part 4: Becoming a Backprop Ninja" from [the playlist](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ). This one you *must* do as an exercise, not a movie: Karpathy poses each gradient as a puzzle — pause, derive it on paper, write the line, then watch the solution.
- [ ] Set up his `cmp` helper, which compares your manually computed gradient with autograd's and prints whether they match exactly and the maximum difference. This is your judge for the whole project.
- [ ] Exercise 1: backprop through the loss and the network one tensor at a time (logits, hidden states, weights, biases, embeddings...). Work backwards from the loss; at each step ask "how does the loss change if this value wiggles?" — chain rule, exactly as in your micrograd from [lesson 22](22-micrograd-backpropagation.md), but now with matrices. Checkpoint: `cmp` prints `exact: True` (or `maxdiff` below 1e-9 — tiny floating-point differences are fine) for every tensor.
- [ ] Exercise 2: derive the gradient of cross-entropy-from-logits as a single simplified expression. Checkpoint: your one-liner is (softmax minus one-hot) / batch_size, and `cmp` approves.
- [ ] Exercise 3: the same for BatchNorm — one condensed backward expression instead of ten intermediate tensors. This is the hardest math in the lesson; paper first, code second. Checkpoint: `cmp` approves.
- [ ] Final boss: train the network using *only your manual gradients* (autograd off for the update). Checkpoint: loss decreases just as it did with autograd — you have replaced PyTorch's backward pass with your own.

<details><summary>Hints</summary>

- The two rules that solve 90% of the puzzles: a tensor used in several places gets its gradients *summed*; a tensor that was *broadcast* (stretched to a bigger shape, like a bias added to every row) gets its gradient summed back over the stretched dimension.
- Stuck on any single gradient? Shape-match first: `dX` must have exactly the shape of `X`. Often the only expression with the right shapes is the right answer.
- It is normal for this project to take several sittings. One exercise per day is a perfectly good pace.

</details>

**Definition of done:** every `cmp` check passes, and the manually-backpropped training run learns. You may now, per Karpathy, claim the rank of backprop ninja.

## Stretch goals

- **LR finder:** train while increasing the learning rate exponentially each step and plot loss vs learning rate (log x-axis). Checkpoint: the plot shows a U-shape — flat, then falling, then exploding; a good LR sits on the downslope just before the minimum.
- **Regularization round:** add two bake-off rows — dropout (randomly zeroing a fraction of activations during training so the net can't over-rely on any one neuron, `torch.nn.Dropout(0.3)`) and weight decay (a penalty pulling weights toward zero; in Adam use `torch.optim.AdamW`). Compare *test* accuracy and the train-test gap.
- **LayerNorm swap:** replace BatchNorm1d with `torch.nn.LayerNorm` (normalizes across each sample's features instead of across the batch — no train/eval weirdness, standard in transformers, which you'll build in lesson 31). Compare curves.
- **Update-to-data ratio:** log Karpathy's favorite training diagnostic — `(lr * grad.std() / weight.std()).log10()` per layer — over training. Healthy is around −3; interpret what you see.

## If you get stuck

- **Loss is NaN or inf:** learning rate too high, or exploding activations. Print the loss every step to find the step where it blew up; then print activation standard deviations per layer at that step.
- **Loss flat at ≈ 2.30 (FashionMNIST) or ≈ 3.29 (makemore):** the net is outputting uniform guesses and learning nothing. Check that the optimizer actually received `model.parameters()`, that you call `optimizer.zero_grad()` each step, and check your init.
- **Test accuracy much worse than train accuracy with BatchNorm:** you forgot `model.eval()`. The other classic: forgetting `model.train()` when resuming training.
- **Your histograms look nothing like the video's:** print `tensor.std()` per layer — numbers are harder to misread than plots. Then check *where* you initialize: after building the optimizer is fine, after training has started is not.
- Standing advice: read error messages bottom-up (the last line usually names the problem); print shapes and a few values before believing any theory; ask an AI assistant for a HINT, not a solution; and type all code yourself — the typos you make and fix are where the learning happens.

## Resources

- [Neural Networks: Zero to Hero playlist](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) — Parts 3 (activations, gradients, BatchNorm) and 4 (backprop ninja) are this lesson's spine.
- [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/) — Karpathy's professional debugging checklist; read before Project 1 and again after Project 2.
- [pytorch.org documentation](https://pytorch.org) — reference for `torch.nn.init`, `torch.optim`, and the LR schedulers as you use them.

## Skills unlocked

- [ ] I can predict what a classifier's loss should be at initialization and know something is wrong when it isn't.
- [ ] I can plot per-layer activation and gradient histograms and diagnose saturation, dead ReLUs, and vanishing/exploding gradients from them.
- [ ] I can explain in plain words why initialization scale matters and what Kaiming init does about it.
- [ ] I can explain what BatchNorm does, why it helps, and the train/eval trap it brings.
- [ ] I can choose between SGD+momentum and Adam with sensible learning rates, and add warmup or a decay schedule.
- [ ] I can apply dropout and weight decay and check whether they helped on *test* data.
- [ ] I follow Karpathy's recipe by reflex: check init loss, overfit one batch, then scale up and regularize.
- [ ] (Ninja) I can derive and verify the gradients of a full network by hand against autograd.

## Next up

You can now train and debug deep networks — time to give them eyes: [26 · Convolutional Neural Networks: Teaching Machines to See](26-cnns-computer-vision.md).
