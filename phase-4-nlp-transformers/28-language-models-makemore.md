# 28 · Language Models 101: makemore (Karpathy 2)

**Phase 4 — NLP and Transformers** · Estimated time: 1-2 weeks · Prerequisites: [22 · Build micrograd](../phase-3-deep-learning/22-micrograd-backpropagation.md), [24 · PyTorch Fundamentals](../phase-3-deep-learning/24-pytorch-fundamentals.md), [25 · Training Deep Nets](../phase-3-deep-learning/25-training-deep-nets.md)

> This is the lesson where you start building the thing that eventually becomes GPT. A **language model** is a program that predicts the next token (here: the next letter) given what came before — that's the whole job. You will build three of them, each better than the last, and use them to invent new human-sounding names out of thin air. First you'll do it by *counting*, then with a *trained neural network*, and you'll watch both arrive at exactly the same answer — one of the most beautiful "aha" moments in all of ML. Hold this thought the entire lesson: GPT plays this exact game, just scaled up a billion-fold.

## What you will build

- **Project 1 — Bigram name generator**: a counting-based model of `names.txt`, a 27×27 probability table visualized as a heatmap, and a sampler that spits out new "names".
- **Project 2 — Neural bigram + MLP**: the same model rebuilt as a one-layer neural net trained by gradient descent, then upgraded to a 3-character-context MLP with embeddings (makemore Part 2). Loss falls from ~2.45 toward ~2.2.
- **Project 3 — Your own corpus**: the MLP retrained on a corpus *you* choose (Indian names, Pokémon, cities, startup names) plus a curated list of your 50 favorite generated inventions.

## Concepts you will learn by doing

- **Language modeling** — predicting the next token; the single idea underneath every chatbot.
- **Character tokenization** — turning letters into integers with `stoi` (string→int) and `itos` (int→string) lookup tables.
- **Bigram counts** — a table of "how often does letter B follow letter A" that *is* a working model.
- **Sampling from a distribution** — using `torch.multinomial` to roll a weighted dice and generate text.
- **Negative log likelihood (NLL)** — the standard loss for "how surprised was my model by the real data".
- **Counts vs learned logits** — gradient descent rediscovers the counting table; they converge to the same answer.
- **Context windows and embeddings** — looking at more than one previous character, and representing characters as small learned vectors.
- **Train/dev/test splits for LMs** — the honesty rules from [lesson 14](../phase-2-classical-ml/14-model-evaluation.md), applied to language.

## Before you start

Check prerequisites: you can build tensors, compute a loss, call `.backward()`, and run a training loop in PyTorch (lesson 24), and you know why we hold out dev/test data (lesson 14).

Everything runs on CPU in seconds — no GPU needed. From the repo root, in your venv:

```bash
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install matplotlib jupyter
```

All of these are likely installed already; if so, both lines are a no-op. PyTorch has a line of its own because the package index it comes from, the same as in lesson 24, does not carry matplotlib or jupyter (if you installed a GPU build in lesson 24, that line leaves it in place). Current PyTorch has no Intel Mac version, so on an Intel Mac do this lesson in Google Colab, as lesson 01 suggested.

Create your work folder and download the dataset (32,000 real names, one per line, no account needed):

```bash
mkdir -p work/28-language-models-makemore
cd work/28-language-models-makemore
curl -L -O https://raw.githubusercontent.com/karpathy/makemore/master/names.txt
wc -l names.txt
head names.txt
```

`wc -l` should print 32032 — it counts line breaks, and the last name has none after it, but Python will still find all 32,033 names. `head` shows the first ten: emma, olivia, ava, ...

You will follow makemore Parts 1 and 2 from the [Karpathy playlist](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ). House rule, same as lesson 22: **type every line yourself, pause the video constantly, never paste.** Work in a Jupyter notebook (`jupyter notebook` from your work folder) so you can inspect tensors as you go.

## Project 1 — Bigram name generator

**Goal**: build a complete working language model with zero neural networks — just counting — and use it to generate new names. This proves the *game* is simple; everything after is just playing it better.

**Milestones**

- [ ] Watch makemore Part 1 (the bigram section) and code along. Load `names.txt` into a Python list of words. Checkpoint: `len(words)` is 32033, `min(len(w) for w in words)` is 2, `max(...)` is 15.
- [ ] Build the character vocabulary: the 26 letters plus one special character `.` used to mark both the start and end of a name (so the model can learn which letters *begin* and *end* names). Build `stoi` (e.g. `stoi['a'] == 1`, `stoi['.'] == 0`) and `itos` (the reverse). This mapping is **tokenization** — converting text to integers a model can eat — in its simplest possible form.
- [ ] A **bigram** is just a pair of adjacent characters. Loop over every word wrapped as `.emma.` and count every bigram into a 27×27 integer tensor `N`, where `N[i, j]` = how many times character `j` followed character `i`. Checkpoint: `N[stoi['.'], stoi['a']]` is 4410 — 4410 names start with "a".
- [ ] Visualize `N` as a heatmap with `plt.imshow`, labeling each cell with the character pair and its count (Karpathy shows how). Stare at it. Checkpoint: the `q` row is nearly empty except `qu` — the model has learned English spelling rules just by counting.
- [ ] Turn counts into probabilities: `P = N.float()` normalized so every **row** sums to 1. Each row is now "given this character, the probability distribution over the next character". Checkpoint: `P[0].sum()` is `1.0`.
- [ ] Generate names: start at `.`, repeatedly use `torch.multinomial` (a weighted random draw — like rolling a 27-sided dice where the sides have different weights) to pick the next character from the current row of `P`, stop when you draw `.` again. Use `torch.Generator().manual_seed(2147483647)` to match the video's outputs. Checkpoint: samples look name-ish but drunk — things like `mor.`, `axx.`, `cexze.`, an occasional accidentally-real `anna`-like string.
- [ ] Score the model with **negative log likelihood**: for every bigram in the data, look up the probability the model assigned to the character that actually came next, take `log`, average, negate. Lower = less surprised = better. Checkpoint: average NLL ≈ **2.454**. Write this number down — it's the score to beat for the rest of the lesson.
- [ ] Add **smoothing**: `P = (N+1).float()` before normalizing, so no bigram has probability exactly 0 (one zero would make the loss infinite the moment a rare pair appears).

<details><summary>Hints</summary>

- Broadcasting bug alert: to normalize rows, divide by `N.sum(1, keepdim=True)`. Without `keepdim=True` you get a silent wrong answer — Karpathy dedicates a whole segment to exactly this trap. Verify with `P.sum(1)` (should be all ones).
- Build the bigram loop with `zip(chs, chs[1:])` where `chs = ['.'] + list(w) + ['.']`.
- For the loss, the pattern is: `probs = P[xs, ys]` (fancy indexing pulls out one probability per bigram), then `-probs.log().mean()`.
- If your samples are pure garbage like `jkqzz`, you are probably sampling from a *column* instead of a *row*, or forgot to normalize.

</details>

**Definition of done**: your heatmap renders, your NLL prints ≈ 2.45, and you can generate 20 drunk-but-name-ish samples with a seeded generator.

## Project 2 — Neural bigram + MLP

**Goal**: replace counting with *learning* — first a one-layer net that provably rediscovers the count table, then makemore Part 2's MLP that reads 3 characters of context and beats it.

**Milestones**

- [ ] Rebuild the bigram model as a neural net (rest of makemore Part 1): inputs are the current character as a **one-hot vector** (a length-27 vector of zeros with a single 1 — recall [lesson 23](../phase-3-deep-learning/23-mlp-numpy-mnist.md)), the only layer is a 27×27 weight matrix `W`, and the outputs are **logits** — raw unnormalized scores that become probabilities after `softmax` (exponentiate, then normalize each row).
- [ ] Train it: forward pass → NLL loss → `loss.backward()` → nudge `W` against the gradient. Exactly the loop you built in micrograd, on exactly the loss you computed by counting. Checkpoint: loss descends and flattens at ≈ **2.45** — the same number as counting. Gradient descent *rediscovered* the count table. Convince yourself why: `W.exp()` behaves like `N`. This is the profound bit — counting and learning are two roads to the same destination, but only learning scales to bigger models.
- [ ] Watch makemore Part 2 and build the MLP. A bigram model is dumb because it sees only 1 character of history. Fix it with a **context window**: the previous 3 characters predict the 4th. Build the dataset with a rolling window over each word. Checkpoint: `X.shape` is `(N, 3)` and `Y.shape` is `(N,)` with N ≈ 228,000 examples.
- [ ] Give each character an **embedding** — a small learned vector (start with 2 numbers) that stands in for the character, instead of a clunky one-hot. Store them in a lookup table `C` of shape `(27, 2)`; then `C[X]` fetches the embeddings for a whole batch in one indexing operation. Checkpoint: `C[X].shape` is `(N, 3, 2)`.
- [ ] Build the network: flatten the 3 embeddings into one vector (`.view(-1, 6)`), pass through a hidden `tanh` layer (start with 100 neurons), then a linear layer producing 27 logits. Use `F.cross_entropy(logits, Y)` for the loss — it is exactly your softmax+NLL, but faster and numerically safer.
- [ ] Train with **minibatches** (a random few-dozen examples per step instead of all 228k — noisier but far faster, as in lesson 25) and pick a learning rate the Karpathy way: sweep it across a range, plot loss vs learning-rate exponent, pick the value just before the curve blows up.
- [ ] Split the data 80/10/10 into **train/dev/test**: train on train, tune hyperparameters (context size, hidden units, embedding size) on dev, touch test exactly once at the very end. With 228k examples memorization is now possible, so the honesty rules apply. Checkpoint: train and dev loss are close to each other (no big overfit).
- [ ] Improve the model until dev loss beats the bigram's 2.45 decisively. Grow the embedding size (2 → 10) and hidden layer, train longer, decay the learning rate. Checkpoint: dev loss around **~2.2 or below**, and samples are noticeably more name-like — `ambrie.`, `khalaysie.`, `jareth.`-style outputs instead of drunk letter soup.
- [ ] Bonus insight: before training, visualize the 2-D embeddings as a scatter plot with character labels. Checkpoint: after training, the vowels `a e i o u` cluster together — the model *learned* they behave alike, nobody told it.

<details><summary>Hints</summary>

- One-hot input times a matrix just *selects a row* of `W` — see this, and the equivalence to the count table stops being mysterious.
- If loss is `nan` or `inf`: you're probably exponentiating huge logits. Switch to `F.cross_entropy`, which handles this internally.
- Sampling from the MLP: keep a rolling context of the last 3 characters (start with `[0, 0, 0]`), feed it through the net, `torch.multinomial` the output, shift the context, repeat until you draw `.`.
- A learning rate of `0.1` with plain gradient descent (`p.data += -lr * p.grad`) is a solid start; remember `p.grad = None` (or zeroing) before each `backward()`.

</details>

**Definition of done**: your neural bigram converges to ≈2.45; your MLP's dev loss is ~2.2 or better; you can sample clearly more name-like names from it; you can explain in two sentences why counting and gradient descent agreed.

## Project 3 — Your own corpus

**Goal**: prove the model is general by retraining it on a corpus you actually care about — the model doesn't know it's making "names", it just learns whatever distribution you feed it.

**Milestones**

- [ ] Pick a corpus: Indian first names, Pokémon names, world city names, startup names, dinosaur names — anything. Find or assemble a plain text file, **one item per line**, ideally 1,000+ lines (search the web for e.g. "list of Indian names txt" or build one from a public dataset you already have from earlier lessons). Save as `mycorpus.txt`.
- [ ] Clean it with Python (lesson 3 skills): lowercase everything, strip whitespace, drop duplicates and blank lines. Decide what to do with spaces/hyphens: either drop those entries or add the characters to your vocabulary. Checkpoint: printing the vocabulary shows every character exactly once, and `len(itos) == len(stoi)`.
- [ ] Point your Project 2 pipeline at the new file. The *only* things that should need to change are the filename and the vocabulary size — if more breaks, you hardcoded `27` somewhere; go generalize it.
- [ ] Retrain (re-tune the learning rate briefly; a smaller corpus may want a smaller network to avoid overfitting — watch the train/dev gap). Checkpoint: dev loss flattens and samples are recognizably in the *style* of your corpus.
- [ ] Generate 50 samples, paste them into a `favorites.md` in your work folder, and mark your top 10. Checkpoint: at least a few are good enough that you'd believe they were real — a fake Pokémon, a plausible startup.
- [ ] Write 5 lines at the bottom of `favorites.md`: what the model captures about your corpus's style, and what it still gets wrong (too short? illegal endings? made-up letter pairs?).

<details><summary>Hints</summary>

- Small corpus (< 2,000 lines) overfitting badly? Shrink the hidden layer and embedding size, or gather more data — the train/dev gap is your gauge.
- Non-English characters are fine: the vocabulary is *built from the data*, so the code doesn't care whether it's 27 symbols or 40.
- If every sample ends after 2-3 characters, check that your corpus lines aren't polluted with short junk entries (initials, headers).

</details>

**Definition of done**: a trained model on your own corpus, `favorites.md` with 50 samples and your top 10, and a 5-line style critique.

## Stretch goals

- **makemore Part 5 (WaveNet)**: follow the video to deepen the MLP into a hierarchical WaveNet-style architecture that fuses characters progressively instead of all at once — dev loss drops below 2.0.
- **Context-size experiment**: train the MLP with context 1, 2, 3, 5, 8 and plot dev loss vs context size. Where does the payoff flatten, and why might that be?
- **Trigram counting model**: extend Project 1's *counting* approach to 2 characters of context (a 27×27×27 tensor). Watch it beat the bigram — then explain why counting can't scale to a context of 10 (hint: count the cells).
- **Temperature knob**: divide the logits by a number `T` before softmax when sampling. `T < 1` makes samples safer and more boring, `T > 1` wilder. Generate at 0.5, 1.0, 1.5 and compare — you've just built the "temperature" setting every LLM API exposes.

## If you get stuck

- Loss stuck exactly at ~3.3 (that's `ln 27`)? Your model is outputting uniform randomness — the gradient isn't flowing. Check you registered all parameters (`requires_grad=True`) and are actually updating them.
- Shape errors are 80% of this lesson's bugs. Print `.shape` at every step of the forward pass and reconcile it with what you *expect* before reading another line of video.
- Silent wrong answers usually mean a broadcasting bug — re-check every `sum`/division for `keepdim`, and every normalization with an explicit `.sum(1)` assert.
- Standing advice: read the error bottom-up; print shapes and a few actual values, not just shapes; ask an AI assistant for a **hint**, not a solution; and type all code yourself — the typos you make and fix *are* the learning.

## Resources

- [Karpathy — Neural Networks: Zero to Hero playlist](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) — makemore Part 1 (bigrams) and Part 2 (MLP) are your core material; Part 5 (WaveNet) is the stretch goal.
- [makemore repository](https://github.com/karpathy/makemore) — the reference code and `names.txt`; consult *after* your own attempt, never during.
- [pytorch.org](https://pytorch.org) — docs for `torch.multinomial`, `F.cross_entropy`, and tensor indexing when you need the exact API.

## Skills unlocked

- [ ] I can explain what a language model does in one sentence, and why GPT is the same game scaled up.
- [ ] I can tokenize text at the character level and build `stoi`/`itos` tables for any corpus.
- [ ] I can build, normalize, and sample from a bigram probability table with `torch.multinomial`.
- [ ] I can compute negative log likelihood and use it to compare language models fairly.
- [ ] I can explain why a trained one-layer net and a count table converge to the same answer.
- [ ] I can build an embedding table and a context-window MLP, and train it with minibatches on proper train/dev/test splits.
- [ ] I can retrain the whole pipeline on a new corpus by changing only the data.

## Next up

Your MLP's context window is stuck at a fixed 3 characters — next, you'll build networks with *memory* that carry information across arbitrarily long sequences: [29 · RNNs and LSTMs: Networks with Memory](29-rnn-lstm-char-generation.md).
