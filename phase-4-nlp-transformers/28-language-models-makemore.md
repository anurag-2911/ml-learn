# 28 · Language Models 101: makemore (Karpathy 2)

**Phase 4 — NLP and Transformers** · Estimated time: 1-2 weeks · Prerequisites: [22 · Build micrograd](../phase-3-deep-learning/22-micrograd-backpropagation.md), [24 · PyTorch Fundamentals](../phase-3-deep-learning/24-pytorch-fundamentals.md), [25 · Training Deep Nets](../phase-3-deep-learning/25-training-deep-nets.md)

> This lesson starts building the thing that eventually becomes GPT. A **language model** is a program that predicts the next token (here, the next letter) given what came before; that is its whole job. The lesson builds three of them, each better than the last, and uses them to invent new human-sounding names. The first model works by *counting* and the second is a *trained neural network*; both arrive at exactly the same answer. Keep one idea in mind for the whole lesson: GPT plays this exact game, only scaled up a billion-fold.

## What this lesson builds

- **Project 1 — Bigram name generator**: a counting-based model of `names.txt`, a 27×27 probability table visualized as a heatmap, and a sampler that produces new "names".
- **Project 2 — Neural bigram + MLP**: the same model rebuilt as a one-layer neural net trained by gradient descent, then upgraded to a 3-character-context MLP with embeddings (makemore Part 2). Loss falls from ~2.45 toward ~2.2.
- **Project 3 — A custom corpus**: the MLP retrained on a freely chosen corpus (Indian names, Pokémon, cities, startup names) plus a curated list of 50 favorite generated inventions.

## Concepts covered

- **Language modeling**: predicting the next token; the single idea underneath every chatbot.
- **Character tokenization**: turning letters into integers with `stoi` (string→int) and `itos` (int→string) lookup tables.
- **Bigram counts**: a table of "how often does letter B follow letter A" that *is* a working model.
- **Sampling from a distribution**: using `torch.multinomial` to roll a weighted dice and generate text.
- **Negative log likelihood (NLL)**: the standard loss for "how surprised was the model by the real data".
- **Counts vs learned logits**: gradient descent rediscovers the counting table; they converge to the same answer.
- **Context windows and embeddings**: looking at more than one previous character, and representing characters as small learned vectors.
- **Train/dev/test splits for LMs**: the honesty rules from [lesson 14](../phase-2-classical-ml/14-model-evaluation.md), applied to language.

## Before starting

Check the prerequisites: building tensors, computing a loss, calling `.backward()` and running a training loop in PyTorch (lesson 24) should all be familiar, and so should the reason for holding out dev/test data (lesson 14).

Everything runs on the CPU in seconds; no GPU is needed. From the repo root, in the venv:

```bash
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install matplotlib jupyter
```

All of these are likely installed already; if so, both lines change nothing. PyTorch has a line of its own because the package index it comes from, the same as in lesson 24, does not carry matplotlib or jupyter (if a GPU build was installed in lesson 24, that line leaves it in place). Current PyTorch has no Intel Mac version, so on an Intel Mac do this lesson in Google Colab, as lesson 01 suggested.

Create the work folder and download the dataset (32,000 real names, one per line, no account needed):

```bash
mkdir -p work/28-language-models-makemore
cd work/28-language-models-makemore
curl -L -O https://raw.githubusercontent.com/karpathy/makemore/master/names.txt
wc -l names.txt
head names.txt
```

`wc -l` should print 32032. It counts line breaks, and the last name has none after it, but Python still finds all 32,033 names. `head` shows the first ten: emma, olivia, ava, ...

The lesson follows makemore Parts 1 and 2 from the [Karpathy playlist](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ). The house rule is the same as in lesson 22: **type every line by hand, pause the video constantly, never paste.** Work in a Jupyter notebook (`jupyter notebook` from the work folder), so that tensors can be inspected along the way.

## Project 1 — Bigram name generator

**Goal**: build a complete working language model with zero neural networks (just counting) and use it to generate new names. This proves that the *game* is simple; everything that follows is just playing it better.

**Milestones**

- [ ] Watch makemore Part 1 (the bigram section) and code along. Load `names.txt` into a Python list of words. Checkpoint: `len(words)` is 32033, `min(len(w) for w in words)` is 2, `max(...)` is 15.
- [ ] Build the character vocabulary: the 26 letters plus one special character `.` used to mark both the start and end of a name (so the model can learn which letters *begin* and *end* names). Build `stoi` (e.g. `stoi['a'] == 1`, `stoi['.'] == 0`) and `itos` (the reverse). This mapping is the simplest possible form of **tokenization**: converting text into integers that a model can use.
- [ ] A **bigram** is just a pair of adjacent characters. Loop over every word wrapped as `.emma.` and count every bigram into a 27×27 integer tensor `N`, where `N[i, j]` = how many times character `j` followed character `i`. Checkpoint: `N[stoi['.'], stoi['a']]` is 4410, meaning that 4410 names start with "a".
- [ ] Visualize `N` as a heatmap with `plt.imshow`, labeling each cell with the character pair and its count (Karpathy shows how). Look at it closely. Checkpoint: the `q` row is nearly empty except for `qu`. The model has learned English spelling rules just by counting.
- [ ] Turn counts into probabilities: `P = N.float()` normalized so every **row** sums to 1. Each row is now "given this character, the probability distribution over the next character". Checkpoint: `P[0].sum()` is `1.0`.
- [ ] Generate names: start at `.`, repeatedly use `torch.multinomial` (a weighted random draw, like rolling a 27-sided dice whose sides have different weights) to pick the next character from the current row of `P`, and stop when `.` is drawn again. Use `torch.Generator().manual_seed(2147483647)` to match the video's outputs. Checkpoint: the samples look name-like but garbled (`mor.`, `axx.`, `cexze.`), with an occasional string like `anna` that happens to be a real name.
- [ ] Score the model with **negative log likelihood**: for every bigram in the data, look up the probability the model assigned to the character that actually came next, take `log`, average, negate. Lower = less surprised = better. Checkpoint: average NLL ≈ **2.454**. Write this number down: it is the score to beat for the rest of the lesson.
- [ ] Add **smoothing**: `P = (N+1).float()` before normalizing, so no bigram has probability exactly 0 (one zero would make the loss infinite the moment a rare pair appears).

<details><summary>Hints</summary>

- Watch out for a broadcasting bug: to normalize rows, divide by `N.sum(1, keepdim=True)`. Without `keepdim=True`, the answer is silently wrong; Karpathy dedicates a whole segment to exactly this trap. Verify with `P.sum(1)` (should be all ones).
- Build the bigram loop with `zip(chs, chs[1:])` where `chs = ['.'] + list(w) + ['.']`.
- For the loss, the pattern is: `probs = P[xs, ys]` (fancy indexing pulls out one probability per bigram), then `-probs.log().mean()`.
- If the samples are pure garbage like `jkqzz`, the code probably samples from a *column* instead of a *row*, or skips the normalization.

</details>

**Definition of done**: the heatmap renders, the NLL prints ≈ 2.45, and a seeded generator produces 20 garbled-but-name-like samples.

## Project 2 — Neural bigram + MLP

**Goal**: replace counting with *learning*. First comes a one-layer net that provably rediscovers the count table, then makemore Part 2's MLP, which reads 3 characters of context and beats it.

**Milestones**

- [ ] Rebuild the bigram model as a neural net (rest of makemore Part 1): inputs are the current character as a **one-hot vector** (a length-27 vector of zeros with a single 1; see [lesson 23](../phase-3-deep-learning/23-mlp-numpy-mnist.md)), the only layer is a 27×27 weight matrix `W`, and the outputs are **logits**, raw unnormalized scores that become probabilities after `softmax` (exponentiate, then normalize each row).
- [ ] Train it: forward pass → NLL loss → `loss.backward()` → nudge `W` against the gradient. This is exactly the loop built in micrograd, on exactly the loss computed by counting. Checkpoint: the loss descends and flattens at ≈ **2.45**, the same number as counting. Gradient descent *rediscovered* the count table. Work out why: `W.exp()` behaves like `N`. This is the central idea: counting and learning are two roads to the same destination, but only learning scales to bigger models.
- [ ] Watch makemore Part 2 and build the MLP. A bigram model is limited because it sees only 1 character of history. Fix it with a **context window**: the previous 3 characters predict the 4th. Build the dataset with a rolling window over each word. Checkpoint: `X.shape` is `(N, 3)` and `Y.shape` is `(N,)` with N ≈ 228,000 examples.
- [ ] Give each character an **embedding**: a small learned vector (start with 2 numbers) that stands in for the character, instead of a clunky one-hot. Store them in a lookup table `C` of shape `(27, 2)`; then `C[X]` fetches the embeddings for a whole batch in one indexing operation. Checkpoint: `C[X].shape` is `(N, 3, 2)`.
- [ ] Build the network: flatten the 3 embeddings into one vector (`.view(-1, 6)`), pass through a hidden `tanh` layer (start with 100 neurons), then a linear layer producing 27 logits. Use `F.cross_entropy(logits, Y)` for the loss. It is exactly the same softmax+NLL as before, but faster and numerically safer.
- [ ] Train with **minibatches** (a random few-dozen examples per step instead of all 228k; noisier but far faster, as in lesson 25) and pick a learning rate the Karpathy way: sweep it across a range, plot loss vs learning-rate exponent, pick the value just before the curve blows up.
- [ ] Split the data 80/10/10 into **train/dev/test**: train on train, tune hyperparameters (context size, hidden units, embedding size) on dev, touch test exactly once at the very end. With 228k examples memorization is now possible, so the honesty rules apply. Checkpoint: train and dev loss are close to each other (no big overfit).
- [ ] Improve the model until dev loss beats the bigram's 2.45 decisively. Grow the embedding size (2 → 10) and hidden layer, train longer, decay the learning rate. Checkpoint: dev loss around **~2.2 or below**, and samples are noticeably more name-like (outputs in the style of `ambrie.`, `khalaysie.` and `jareth.` instead of garbled letter soup).
- [ ] Bonus insight: before training, visualize the 2-D embeddings as a scatter plot with character labels. Checkpoint: after training, the vowels `a e i o u` cluster together. The model *learned* that they behave alike; nobody told it.

<details><summary>Hints</summary>

- A one-hot input times a matrix just *selects a row* of `W`. Once this is clear, the equivalence to the count table stops being mysterious.
- If the loss is `nan` or `inf`, the code is probably exponentiating huge logits. Switch to `F.cross_entropy`, which handles this internally.
- Sampling from the MLP: keep a rolling context of the last 3 characters (start with `[0, 0, 0]`), feed it through the net, `torch.multinomial` the output, shift the context, and repeat until `.` is drawn.
- A learning rate of `0.1` with plain gradient descent (`p.data += -lr * p.grad`) is a solid start; remember `p.grad = None` (or zeroing) before each `backward()`.

</details>

**Definition of done**: the neural bigram converges to ≈2.45; the MLP's dev loss is ~2.2 or better; samples from it are clearly more name-like; and the reason counting and gradient descent agreed can be explained in two sentences.

## Project 3 — A custom corpus

**Goal**: prove that the model is general by retraining it on a corpus of genuine personal interest. The model does not know it is making "names"; it just learns whatever distribution it is fed.

**Milestones**

- [ ] Pick a corpus: Indian first names, Pokémon names, world city names, startup names, dinosaur names, or anything else. Find or assemble a plain text file, **one item per line**, ideally 1,000+ lines (search the web for e.g. "list of Indian names txt", or build one from a public dataset already on hand from earlier lessons). Save it as `mycorpus.txt`.
- [ ] Clean it with Python (lesson 3 skills): lowercase everything, strip whitespace, drop duplicates and blank lines. Decide what to do with spaces/hyphens: either drop those entries or add the characters to the vocabulary. Checkpoint: printing the vocabulary shows every character exactly once, and `len(itos) == len(stoi)`.
- [ ] Point the Project 2 pipeline at the new file. The *only* things that should need to change are the filename and the vocabulary size. If more breaks, `27` is hardcoded somewhere; generalize it.
- [ ] Retrain (re-tune the learning rate briefly; a smaller corpus may want a smaller network to avoid overfitting; watch the train/dev gap). Checkpoint: dev loss flattens and samples are recognizably in the *style* of the corpus.
- [ ] Generate 50 samples, paste them into a `favorites.md` in the work folder, and mark the top 10. Checkpoint: at least a few are good enough to pass for real, such as a fake Pokémon or a plausible startup.
- [ ] Write 5 lines at the bottom of `favorites.md`: what the model captures about the corpus's style, and what it still gets wrong (too short? illegal endings? made-up letter pairs?).

<details><summary>Hints</summary>

- Small corpus (< 2,000 lines) overfitting badly? Shrink the hidden layer and embedding size, or gather more data. The train/dev gap is the gauge.
- Non-English characters are fine: the vocabulary is *built from the data*, so the code does not care whether it is 27 symbols or 40.
- If every sample ends after 2-3 characters, check that the corpus lines are not polluted with short junk entries (initials, headers).

</details>

**Definition of done**: a model trained on the custom corpus, `favorites.md` with 50 samples and the top 10, and a 5-line style critique.

## Stretch goals

- **makemore Part 5 (WaveNet)**: follow the video to deepen the MLP into a hierarchical WaveNet-style architecture that fuses characters progressively instead of all at once. The dev loss drops below 2.0.
- **Context-size experiment**: train the MLP with context 1, 2, 3, 5, 8 and plot dev loss vs context size. Where does the payoff flatten, and why might that be?
- **Trigram counting model**: extend Project 1's *counting* approach to 2 characters of context (a 27×27×27 tensor). Watch it beat the bigram, then explain why counting cannot scale to a context of 10 (hint: count the cells).
- **Temperature knob**: divide the logits by a number `T` before softmax when sampling. `T < 1` makes samples safer and more boring, `T > 1` wilder. Generate at 0.5, 1.0, 1.5 and compare. This knob is the "temperature" setting that every LLM API exposes.

## Getting unstuck

- Loss stuck exactly at ~3.3 (that is `ln 27`)? The model is outputting uniform randomness; the gradient is not flowing. Check that all parameters are registered (`requires_grad=True`) and are actually being updated.
- Shape errors are 80% of this lesson's bugs. Print `.shape` at every step of the forward pass and reconcile it with the *expected* shape before moving on to the next line of the video.
- Silent wrong answers usually mean a broadcasting bug. Re-check every `sum`/division for `keepdim`, and every normalization with an explicit `.sum(1)` assert.
- Standing advice: read the error bottom-up; print shapes and a few actual values, not just shapes; ask an AI assistant for a **hint**, not a solution; and type all code by hand, because the typos made and fixed along the way *are* the learning.

## Resources

- [Karpathy — Neural Networks: Zero to Hero playlist](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) — makemore Part 1 (bigrams) and Part 2 (MLP) are the core material; Part 5 (WaveNet) is the stretch goal.
- [makemore repository](https://github.com/karpathy/makemore) — the reference code and `names.txt`; consult it *after* an independent attempt, never during.
- [pytorch.org](https://pytorch.org) — docs for `torch.multinomial`, `F.cross_entropy`, and tensor indexing when the exact API is needed.

## Skills unlocked

- [ ] I can explain what a language model does in one sentence, and why GPT is the same game scaled up.
- [ ] I can tokenize text at the character level and build `stoi`/`itos` tables for any corpus.
- [ ] I can build, normalize, and sample from a bigram probability table with `torch.multinomial`.
- [ ] I can compute negative log likelihood and use it to compare language models fairly.
- [ ] I can explain why a trained one-layer net and a count table converge to the same answer.
- [ ] I can build an embedding table and a context-window MLP, and train it with minibatches on proper train/dev/test splits.
- [ ] I can retrain the whole pipeline on a new corpus by changing only the data.

## Next up

The MLP's context window is stuck at a fixed 3 characters, so the next lesson builds networks with *memory* that carry information across arbitrarily long sequences: [29 · RNNs and LSTMs: Networks with Memory](29-rnn-lstm-char-generation.md).
