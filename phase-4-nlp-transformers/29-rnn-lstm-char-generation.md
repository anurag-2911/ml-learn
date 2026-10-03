# 29 · RNNs and LSTMs: Networks with Memory

**Phase 4 — NLP and Transformers** · Estimated time: 1-2 weeks · Prerequisites: [24 · PyTorch Fundamentals](../phase-3-deep-learning/24-pytorch-fundamentals.md), [25 · The Dark Arts of Training Deep Networks](../phase-3-deep-learning/25-training-deep-nets.md), [28 · Language Models 101: makemore](28-language-models-makemore.md)

> Every network built so far in this course has amnesia: it sees one input, makes one prediction, and forgets everything. But language, music, and time series are *sequences*: what comes next depends on what came before. This lesson gives a network memory by training a character-level LSTM on Tiny Shakespeare, about 1 MB of dialogue drawn from Shakespeare's plays. Samples saved during training show the model learning to write: random noise at first, then real English words, then dialogue with character names and line breaks in the right places. Transformers (lesson 31) have since replaced RNNs for most of this work, but RNNs teach how to *think in sequences*, and running into their limits first shows why attention is needed.

## What this lesson builds

- **A Shakespeare generator:** a character-level LSTM trained on Tiny Shakespeare, plus saved text samples from epoch 1, 5, and 20 showing it going from gibberish to Shakespeare-shaped dialogue.
- **A temperature dial:** a sampling function with a temperature knob, and a short write-up comparing generations at 0.3, 0.8, and 1.5 (the same knob that every LLM API exposes, as lesson 34 shows).
- **A surname nationality classifier:** a char-RNN that reads a name letter by letter and predicts its language of origin, evaluated with accuracy and an 18×18 confusion matrix.

## Concepts covered

- **Hidden state**: a vector the network carries from step to step; its "memory" of everything seen so far.
- **RNN cell mechanics**: the *same* weights are applied at every time step; only the hidden state changes.
- **Backprop through time (BPTT)**: unrolling the loop into one long chain of operations, then backpropagating through it like any other network.
- **Vanishing gradients over long sequences**: why plain RNNs forget (the same multiplied-gradients problem from lesson 25, now multiplied once per time step).
- **LSTM/GRU gating**: learned gates that decide what to keep, what to forget, and what to output, keeping gradients alive.
- **Sampling with temperature**: scaling the logits before softmax to trade safety for creativity.
- **`nn.RNN` / `nn.LSTM` in PyTorch**: the ready-made recurrent layers and their (slightly weird) tensor shapes.

## Before starting

Check that these skills are still in place. If not, revisit the linked lessons:

- Build and train a model with the lesson-24 training-loop template (model, loss, optimizer, loop).
- Explain what cross-entropy loss measures (lesson 13/24) and what a learning-rate that is too high looks like (lesson 25).
- Recall the bigram and MLP name generators from lesson 28. This lesson is the same idea with real memory.

Set up the work folder and data (the venv lives at the repo root). PyTorch and matplotlib are probably installed already; if not, the pip lines install torch from PyTorch's own package index, as lesson 24 did, and matplotlib with a separate command, because that index does not carry it:

```bash
cd ~/ml/ml-learn
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install matplotlib
mkdir -p work/29-rnn-lstm
cd work/29-rnn-lstm
curl -L -o input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
wc -c input.txt
```

`wc -c input.txt` should print about 1115394 (~1.1 MB). On an Intel Mac, current PyTorch cannot be installed at all, so do this lesson in a free Google Colab notebook, as lesson 01 suggested.

For Project 3, download the surname dataset used by the official PyTorch tutorial:

```bash
cd ~/ml/ml-learn/work/29-rnn-lstm
curl -L -O https://download.pytorch.org/tutorial/data.zip
unzip data.zip
```

`unzip` creates `data/names/*.txt`: 18 files, one language each. If the download link ever moves, the zip is linked from the top of the [char-RNN classification tutorial](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html).

This lesson has one assigned reading, Karpathy's [The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/). It is short and a pleasure to read. Read it *after* Project 1's first milestone, so that the generated samples in the post mean something.

Everything trains fine on CPU if the model stays modest (hidden size 128-256). Expect minutes per epoch, not seconds.

## Project 1 — Tiny Shakespeare

**Goal:** Train a character-level LSTM that predicts the next character of Shakespeare, then sample from it and watch its writing improve as training progresses.

**Milestones**

- [ ] Load `input.txt` into one Python string. Build the vocabulary: the sorted list of unique characters, plus two dicts `stoi` (char → int) and `itos` (int → char), exactly like lesson 28. Encode the whole text as one long tensor of ints. Checkpoint: vocab size is 65, and `itos[stoi['a']] == 'a'`.
- [ ] Build the data pipeline: a function that returns a batch of random `(chunk, target)` pairs, where `chunk` is `seq_len` consecutive encoded characters and `target` is the same window shifted one character right. That shift is the whole supervision trick: at every position, the label is simply "the next character". Checkpoint: decode one pair and check that the target text is the input text moved over by one letter.

```python
def get_batch(data, batch_size=64, seq_len=100):
    ix = torch.randint(len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[i : i + seq_len] for i in ix])
    y = torch.stack([data[i + 1 : i + seq_len + 1] for i in ix])
    return x, y   # both shape (batch_size, seq_len)
```

- [ ] Define the model as an `nn.Module` with three layers: `nn.Embedding(vocab_size, emb_dim)` to turn char ids into vectors, `nn.LSTM(emb_dim, hidden_size, batch_first=True)`, and `nn.Linear(hidden_size, vocab_size)` to produce logits for every position. `forward` should accept and return the hidden state, so that memory can be carried between calls. Start with `emb_dim=64, hidden_size=256`. Checkpoint: feeding a `(64, 100)` batch returns logits of shape `(64, 100, 65)`.
- [ ] Before training, compute the loss on one batch. Checkpoint: it is close to `ln(65) ≈ 4.17`, the loss of pure random guessing over 65 characters. (This is the same sanity check as in lesson 25.)
- [ ] Train with the lesson-24 loop: cross-entropy on the logits (reshape `(B, T, 65)` → `(B*T, 65)` and the targets to `(B*T,)`), Adam with `lr=3e-3`, and define one "epoch" as `len(data) // (batch_size * seq_len)` batches (one full pass worth of text). Print the loss every 100 steps. Checkpoint: the loss drops below 2.0 within the first epoch and keeps falling; below ~1.5 after 20 epochs is a good run.
- [ ] Pause to understand what the machinery is doing. The LSTM applies the *same* weights at every one of the 100 time steps, passing a hidden vector forward, and backprop *unrolls* that loop into a 100-layer-deep chain and pushes gradients back through it. That is backprop through time. A plain `nn.RNN` multiplies the gradient by roughly the same matrix at each of those steps, so it shrinks toward zero: the vanishing-gradient problem from lesson 25, once per character. The LSTM's gates (little learned sigmoids deciding "keep this, forget that") give gradients a protected path, which is why it can remember an opening quote 80 characters later.
- [ ] Write a `sample(model, length)` function: start from a newline character, run one step, softmax the logits into probabilities, pick the next char with `torch.multinomial`, feed it back in and, crucially, keep passing the hidden state forward. Wrap it in `torch.no_grad()`. Checkpoint: it returns text without crashing, even from an untrained model (the text will be noise).
- [ ] Generate and save a ~1000-character sample after epoch 1, epoch 5, and epoch 20 into `samples_epoch01.txt`, `samples_epoch05.txt`, `samples_epoch20.txt`. After epoch 20, also save the weights with `torch.save(model.state_dict(), "shakespeare_lstm.pt")` (lesson 24); Project 2 loads them. Checkpoint: epoch 1 is gibberish, perhaps with some short real words; epoch 5 has mostly real English words and line structure; epoch 20 has Shakespeare-shaped dialogue (speaker names, mostly CAPITALIZED, on their own line and followed by a colon, then a few lines of verse), even though the sentences are nonsense. Tiny Shakespeare contains no stage directions, so none appear in the samples.
- [ ] Put the three samples side by side in a short note in the work folder. What did the model learn first: spelling, words, or structure? (If the Karpathy blog post is still unread, read it now; its samples section will feel like déjà vu.)

<details><summary>Hints</summary>

- With `batch_first=True`, `nn.LSTM` takes input `(batch, seq, features)` and its hidden state is a *tuple* `(h, c)` of two tensors shaped `(num_layers, batch, hidden_size)`. Passing `None` as the hidden state means "start from zeros", which is fine for training on random chunks.
- The reshape for the loss: `loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))`. Cross-entropy averages over all batch × time positions at once.
- If a hidden state is ever carried across training batches, call `.detach()` on it first. Otherwise PyTorch tries to backprop into the previous batch's graph and throws a "backward through the graph a second time" error. (With independent random chunks and `None`, the problem never comes up.)
- During sampling the input is one character at a time: shape `(1, 1)`. Take the logits at the last (only) time step: `logits[:, -1, :]`.

</details>

**Definition of done:** Three saved sample files showing clear progression from noise to structured pseudo-Shakespeare, and a training run whose final loss is under ~1.6.

## Project 2 — The temperature dial

**Goal:** Add a temperature parameter to the sampler and see, in the model's own generated text, the trade-off between boring-and-safe and creative-and-chaotic.

**Milestones**

- [ ] Modify `sample()` to accept `temperature`, and divide the logits by it *before* softmax: `probs = F.softmax(logits / temperature, dim=-1)`. Temperature below 1 sharpens the distribution (the top choice gets even more likely); above 1 flattens it (long-shot characters get a real chance); exactly 1 is the Project 1 sampler unchanged.
- [ ] Rebuild the model, load the epoch-20 weights with `model.load_state_dict(torch.load("shakespeare_lstm.pt"))`, and generate ~500 characters at temperature 0.3, 0.8, and 1.5. Save all three to `temperature_comparison.txt`. Checkpoint: 0.3 is repetitive and safe (correctly spelled, common words, loops in phrasing); 0.8 reads best; 1.5 is chaotic, with misspelled and invented words.
- [ ] Write 3-4 original sentences in that file describing the trade-off: what is gained and what is lost as the dial turns up? When would each setting be the right choice?
- [ ] Remember this: `temperature` in the OpenAI/Anthropic/every-LLM API (lesson 34) is *exactly this line of code*. This project shows what the knob physically does to the probabilities.

<details><summary>Hints</summary>

- Temperature 0.3 makes small logit gaps huge after division. Think about what dividing by a small number does to the *differences* between logits before the softmax exponentiates them.
- Try temperature 0.01 as an extra experiment: sampling becomes nearly deterministic, almost `argmax` at every step. Watch it get stuck in a loop.

</details>

**Definition of done:** `temperature_comparison.txt` contains three clearly different generations plus a written comparison of the trade-off.

## Project 3 — Name nationality classifier

**Goal:** Switch from generation to sequence *classification*: read a surname one character at a time and predict which of 18 languages it comes from. The recurrent machinery is the same; only the head is different.

**Milestones**

- [ ] Load `data/names/*.txt`: each file is a language, each line a surname. Normalize the names to plain ASCII (the tutorial page shows a `unicodeToAscii` helper using the `unicodedata` module; understand it, then write a version by hand). Build the char vocabulary from the data and a `label` int for each language. Checkpoint: 18 languages, ~20,000 names total, and "Ślusàrski" normalizes to "Slusarski".
- [ ] First drop exact duplicates within each language (`sorted(set(names))`): Arabic.txt repeats its 108 names about 18 times each, so without this step the same name lands in both train and test. Checkpoint: about 18,000 names remain. Then shuffle and split 80/20 into train and test sets *before* looking at accuracy; the lesson-14 rules still apply. Note that the classes are very imbalanced (Russian and English are huge, Korean is tiny), and record the class counts.
- [ ] Build the classifier: embedding → `nn.LSTM` → take the hidden state after the *last* character → `nn.Linear(hidden_size, 18)`. Key difference from Project 1: one prediction per *sequence*, not per step. The final hidden vector is a summary of the whole name. Train with cross-entropy; the simplest correct version processes one name at a time (batching variable-length names needs padding; skip it, or see the stretch goal). Checkpoint: initial loss ≈ `ln(18) ≈ 2.89`, and it falls steadily.
- [ ] Evaluate on the test set. Copy the lesson-14 metrics library here (`cp ../14-model-evaluation/mymetrics.py .`) and use its `accuracy` for overall accuracy. Its `confusion_matrix` handles only two classes, so build the 18×18 matrix as lesson 23 did (`cm = np.zeros((18, 18), dtype=int)`, then `np.add.at(cm, (y_true, y_pred), 1)`) and check it once against `sklearn.metrics.confusion_matrix`. Render it with `matplotlib.pyplot.matshow` (lesson 7 skills). Checkpoint: accuracy above 0.6, and the matrix has a bright diagonal. The 1/18 ≈ 5.6% random baseline is the wrong comparison here: Russian is about half of all names, so a model that always answers "Russian" already scores about 0.5 (lesson 14's Dr. Useless again).
- [ ] Read the confusion matrix like a detective: which pairs get confused? Checkpoint: name at least two confusable pairs and explain them (e.g. English/Scottish share spelling patterns; Spanish/Portuguese/Italian blur together), and name two easy classes (Greek and Japanese, with their distinctive romanization). Expect Korean to be one of the hardest classes, often predicted as Chinese: many Korean surnames, such as Kang, Song and Wang, also appear word for word in the Chinese file.
- [ ] Try the learner's own surname and five friends' names. Does the model's guess make sense? Print the top-3 predictions with their probabilities, not just the winner.

<details><summary>Hints</summary>

- For one-name-at-a-time training, a name of length L becomes input shape `(1, L)` after encoding. `nn.LSTM` returns `output, (h, c)`; for the last layer's final hidden state, use `h[-1]`, shape `(1, hidden_size)`.
- One name per optimizer step is noisy; a smaller learning rate (`1e-3`) and 2-3 passes over the training set help. Or accumulate the loss over 32 names before calling `backward()`, which works as a cheap stand-in for a batch.
- If accuracy sits near 0.5 and the matrix shows almost everything predicted as "Russian", that is class imbalance from lesson 14 in the wild. Balanced sampling (draw a random language, then a random name from that language's training names) fixes it, at the price of putting *per-class* fairness ahead of raw accuracy.

</details>

**Definition of done:** Test accuracy above 0.6 (clearly above the ~0.5 always-"Russian" baseline) with a plotted confusion matrix, plus a few sentences naming the confusable language pairs and why.

## Stretch goals

- **GRU shootout:** swap `nn.LSTM` for `nn.GRU` (a simpler gated cell: two gates instead of three, no separate cell state) in Project 1. Keep the same number of epochs, and compare the final loss and sample quality.
- **Build the LSTM cell from scratch:** implement one LSTM step from `nn.Linear` and sigmoids/tanh (forget gate, input gate, output gate, cell state), and verify it against `nn.LSTMCell` on the same weights. This is the micrograd spirit from lesson 22 applied here.
- **A custom corpus:** retrain Project 1 on any plain text of choice: song lyrics, a favorite public-domain book, or code from a personal project. The samples then show the model learning the structure of that particular text.
- **Proper batching for Project 3:** pad names to equal length within a batch and use `torch.nn.utils.rnn.pack_padded_sequence` so that the LSTM ignores the padding. It is fiddly, instructive, and much faster.

## Getting unstuck

- **Shape errors around the LSTM** are this lesson's #1 time sink. Print `x.shape` before and after every layer. Ninety percent of the time, the cause is a missing `batch_first=True` or forgetting that the hidden state is a tuple `(h, c)`.
- **"Trying to backward through the graph a second time":** a hidden state was carried between batches without `.detach()`. See the Project 1 hints.
- **Loss goes to NaN or explodes:** the learning rate is too high, or the gradients are exploding (lesson 25). Drop `lr` to `1e-3` and add `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)` after `backward()`.
- **Samples repeat one character forever:** the sampler uses `argmax` instead of `torch.multinomial`, or does not pass the hidden state forward between sampling steps.
- Standing advice: read the traceback from the bottom up; print shapes and a few actual values, not just "it's wrong"; when truly stuck, ask an AI assistant for a *hint*, never the solution; and type every line by hand, with no pasting.

## Resources

- [The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) — the classic Karpathy post and this lesson's one assigned reading, with famous generated samples (Shakespeare, LaTeX, Linux source).
- [PyTorch char-RNN classification tutorial](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html) — source of Project 3's dataset and the `unicodeToAscii` idea; peek at its approach *after* attempting one from scratch.
- `nn.LSTM` API reference — search "nn.LSTM" in the docs at [pytorch.org](https://pytorch.org) for exact input/output shapes and the meaning of `num_layers`.

## Skills unlocked

- [ ] I can explain what a hidden state is and why sequence models need one.
- [ ] I can describe how an RNN applies the same weights at every time step, and what backprop through time unrolls.
- [ ] I can explain why plain RNNs forget over long sequences and how LSTM gates keep gradients alive.
- [ ] I can build a `(chunk, shifted-chunk)` next-character data pipeline from raw text.
- [ ] I can train an `nn.LSTM` model in PyTorch and sanity-check its starting loss against `ln(vocab_size)`.
- [ ] I can sample from a language model with temperature and predict how the output changes as I turn the dial.
- [ ] I can use the final hidden state for sequence classification and read the resulting confusion matrix.

## Next up

The LSTM in this lesson turned characters into vectors with `nn.Embedding` without asking why that works; the next lesson builds word2vec from scratch and shows that these learned vectors *are meaning as geometry*: [30 · Embeddings: Meaning as Geometry (word2vec from Scratch)](30-embeddings-word2vec.md).
