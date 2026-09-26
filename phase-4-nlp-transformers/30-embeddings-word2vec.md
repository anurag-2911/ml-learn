# 30 · Embeddings: Meaning as Geometry (word2vec from Scratch)

**Phase 4 — NLP & Transformers** · Estimated time: 1 week · Prerequisites: [28 · Language Models 101: makemore](28-language-models-makemore.md), [24 · PyTorch Fundamentals](../phase-3-deep-learning/24-pytorch-fundamentals.md), [18 · Unsupervised Learning: k-Means and PCA](../phase-2-classical-ml/18-unsupervised-kmeans-pca.md), [08 · Linear Algebra by Writing Your Own](../phase-1-math/08-linear-algebra-by-code.md)

> In makemore you used `nn.Embedding` as a lookup table that turns a character into a vector — and it just worked, no questions asked. This lesson answers the question you should have been asking: *why do those learned vectors capture meaning?* You will train a small network on a few novels until the word vectors arrange themselves so that geometric distance equals semantic similarity — and then compute `king - man + woman` and watch `queen` fall out of pure arithmetic. This is arguably the most beautiful idea in NLP, and it is the direct ancestor of the embeddings that power semantic search and RAG (lesson 35) inside every modern LLM system.

## What you will build

- **Project 1 — `word2vec.py`**: a skip-gram-with-negative-sampling model, trained from scratch in PyTorch on a free Project Gutenberg corpus, plus a `nearest(word)` tool that prints a word's closest neighbors.
- **Project 2 — `analogy.py` + `word_map.png`**: a word-arithmetic function `analogy(a, b, c)` tested on classic analogies, and a 2D PCA map of 200 common words with the clusters you find circled and labeled.
- **Project 3 — `semantic_search.py`**: pretrained GloVe vectors loaded with gensim, compared head-to-head with your own vectors, then used to build a crude semantic search over sentences — a working preview of RAG.

## Concepts you will learn by doing

- **One-hot vs dense vectors** — representing a word as a giant vector of zeros with a single 1, versus a short learned vector of real numbers.
- **The distributional hypothesis** — "you shall know a word by the company it keeps": words in similar contexts have similar meanings.
- **The skip-gram task** — train a network to predict a word's neighbors, and meaning emerges as a side effect.
- **Negative sampling** — a trick that replaces an impossibly slow softmax over the whole vocabulary with a handful of cheap yes/no questions.
- **Cosine similarity** — measuring the angle between vectors to compare meanings (you built this in lesson 08, project 3).
- **Analogies as vector arithmetic** — relationships like gender or plurality become *directions* in embedding space.
- **Visualizing embeddings with PCA** — squashing 100 dimensions down to 2 with your lesson-18 PCA so you can actually look at meaning.

## Before you start

Check you can do these (all from earlier lessons): train a small PyTorch model with a manual training loop (24), explain what `nn.Embedding` does mechanically (28), compute cosine similarity from a dot product (08), and run your own PCA (18).

Activate the venv and install what's new:

```bash
cd ~/ml/ml-learn
source .venv/bin/activate
pip install gensim tqdm        # torch, numpy, matplotlib already installed
```

If `pip install gensim` errors (it pins an older NumPy and can lag the newest Python releases, so on a very fresh setup it may try — and fail — to build from source), don't fight it: skip the install and use the no-gensim fallback in Project 3's hints instead. The fallback path is equivalent — everything the project does still works.

Download a corpus — three classic novels from Project Gutenberg (free, no account needed):

```bash
mkdir -p data/gutenberg work/30-embeddings-word2vec
cd data/gutenberg
wget -O pride.txt    https://www.gutenberg.org/cache/epub/1342/pg1342.txt
wget -O moby.txt     https://www.gutenberg.org/cache/epub/2701/pg2701.txt
wget -O sherlock.txt https://www.gutenberg.org/cache/epub/1661/pg1661.txt
cat pride.txt moby.txt sherlock.txt > corpus.txt
wc -w corpus.txt     # expect roughly 450,000-500,000 words
```

If any link 404s, go to gutenberg.org, search the title (*Pride and Prejudice*, *Moby-Dick*, *The Adventures of Sherlock Holmes*), and copy the "Plain Text UTF-8" link instead. All your code for this lesson goes in `work/30-embeddings-word2vec/`.

## Project 1 — word2vec from scratch

**Goal:** Implement skip-gram with negative sampling in PyTorch, train it on your novel corpus, and end up with a vector for every word such that nearby vectors mean similar things.

Before you write code, get the idea straight. A **one-hot vector** represents a word as a vector as long as the vocabulary — say 10,000 numbers — that is all zeros except a single 1. Every word is equally distant from every other word: "cat" is as far from "kitten" as from "carburetor". Useless for meaning. A **dense embedding** is instead a short vector (we'll use 100 numbers) of learned values, and we want *training* to place similar words near each other. The trick that makes this happen is the **distributional hypothesis**: words that appear in similar contexts ("the ___ sailed across the ocean") tend to mean similar things. So we invent a fake task — given a center word, predict the words around it (**skip-gram**) — and the embeddings that get good at this task are forced to encode meaning.

**Milestones:**

- [ ] **Tokenize and build a vocabulary.** Read `corpus.txt`, lowercase it, and split it into words with a simple regex like `re.findall(r"[a-z']+", text)`. Count word frequencies with a `Counter` (lesson 03). Keep only words appearing at least 5 times; drop the rest. Build `word_to_idx` and `idx_to_word` dicts. Checkpoint: vocabulary size lands somewhere between 4,000 and 10,000 words, and `word_to_idx["whale"]` exists.
- [ ] **Generate (center, context) training pairs.** Slide a window over the token stream: for each center word, pair it with every word up to `window=4` positions to its left and right (skipping dropped words). Store pairs as two parallel NumPy arrays of indices. Checkpoint: you have a few million pairs, and printing 10 random pairs as words shows genuine neighbors from the text.
- [ ] **Sanity-check the pipeline on a toy sentence.** Run your pair generator on `"the cat sat on the mat"` with `window=2` and verify by hand that the pairs are exactly right. This ten-minute check saves hours later.
- [ ] **Set up the model: two embedding tables.** Create `nn.Embedding(vocab_size, 100)` twice — one table for center words, one for context words. Why softmax won't do: predicting the context word properly means a probability over the *whole* vocabulary for every single pair, i.e. a 10,000-way (or for real corpora, 50,000-way) classification millions of times per epoch — brutally slow. **Negative sampling** replaces it with binary questions: for a true (center, context) pair, push the score `center · context` up; for `k=5` randomly sampled fake context words ("negatives"), push the score down. Each pair costs 6 tiny dot products instead of one giant softmax.
- [ ] **Implement the negative-sampling loss.** For a batch: look up center vectors and true-context vectors, sample 5 negative word indices per pair, and compute the loss `-log σ(pos_score) - Σ log σ(-neg_score)` using `F.logsigmoid` (σ is the sigmoid from lesson 13). Checkpoint: before any training, loss is close to `(1 + 5) × ln 2 ≈ 4.16` — six coin-flip questions the untrained model answers at chance.
- [ ] **Train.** Batches of ~512 pairs, `Adam` with lr around 3e-3, 2-3 passes over all pairs, printing loss every few hundred steps (wrap the loop in `tqdm` for a progress bar). Checkpoint: loss falls steadily and settles somewhere below 2.5. If it plateaus at 4.16, the model is learning nothing — see "If you get stuck".
- [ ] **Build `nearest(word, k=10)`.** Take the *center* embedding table, normalize every row to unit length, and compute cosine similarity between your query word's vector and all others — exactly your lesson-08 cosine similarity, now as one matrix-vector product. Print the top 10. Checkpoint: `nearest("sea")` returns mostly nautical words (ship, ocean, whale, deck...) and `nearest("mr")` returns titles and names. Random-looking neighbors mean training failed.

<details><summary>Hints</summary>

- Sample negatives with `torch.randint(0, vocab_size, (batch, 5))` to start. The original paper samples from word frequency raised to the power 0.75 (`torch.multinomial` on that distribution) — noticeably better neighbors, worth doing once the plain version trains.
- Scores are batched dot products: with center vectors `(B, 100)` and negative vectors `(B, 5, 100)`, `torch.bmm(neg, center.unsqueeze(2))` gives all negative scores at once. Print every shape the first time through.
- Don't feed pairs in corpus order — one novel's vocabulary would dominate each stretch of training. Shuffle indices each epoch with `torch.randperm(n_pairs)`.
- Occasionally a sampled negative is accidentally the true context word. Ignore it — it's rare enough not to matter.

</details>

**Definition of done:** Training loss drops from ~4.16 to below 2.5, and `nearest()` returns semantically related neighbors for at least 5 query words you try.

## Project 2 — Word arithmetic

**Goal:** Show that relationships between words are *directions* in your embedding space, then draw a 2D map of the space itself.

The famous claim: `vec("king") - vec("man") + vec("woman") ≈ vec("queen")`. Subtracting "man" from "king" isolates a *royalty-minus-maleness* direction; adding "woman" lands near "queen". If that works even roughly in vectors *you* trained on three novels, you have witnessed meaning become geometry.

**Milestones:**

- [ ] **Implement `analogy(a, b, c)`** returning the nearest words to `vec(b) - vec(a) + vec(c)` by cosine similarity — excluding `a`, `b`, and `c` themselves from the results (they are always suspiciously close, and every implementation must handle this).
- [ ] **Test classic analogies:** `analogy("man", "king", "woman")` hoping for queen; also try `("he", "his", "she")` → her, `("small", "smaller", "great")` → greater, and 5 more you invent from your corpus's world. Checkpoint: at least 2 of your analogies put the expected word in the top 5. Be honest in your notes about which failed — a corpus of three novels is thousands of times smaller than what the original word2vec used, so fuzzy results are the *expected* scientific outcome here (Project 3 shows what scale fixes).
- [ ] **Map the space with PCA.** Take the vectors of the 200 most frequent words (skip the top ~20 stopwords like "the" and "of" — they only add clutter), run your lesson-18 PCA to project 100 dimensions down to 2, and scatter-plot them with each word annotated via `plt.annotate`. Checkpoint: the plot renders with 200 readable labels; save it as `word_map.png`.
- [ ] **Find and circle the clusters.** Stare at the map. You should find at least 3 groups of related words sitting together — character names, pronouns, sea/ship words, time words... Circle them (matplotlib shapes, or honestly just a screenshot markup) and label each. Checkpoint: 3+ believable clusters, and you can explain in one sentence each *why* those words co-occur.

<details><summary>Hints</summary>

- Normalize all vectors to unit length once, up front. Then cosine similarity against the whole vocabulary is a single matrix-vector multiply and analogies get much cleaner.
- If `plt.annotate` labels overlap into mush, plot fewer words, shrink the font, or add a small random jitter to positions.
- PCA keeps only the 2 highest-variance directions out of 100, so the map is a heavy simplification — clusters that look merged may be far apart in the full space. Trust `nearest()` over the picture when they disagree.

</details>

**Definition of done:** At least 2 working analogies, and `word_map.png` with 3+ circled, labeled clusters checked into your work folder.

## Project 3 — Embeddings in the wild

**Goal:** Load industrial-strength pretrained vectors (GloVe, trained on 6 billion words), see what scale buys you, and use them to build a tiny semantic search engine — the core mechanism of RAG in lesson 35.

**GloVe** is a different algorithm from word2vec (it factorizes global co-occurrence counts rather than sliding a predictive window) but produces the same kind of object: one dense vector per word. Stanford gives the vectors away — see the [GloVe project page](https://nlp.stanford.edu/projects/glove/).

**Milestones:**

- [ ] **Load GloVe with gensim** (a Python library for word vectors): `import gensim.downloader; glove = gensim.downloader.load("glove-wiki-gigaword-100")`. First run downloads ~130 MB into `~/gensim-data/`. Checkpoint: `glove["king"].shape` is `(100,)`.
- [ ] **Rematch Project 1 and 2 against GloVe.** Compare `glove.most_similar("sea")` with your `nearest("sea")`, and run your failed analogies via `glove.most_similar(positive=["king", "woman"], negative=["man"])`. Checkpoint: king − man + woman = **queen** is the top result, and the neighbors are visibly crisper than yours. Write 3 sentences in a `notes.md`: what did 6 billion training words buy over 500 thousand?
- [ ] **Embed whole sentences.** Simplest possible recipe: a sentence's vector is the *average* of its words' GloVe vectors (skip words GloVe doesn't know). Write `embed(sentence) -> np.array` and collect ~30 sentences on mixed topics — cooking, sports, programming, weather — either hand-written or pulled from your corpus.
- [ ] **Build `search(query, k=3)`:** embed the query, cosine-rank all sentence vectors, return the top 3. Checkpoint: searching **"preparing food"** surfaces your cooking sentences above sports and code — *even when they don't share a single word with the query*. That last property is exactly what keyword search can't do and exactly why RAG (lesson 35) is built on embeddings; the embeddings there come from full LLM-scale transformer models, but the geometry-plus-cosine machinery is precisely what you just built.

<details><summary>Hints</summary>

- `glove.most_similar`, `glove.similarity("cat", "dog")`, and `"word" in glove.key_to_index` cover almost everything you need from gensim.
- **No gensim? No problem.** Download `glove.6B.zip` from the [GloVe project page](https://nlp.stanford.edu/projects/glove/), unzip it, and load `glove.6B.100d.txt` yourself — each line is a word followed by 100 numbers, so ~10 lines of plain Python (split each line, `np.array` the numbers) build a `dict` of word → vector. That dict supports everything this project uses: `similarity` is one cosine call with your lesson-08 code, and `most_similar` is that same cosine against every vector, sorted — exactly the `nearest()` you wrote in Project 1.
- Lowercase everything before lookups — this GloVe model has no capitalized entries, and silent lookup misses are the classic bug here.
- Averaging drowns rare, meaningful words under common ones. Skipping stopwords before averaging noticeably sharpens search results.

</details>

**Definition of done:** GloVe wins the comparison (documented in `notes.md`), and `search()` retrieves topically-right sentences for 3 different queries with no keyword overlap.

## Stretch goals

- **Subsample frequent words** as in the original word2vec paper: randomly drop very common words like "the" from the token stream before pair generation. Retrain and compare neighbor quality.
- **Scale up the corpus:** the classic benchmark is **text8**, 100 MB of cleaned Wikipedia text (search for "text8 dataset"). Train on a 10-30 MB slice and watch your analogies sharpen.
- **Implement CBOW**, skip-gram's mirror image: average the context vectors and predict the *center* word. Compare training speed and neighbor quality.
- **Bias probe:** compute `analogy("man", "doctor", "woman")` and similar in GloVe. Embeddings learn the statistics of their training text, stereotypes included — a real deployment issue worth seeing with your own eyes.

## If you get stuck

- **Loss stuck at ~4.16:** the model is at chance. Usual suspects, in order: pairs are wrong (rerun your toy-sentence check), you're mixing up the two embedding tables, or a sign error in `-log σ(-neg_score)`. Check that `loss.backward()` actually populates `model.center_emb.weight.grad` with nonzero values.
- **Loss near zero but garbage neighbors:** classic leak — the model found a shortcut. Check you aren't accidentally passing the true context word as its own negative, or comparing a word with itself in `nearest()`.
- **Shapes:** almost every bug in this lesson is a shape bug. Print `center.shape`, `context.shape`, `neg.shape`, `score.shape` for one batch before trusting the loop.
- **Slow training:** vectorize — no Python loop over pairs inside a batch. One epoch on ~3M pairs should take minutes on CPU, not hours.
- Standing advice: read error tracebacks bottom-up (the last line is the actual error), print shapes and a few real values at every step, ask an AI assistant for a *hint* rather than a solution, and type all code yourself — no pasting.

## Resources

- [The Illustrated Word2vec — Jay Alammar](https://jalammar.github.io/illustrated-word2vec/) — the best visual walkthrough of skip-gram and negative sampling; read it after Milestone 1 of Project 1, not before.
- [GloVe project page](https://nlp.stanford.edu/projects/glove/) — Stanford's pretrained vectors used in Project 3, with a short explanation of how GloVe differs from word2vec.
- [pytorch.org documentation](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html) — `nn.Embedding` reference, for exact semantics of the lookup tables you're training.
- *Distributed Representations of Words and Phrases and their Compositionality* (Mikolov et al., 2013) — the original negative-sampling paper; search the title. Short, and after this lesson you can genuinely read it.

## Skills unlocked

- [ ] I can explain why one-hot vectors can't capture similarity and dense embeddings can.
- [ ] I can state the distributional hypothesis and explain how the skip-gram task exploits it.
- [ ] I can explain negative sampling and why it beats a full softmax over the vocabulary.
- [ ] I can implement and train skip-gram word2vec in PyTorch from raw text to trained vectors.
- [ ] I can find nearest neighbors and compute analogies with cosine similarity in embedding space.
- [ ] I can visualize high-dimensional embeddings in 2D with PCA and interpret the clusters.
- [ ] I can load pretrained vectors with gensim and build embedding-based semantic search over sentences.

## Next up

Embeddings are the input layer of every transformer — next you build the rest of the machine, attention and all, in [31 · Attention and the Transformer: Let's Build GPT](31-attention-build-gpt.md).
