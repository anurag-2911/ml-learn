# 17 · Text Classification: Naive Bayes Spam Filter (+ SVM)

**Phase 2 — Classical Machine Learning** · Estimated time: 1 week · Prerequisites: [10 · Probability and Statistics](../phase-1-math/10-probability-statistics.md), [14 · Model Evaluation](14-model-evaluation.md)

> This is your first contact with NLP — natural language processing, the art of getting machines to work with human language. That is worth celebrating: everything from spam filters to ChatGPT lives on this branch of the tree. This week you build a real spam filter from scratch in about 60 lines of pure Python, using nothing but counting and the Bayes' rule you met in lesson 10 — and it will hit ~97% accuracy on real SMS messages. Then you will open the filter up and make it *explain itself*, and finally meet TF-IDF and Support Vector Machines through sklearn. The big idea of the week: before a machine can learn from text, you must turn text into numbers.

## What you will build

- **Project 1 — Spam filter from scratch**: `spam_filter.py`, a Naive Bayes classifier built from raw word counts on the UCI SMS Spam Collection, evaluated with your own metrics library from lesson 14.
- **Project 2 — Explain the filter**: `explain.py`, which prints the 15 most spam-indicative and 15 most ham-indicative words your model learned.
- **Project 3 — sklearn text pipeline**: `sklearn_spam.py`, comparing CountVectorizer/TfidfVectorizer + MultinomialNB and LinearSVC pipelines against your from-scratch model in one results table.

## Concepts you will learn by doing

- **Bag of words** — representing a text as word counts, throwing away word order.
- **Bayes' rule as a classifier** — "given these words, what is the probability this is spam?"
- **The naive independence assumption** — pretending words appear independently of each other, and why this wrong assumption still works.
- **Laplace smoothing** — why a single unseen word would otherwise veto everything with a zero.
- **Log-probabilities** — multiplying 100 tiny numbers underflows to 0.0; adding their logs does not.
- **TF-IDF weighting** — down-weighting words that appear everywhere, so rare words carry more signal.
- **SVM margin intuition** — a classifier that draws the line with the widest possible "safety gap" between classes.

## Before you start

Check you have what you need:

- You finished [lesson 10](../phase-1-math/10-probability-statistics.md) — you can state Bayes' rule and explain it on a concrete example.
- You have your metrics library from [lesson 14](14-model-evaluation.md) (accuracy, precision, recall, confusion matrix).

Set up (from the repo root):

```bash
source .venv/bin/activate
pip install scikit-learn        # numpy/pandas you already have from Phase 0
mkdir -p work/17-naive-bayes-spam
cd work/17-naive-bayes-spam
```

Download the dataset — 5,572 real SMS messages labeled spam or ham ("ham" = not spam):

```bash
curl -L -o smsspam.zip "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
unzip smsspam.zip
head -5 SMSSpamCollection
```

If the direct link fails, open the dataset page in your browser and click Download: <https://archive.ics.uci.edu/dataset/228/sms+spam+collection> — then move the zip into this folder. The file `SMSSpamCollection` is plain text: one message per line, `label<TAB>message`.

Also copy your metrics code here so you can import it:

```bash
cp ../14-model-evaluation/mymetrics.py .
```

(Adjust the path if your lesson 14 `mymetrics.py` lives elsewhere.)

## Project 1 — Spam filter from scratch

**Goal**: Implement Naive Bayes with Laplace smoothing and log-probabilities in pure Python/numpy, and reach ~97%+ accuracy on held-out SMS messages.

**Milestones**

- [ ] **Load the data.** Read `SMSSpamCollection` line by line, split each line on the tab character into `(label, message)`. Checkpoint: you have 5,572 messages, and counting labels gives 4,825 ham and 747 spam (about 13% spam — an imbalanced dataset, like lesson 14 warned you about).

  ```python
  with open("SMSSpamCollection", encoding="utf-8") as f:
      pairs = [line.rstrip("\n").split("\t", 1) for line in f]
  ```

- [ ] **Split train/test.** Shuffle with a fixed seed (`random.seed(42)`) and hold out 20% for testing, exactly as you did in lesson 14. From now on, *only* the training set may influence the model. Checkpoint: ~4,457 training and ~1,115 test messages.
- [ ] **Tokenize.** Write `tokenize(text)` that lowercases the message and splits it into words (start simple: keep runs of letters/digits, drop punctuation). This is the **bag of words** step: "WIN a FREE prize!!" becomes `["win", "a", "free", "prize"]` — the counts of words, with order thrown away. Checkpoint: `tokenize("Free entry!! Win FREE tickets")` returns `free` twice.
- [ ] **Count everything (this IS training).** From the training set only, build: the two class priors `P(spam)` and `P(ham)` (fractions of messages in each class); a dict of word counts for spam and another for ham; the total word count in each class; and the vocabulary (every distinct training word). Naive Bayes has no gradient descent — training is literally counting. Checkpoint: vocabulary size is somewhere in the 7,000–9,000 range (depends on your tokenizer), and `spam_counts["free"]` is much larger than you would expect for 13% of the data.
- [ ] **Write down the classifier before coding it.** Bayes' rule from lesson 10, turned into a classifier: `P(spam | words) ∝ P(spam) · P(words | spam)`. The **naive** part: assume the words are independent given the class, so `P(words | spam)` is just the product of `P(word | spam)` for each word, where `P(word | spam) = count(word in spam) / total words in spam`. This assumption is false ("free" and "prize" travel together) — but it turns an impossible estimation problem into simple counting, and it works remarkably well. Say the whole formula out loud in plain English before you write any code.
- [ ] **Hit the zero problem, then fix it with Laplace smoothing.** Try scoring a test message containing a word that appears in ham training data but never in spam. Its `P(word | spam)` is 0, and one zero in a product wipes out all other evidence — one unseen word would veto the whole message. **Laplace smoothing** fixes it: pretend every vocabulary word was seen once more than it was: `P(word | class) = (count + 1) / (total_words_in_class + vocab_size)`. Checkpoint: with smoothing, no word in the vocabulary has probability 0 for either class.
- [ ] **Hit the underflow problem, then fix it with logs.** Run this in a Python shell: `0.001 ** 400` → `0.0`. A 40-word message means multiplying ~40 numbers each around 0.0001, and floats silently underflow to zero — both classes score 0.0 and the comparison is garbage. Fix: compare **log-probabilities** instead. Since `log(a·b) = log(a) + log(b)`, replace the product with a sum: `log P(spam) + Σ log P(word|spam)`, using `math.log`. The scores become manageable negative numbers, and whichever is larger wins. Checkpoint: scores are roughly in the −20 to −300 range, never 0.0 or `-inf`.
- [ ] **Predict.** Write `predict(message)` that tokenizes, sums log-probabilities under each class, and returns the class with the higher score. Decide what to do with words not in the vocabulary at all (see Hints). Checkpoint: `predict("WINNER!! You have won a free prize, txt CLAIM to 81010")` → spam; `predict("ok, see you at 7 for dinner")` → ham.
- [ ] **Evaluate with YOUR metrics library.** On the test set, compute accuracy, the confusion matrix, and precision and recall *for the spam class*, using your lesson 14 code. Checkpoint: accuracy ≥ 0.97, spam precision ≥ 0.95.
- [ ] **Think about the cost of mistakes.** Look at your confusion matrix. A false positive here means a real message from a friend lands in the spam folder — much worse than letting one spam through. Which metric measures that? (Precision on spam.) Read the actual misclassified messages and note in a comment which errors bother you.

<details><summary>Hints</summary>

- Store counts as `{"spam": Counter(), "ham": Counter()}` — `collections.Counter` (lesson 03) makes the counting loop three lines.
- For words at prediction time that are not in the vocabulary at all, the standard simple choice is to skip them entirely. Smoothing handles vocabulary words that are missing from *one* class.
- Precompute `log(P(word|class))` for every vocabulary word once after training, so prediction is just dictionary lookups and a sum.
- The smoothing denominator is `total_words_in_class + vocab_size` — vocab size of the *combined* vocabulary, not the per-class one. Off-by-something here still "works" but shifts your accuracy down a point.

</details>

**Definition of done**: Your from-scratch filter scores ≥ 0.97 accuracy and ≥ 0.95 spam precision on the test set, using smoothed log-probabilities and your own metrics code.

## Project 2 — Explain the filter

**Goal**: Open the black box: rank every word by how strongly it signals spam vs ham, and print the top 15 in each direction.

**Milestones**

- [ ] **Compute a likelihood ratio per word.** For each vocabulary word, compute `P(word | spam) / P(word | ham)` using your *smoothed* probabilities (unsmoothed ones divide by zero). A ratio far above 1 means "seeing this word is strong evidence of spam"; far below 1 means strong ham evidence. Working in logs, this is just `log P(word|spam) − log P(word|ham)`.
- [ ] **Print the top 15 spam-indicative words** (highest ratio) and **top 15 ham-indicative words** (lowest ratio), each with its ratio, nicely formatted. Checkpoint: the spam list is dominated by words like `free`, `win`/`winner`, `txt`, `claim`, `prize`, `mobile`, `150p` — SMS-scam vocabulary. The ham list looks like ordinary chat between friends.
- [ ] **Stress-test your understanding.** Pick one word from the spam list and verify its ratio by hand from the raw counts. Then write two messages of your own — one stuffed with spam-list words, one with ham-list words — and confirm `predict` classifies both the way the word lists predict.
- [ ] **Reflect (one paragraph, as a comment or in your notes).** Your model "understands" nothing — it has never seen word order or meaning, only counts. Yet the word lists look meaningful. In [lesson 30](../phase-4-nlp-transformers/30-embeddings-word2vec.md) you will replace bag-of-words with **embeddings** — a representation where words get coordinates and "free" and "gratis" land near each other. Richer representations of text are coming; counting is just the first rung.

<details><summary>Hints</summary>

- You already have every number you need from Project 1 — this project is sorting a dictionary, not new modeling. `sorted(vocab, key=..., reverse=True)[:15]`.
- If your top spam words look like garbage tokens (single letters, numbers), that is your tokenizer talking. Filtering out words seen fewer than ~5 times in training cleans the list up nicely.

</details>

**Definition of done**: Running `python explain.py` prints two clean 15-word tables, and the spam table passes the "free/win/txt" sanity check.

## Project 3 — sklearn text pipeline

**Goal**: Reproduce and beat your filter with sklearn's text tools, meet TF-IDF and a linear SVM, and put every model in one comparison table.

**Milestones**

- [ ] **Rebuild your model in 5 lines with sklearn.** A `Pipeline` (lesson 19 will go deep on these — here you get a preview) chaining `CountVectorizer` (sklearn's bag-of-words: text in, count matrix out) and `MultinomialNB` (sklearn's Naive Bayes, `alpha=1.0` is exactly your Laplace smoothing). Use the *same* train/test split as Project 1 so numbers are comparable. Checkpoint: accuracy within about a point of your from-scratch model — you just verified 60 lines of your own Python against an industrial library.

  ```python
  from sklearn.pipeline import Pipeline
  from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
  from sklearn.naive_bayes import MultinomialNB
  from sklearn.svm import LinearSVC

  model = Pipeline([("vec", CountVectorizer()), ("clf", MultinomialNB())])
  model.fit(train_texts, train_labels)   # raw strings in — the pipeline does the rest
  ```

- [ ] **Swap in TF-IDF.** Replace `CountVectorizer` with `TfidfVectorizer`. **TF-IDF** (term frequency – inverse document frequency) re-weights the counts: a word scores high in a message if it is frequent *in that message* but rare *across all messages*. Words like "the" and "to" appear everywhere, so they carry almost no information about spam-ness — TF-IDF shrinks them toward zero automatically, instead of you hand-filtering them. Checkpoint: accuracy is in the same ~0.96–0.98 neighborhood (on short SMS messages TF-IDF may not beat raw counts — that is a legitimate finding, not a bug).
- [ ] **Try a linear SVM.** Pipeline: `TfidfVectorizer` + `LinearSVC`. A **Support Vector Machine** takes a completely different route from Naive Bayes: no probabilities, pure geometry. Each message is now a point in ~8,000-dimensional space (one axis per word). Many boundaries could separate spam from ham points; the SVM picks the one with the **maximum margin** — the widest empty buffer zone between the boundary and the nearest points on each side. Those nearest points are the "support vectors": the borderline messages that alone define the boundary. Wider buffer, safer generalization — that is the whole intuition, no scary math required. High-dimensional sparse text is the classic terrain where linear SVMs shine. Checkpoint: this is likely your best model, around 0.98–0.99 accuracy.
- [ ] **Build the results table.** One row per approach — your from-scratch NB, CountVectorizer+NB, TfidfVectorizer+NB, TfidfVectorizer+LinearSVC — with columns for accuracy, spam precision, spam recall (all computed with your lesson 14 metrics on the same test set). Print it aligned. Checkpoint: every model ≥ 0.96 accuracy, and you can say in one sentence which you would actually deploy and why (remember: false positives are costly, so precision may outrank raw accuracy).
- [ ] **Attack your own filter.** Write your best hand-crafted spammy message and run it through all models. Then try to *evade* them: reword it ("fr3e", "w1n") until some model calls it ham. You have just discovered adversarial inputs — and why real spam looks the way it does.

<details><summary>Hints</summary>

- `model.predict(["your message here"])` — the input must be a list of strings, not a bare string. If you get a cryptic error about the input shape, that is why.
- To keep the comparison fair, split the raw *texts* once, then feed identical train/test lists to every pipeline (and to your Project 1 code).
- The sklearn text tutorial linked in Resources walks through exactly this vectorizer + classifier pattern if you get lost.

</details>

**Definition of done**: One script prints a 4-row comparison table of accuracy/precision/recall, and you can explain in plain words what TF-IDF changes and what an SVM margin is.

## Stretch goals

- **Bigrams.** Pass `ngram_range=(1, 2)` to your vectorizer so "free entry" counts as a feature alongside "free" and "entry" — a small step back toward word order. Does it help?
- **Tune the smoothing.** Loop `alpha` over `0.01, 0.1, 0.5, 1, 2, 5, 10` in your from-scratch model, plot accuracy vs alpha (log-scale x-axis, lesson 07 skills). You are doing hyperparameter tuning by hand.
- **Ship it as a CLI.** `python spam_check.py "some message"` prints SPAM or HAM plus the top 3 words that drove the decision — Project 1 and Project 2 glued into a tool.
- **A harder dataset.** Rerun your whole comparison on a different text classification problem (e.g. sklearn's built-in `fetch_20newsgroups`, no download account needed) and see which conclusions survive.

## If you get stuck

- **Accuracy stuck near 0.87?** That is exactly the ham fraction — your model is predicting ham for everything. Usual suspects: mixed-up spam/ham counts, or comparing a log-score against a raw probability.
- **Scores are `-inf` or `nan`?** You took `log(0)` somewhere — a word slipped past smoothing, or you smoothed counts but forgot the adjusted denominator. Print `P(word|class)` for the offending word.
- **From-scratch and sklearn NB disagree wildly (>3 points)?** Different tokenizers. Print `CountVectorizer().build_analyzer()("Free entry!! Win")` and compare with your `tokenize` on the same string.
- Standing advice: read error tracebacks bottom-up (the last line names the real error), print intermediate values (counts, vocab size, one message's per-word log-probs) instead of staring at code, ask an AI assistant for a HINT — "what concept am I missing?" — never for the solution, and type every line yourself. Copy-pasted code teaches nothing.

## Resources

- **StatQuest: Naive Bayes** — the clearest visual walkthrough of exactly the classifier you are building; search YouTube for "StatQuest Naive Bayes, Clearly Explained". Watch it *after* your first attempt at Project 1.
- **StatQuest: Support Vector Machines** — margins and support vectors with pictures; search YouTube for "StatQuest Support Vector Machines". Pairs with Project 3.
- **sklearn — Working with text data**: <https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html> — the official tutorial behind Project 3's vectorizer + classifier pattern.
- **UCI SMS Spam Collection**: <https://archive.ics.uci.edu/dataset/228/sms+spam+collection> — your dataset's home page, with the original paper describing how it was collected.

## Skills unlocked

- [ ] I can explain bag-of-words and turn raw text into numbers a model can use.
- [ ] I can write Bayes' rule as a classifier and state the naive independence assumption — and why it is wrong but useful.
- [ ] I can explain what breaks without Laplace smoothing and without log-probabilities, and fix both.
- [ ] I built a ~97% spam filter in pure Python and verified it against sklearn.
- [ ] I can make a text classifier explain itself via per-word likelihood ratios.
- [ ] I can explain TF-IDF in one sentence and say when it helps.
- [ ] I can describe an SVM's maximum-margin idea without equations.
- [ ] I choose evaluation metrics based on the real-world cost of each error type.

## Next up

You have now trained models on labels of every kind — next you take the labels away entirely and let the machine find structure on its own: [18 · Unsupervised Learning: k-Means and PCA from Scratch](18-unsupervised-kmeans-pca.md).
