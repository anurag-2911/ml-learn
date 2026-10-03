# 17 · Text Classification: Naive Bayes Spam Filter (+ SVM)

**Phase 2 — Classical Machine Learning** · Estimated time: 1 week · Prerequisites: [10 · Probability and Statistics](../phase-1-math/10-probability-statistics.md), [14 · Model Evaluation](14-model-evaluation.md)

> This lesson is a first look at NLP (natural language processing), the art of getting machines to work with human language. Everything from spam filters to ChatGPT is part of this field. The lesson builds a real spam filter from scratch in about 60 lines of pure Python, using nothing but counting and Bayes' rule from lesson 10, and the filter reaches ~97% accuracy on real SMS messages. It then opens the filter up to make it *explain itself*, and finally introduces TF-IDF and Support Vector Machines through sklearn. The big idea of the lesson: before a machine can learn from text, the text must be turned into numbers.

## What this lesson builds

- **Project 1 — Spam filter from scratch**: `spam_filter.py`, a Naive Bayes classifier built from raw word counts on the UCI SMS Spam Collection, evaluated with the metrics library built from scratch in lesson 14.
- **Project 2 — Explain the filter**: `explain.py`, which prints the 15 most spam-indicative and 15 most ham-indicative words the model learned.
- **Project 3 — sklearn text pipeline**: `sklearn_spam.py`, comparing CountVectorizer/TfidfVectorizer + MultinomialNB and LinearSVC pipelines against the from-scratch model in one results table.

## Concepts covered

- **Bag of words**: representing a text as word counts, throwing away word order.
- **Bayes' rule as a classifier**: "given these words, what is the probability this is spam?"
- **The naive independence assumption**: pretending words appear independently of each other, and why this wrong assumption still works.
- **Laplace smoothing**: without it, a single unseen word would veto everything with a zero.
- **Log-probabilities**: multiplying 100 tiny numbers underflows to 0.0; adding their logs does not.
- **TF-IDF weighting**: down-weighting words that appear everywhere, so rare words carry more signal.
- **SVM margin intuition**: a classifier that draws the line with the widest possible "safety gap" between classes.

## Before starting

Check that these are in place:

- [Lesson 10](../phase-1-math/10-probability-statistics.md) is finished, and Bayes' rule can be stated and explained on a concrete example.
- The metrics library from [lesson 14](14-model-evaluation.md) is at hand (accuracy, precision, recall, confusion matrix).

Run the setup from the repo root. Phase 0 already installed numpy and pandas, so the install line needs only scikit-learn:

```bash
source .venv/bin/activate
pip install scikit-learn
mkdir -p work/17-naive-bayes-spam
cd work/17-naive-bayes-spam
```

Download the dataset of 5,574 real SMS messages labeled spam or ham ("ham" = not spam):

```bash
curl -L -o smsspam.zip "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
unzip smsspam.zip
head -5 SMSSpamCollection
```

If the direct link fails, open the dataset page in a browser and click Download: <https://archive.ics.uci.edu/dataset/228/sms+spam+collection>. Then move the zip into this folder. The file `SMSSpamCollection` is plain text: one message per line, `label<TAB>message`.

Also copy the metrics code here so that it can be imported:

```bash
cp ../14-model-evaluation/mymetrics.py .
```

(Adjust the path if the lesson 14 `mymetrics.py` lives elsewhere.)

## Project 1 — Spam filter from scratch

**Goal**: Implement Naive Bayes with Laplace smoothing and log-probabilities in pure Python/numpy, and reach ~97%+ accuracy on held-out SMS messages.

**Milestones**

- [ ] **Load the data.** Read `SMSSpamCollection` line by line and split each line on the tab character into `(label, message)`. Checkpoint: the loaded data holds 5,574 messages, and counting labels gives 4,827 ham and 747 spam (about 13% spam: an imbalanced dataset, of the kind lesson 14 warned about).

  ```python
  with open("SMSSpamCollection", encoding="utf-8") as f:
      pairs = [line.rstrip("\n").split("\t", 1) for line in f]
  ```

- [ ] **Split train/test.** Shuffle with a fixed seed (`random.seed(42)`) and hold out 20% for testing, exactly as in lesson 14. From now on, *only* the training set may influence the model. Checkpoint: ~4,459 training and ~1,115 test messages.
- [ ] **Tokenize.** Write `tokenize(text)` that lowercases the message and splits it into words (start simple: keep runs of letters/digits, drop punctuation). This is the **bag of words** step, which keeps the counts of words and throws away their order: "WIN a FREE prize!!" becomes `["win", "a", "free", "prize"]`. Checkpoint: `tokenize("Free entry!! Win FREE tickets")` returns `free` twice.
- [ ] **Count everything (this *is* training).** From the training set only, build: the two class priors `P(spam)` and `P(ham)` (fractions of messages in each class); a dict of word counts for spam and another for ham; the total word count in each class; and the vocabulary (every distinct training word). Naive Bayes has no gradient descent; training is literally counting. Checkpoint: the vocabulary size is somewhere in the 7,000–9,000 range (it depends on the tokenizer), and `spam_counts["free"]` is much larger than expected for 13% of the data.
- [ ] **Write down the classifier before coding it.** Bayes' rule from lesson 10, turned into a classifier: `P(spam | words) ∝ P(spam) · P(words | spam)`. The **naive** part: assume the words are independent given the class, so `P(words | spam)` is just the product of `P(word | spam)` for each word, where `P(word | spam) = count(word in spam) / total words in spam`. This assumption is false ("free" and "prize" travel together), but it turns an impossible estimation problem into simple counting, and it works remarkably well. Say the whole formula out loud in plain English before writing any code.
- [ ] **Hit the zero problem, then fix it with Laplace smoothing.** Try scoring a test message containing a word that appears in ham training data but never in spam. Its `P(word | spam)` is 0, and one zero in a product wipes out all other evidence: one unseen word would veto the whole message. **Laplace smoothing** fixes this by pretending that every vocabulary word was seen once more than it was: `P(word | class) = (count + 1) / (total_words_in_class + vocab_size)`. Checkpoint: with smoothing, no word in the vocabulary has probability 0 for either class.
- [ ] **Hit the underflow problem, then fix it with logs.** Run this in a Python shell: `0.001 ** 400` → `0.0`. A 100-word message means multiplying ~100 numbers each around 0.0001 (about 1e-400, below the smallest number a float can hold), and floats silently underflow to zero, so both classes score 0.0 and the comparison is garbage. Fix: compare **log-probabilities** instead. Since `log(a·b) = log(a) + log(b)`, replace the product with a sum: `log P(spam) + Σ log P(word|spam)`, using `math.log`. The scores become manageable negative numbers, and whichever is larger wins. Checkpoint: scores are roughly in the −20 to −300 range, never 0.0 or `-inf`.
- [ ] **Predict.** Write `predict(message)` that tokenizes, sums log-probabilities under each class, and returns the class with the higher score. Decide what to do with words not in the vocabulary at all (see Hints). Checkpoint: `predict("WINNER!! You have won a free prize, txt CLAIM to 81010")` → spam; `predict("ok, see you at 7 for dinner")` → ham.
- [ ] **Evaluate with the home-made metrics library.** On the test set, compute accuracy, the confusion matrix, and precision and recall *for the spam class*, using the lesson 14 code. The lesson 14 functions expect 1 = positive and 0 = negative, and given the strings `"spam"`/`"ham"` the confusion matrix, precision and recall silently come out as zeros. Convert both the true labels and the predictions first, with spam as the positive class: `np.array([1 if lab == "spam" else 0 for lab in labels])`. Apply the same conversion to every model's predictions in Project 3. Checkpoint: accuracy ≥ 0.97, spam precision ≥ 0.95.
- [ ] **Think about the cost of mistakes.** Look at the confusion matrix. A false positive here means a real message from a friend lands in the spam folder, which is much worse than letting one spam through. Which metric measures that? (Precision on spam.) Read the actual misclassified messages and note in a comment which errors are the most troubling.

<details><summary>Hints</summary>

- Store counts as `{"spam": Counter(), "ham": Counter()}`; `collections.Counter` (lesson 03) makes the counting loop three lines.
- For words at prediction time that are not in the vocabulary at all, the standard simple choice is to skip them entirely. Smoothing handles vocabulary words that are missing from *one* class.
- Precompute `log(P(word|class))` for every vocabulary word once after training, so prediction is just dictionary lookups and a sum.
- The smoothing denominator is `total_words_in_class + vocab_size`, where vocab size means the size of the *combined* vocabulary, not the per-class one. An off-by-something error here still "works" but shifts the accuracy down a point.

</details>

**Definition of done**: The from-scratch filter scores ≥ 0.97 accuracy and ≥ 0.95 spam precision on the test set, using smoothed log-probabilities and the home-made metrics code.

## Project 2 — Explain the filter

**Goal**: Open the black box: rank every word by how strongly it signals spam vs ham, and print the top 15 in each direction.

**Milestones**

- [ ] **Compute a likelihood ratio per word.** For each vocabulary word, compute `P(word | spam) / P(word | ham)` using the *smoothed* probabilities (unsmoothed ones divide by zero). A ratio far above 1 means "seeing this word is strong evidence of spam"; far below 1 means strong ham evidence. Working in logs, this is just `log P(word|spam) − log P(word|ham)`.
- [ ] **Print the top 15 spam-indicative words** (highest ratio) and **top 15 ham-indicative words** (lowest ratio), each with its ratio, nicely formatted. Checkpoint: the spam list is dominated by SMS-scam vocabulary such as `claim`, `prize`, `150p`, `guaranteed`, `awarded`, `ringtone` and `www`, plus prize amounts like `500` and `1000`. Common spam words such as `free`, `win` and `txt` rank much lower, because they also turn up in ordinary chat and the ratio rewards words that are rare in ham. The ham list looks like ordinary chat between friends.
- [ ] **Stress-test the explanation.** Pick one word from the spam list and verify its ratio by hand from the raw counts. Then write two new messages, one stuffed with spam-list words and one with ham-list words, and confirm that `predict` classifies both the way the word lists predict.
- [ ] **Reflect (one paragraph, as a comment or in personal notes).** The model "understands" nothing: it has never seen word order or meaning, only counts. Yet the word lists look meaningful. In [lesson 30](../phase-4-nlp-transformers/30-embeddings-word2vec.md), bag-of-words is replaced by **embeddings**, a representation where words get coordinates and "free" and "gratis" land near each other. Richer representations of text are coming; counting is just the first rung.

<details><summary>Hints</summary>

- Project 1 already computed every number needed here. This project is sorting a dictionary, not new modeling: `sorted(vocab, key=..., reverse=True)[:15]`.
- If the top spam words look like garbage tokens (single letters, word fragments), the tokenizer is the cause. Numbers such as `500`, `1000` and `18` are real signals here: prize amounts and age limits in scam texts. Filtering out words seen fewer than ~5 times in training cleans the list up nicely.

</details>

**Definition of done**: Running `python explain.py` prints two clean 15-word tables, and the spam table passes the "claim/prize/150p" sanity check.

## Project 3 — sklearn text pipeline

**Goal**: Reproduce and beat the from-scratch filter with sklearn's text tools, meet TF-IDF and a linear SVM, and put every model in one comparison table.

**Milestones**

- [ ] **Rebuild the model in 5 lines with sklearn.** A `Pipeline` (lesson 19 goes deep on these; this is a preview) chains `CountVectorizer` (sklearn's bag-of-words: text in, count matrix out) and `MultinomialNB` (sklearn's Naive Bayes; `alpha=1.0` is exactly the Laplace smoothing from Project 1). Use the *same* train/test split as Project 1 so that the numbers are comparable. Checkpoint: accuracy within about a point of the from-scratch model. This check verifies 60 lines of hand-written Python against an industrial library.

  ```python
  from sklearn.pipeline import Pipeline
  from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
  from sklearn.naive_bayes import MultinomialNB
  from sklearn.svm import LinearSVC

  model = Pipeline([("vec", CountVectorizer()), ("clf", MultinomialNB())])
  model.fit(train_texts, train_labels)   # raw strings in — the pipeline does the rest
  ```

- [ ] **Swap in TF-IDF.** Replace `CountVectorizer` with `TfidfVectorizer`. **TF-IDF** (term frequency – inverse document frequency) re-weights the counts: a word scores high in a message if it is frequent *in that message* but rare *across all messages*. Words like "the" and "to" appear everywhere, so they carry almost no information about spam-ness. TF-IDF shrinks them toward zero automatically, with no need to filter them out by hand. Checkpoint: accuracy lands around 0.95–0.97, below raw counts, and spam recall drops to roughly 0.65–0.80 while spam precision stays near 1.0. TF-IDF values are small fractions, so the default `alpha=1.0` smoothing swamps them; `MultinomialNB(alpha=0.1)` recovers most of the gap. On short SMS messages TF-IDF may not beat raw counts; that is a legitimate finding, not a bug.
- [ ] **Try a linear SVM.** Pipeline: `TfidfVectorizer` + `LinearSVC`. A **Support Vector Machine** takes a completely different route from Naive Bayes: no probabilities, pure geometry. Each message is now a point in ~8,000-dimensional space (one axis per word). Many boundaries could separate spam from ham points; the SVM picks the one with the **maximum margin**, the widest empty buffer zone between the boundary and the nearest points on each side. Those nearest points are the "support vectors": the borderline messages that alone define the boundary. A wider buffer means safer generalization. That is the whole intuition, with no difficult math required. High-dimensional sparse text is the classic terrain where linear SVMs shine. Checkpoint: this is likely the best model in the lesson, around 0.98–0.99 accuracy.
- [ ] **Build the results table.** Give it one row per approach: the from-scratch NB, CountVectorizer+NB, TfidfVectorizer+NB and TfidfVectorizer+LinearSVC. The columns are accuracy, spam precision and spam recall, all computed with the lesson 14 metrics on the same test set. Print it aligned. Checkpoint: every model scores about 0.95 accuracy or better (TF-IDF + NB with the default `alpha` is the weakest), and one sentence explains which model to actually deploy and why (remember: false positives are costly, so precision may outrank raw accuracy).
- [ ] **Attack the filter.** Hand-craft the best spammy message possible and run it through all models. Then try to *evade* them: reword it ("fr3e", "w1n") until some model calls it ham. Reworded messages like these are adversarial inputs, and they explain why real spam looks the way it does.

<details><summary>Hints</summary>

- In `model.predict(["your message here"])`, the input must be a list of strings, not a bare string. A bare string raises `ValueError: Iterable over raw text documents expected, string object received.`
- To keep the comparison fair, split the raw *texts* once, then feed identical train/test lists to every pipeline (and to the Project 1 code).
- If the pipelines get confusing, the sklearn text-classification example linked in Resources walks through this vectorizer + classifier pattern.

</details>

**Definition of done**: One script prints a 4-row comparison table of accuracy/precision/recall, and two ideas can be explained in plain words: what TF-IDF changes, and what an SVM margin is.

## Stretch goals

- **Bigrams.** Pass `ngram_range=(1, 2)` to the vectorizer so that "free entry" counts as a feature alongside "free" and "entry", a small step back toward word order. Does it help?
- **Tune the smoothing.** Loop `alpha` over `0.01, 0.1, 0.5, 1, 2, 5, 10` in the from-scratch model and plot accuracy vs alpha with a log-scale x-axis (`plt.xscale("log")`, as in lesson 10). This is hyperparameter tuning by hand.
- **Ship it as a CLI.** `python spam_check.py "some message"` prints SPAM or HAM plus the top 3 words that drove the decision: Project 1 and Project 2 glued into a tool.
- **A harder dataset.** Rerun the whole comparison on a different text classification problem (e.g. sklearn's built-in `fetch_20newsgroups`, no download account needed) and see which conclusions survive.

## Getting unstuck

- **Accuracy stuck near 0.87?** That is exactly the ham fraction, so the model is predicting ham for everything. Usual suspects: mixed-up spam/ham counts, or comparing a log-score against a raw probability.
- **`ValueError: math domain error` (or `-inf` scores when using `np.log`)?** The code took `log(0)` somewhere: a word slipped past smoothing (for example, one code path uses the raw count, or the precomputed table covers only the words seen in that class). Print `P(word|class)` for the offending word.
- **From-scratch and sklearn NB disagree wildly (>3 points)?** Different tokenizers. Print `CountVectorizer().build_analyzer()("Free entry!! Win")` and compare it with `tokenize` on the same string.
- Standing advice: read error tracebacks bottom-up (the last line names the real error), print intermediate values (counts, vocab size, one message's per-word log-probs) instead of staring at code, ask an AI assistant for a hint ("what concept am I missing?"), never for the solution, and type every line by hand. Copy-pasted code teaches nothing.

## Resources

- **StatQuest: Naive Bayes** — the clearest visual walkthrough of exactly the classifier built in this lesson; search YouTube for "StatQuest Naive Bayes, Clearly Explained". Watch it *after* a first attempt at Project 1.
- **StatQuest: Support Vector Machines** — margins and support vectors with pictures; search YouTube for "StatQuest Support Vector Machines". Pairs with Project 3.
- **sklearn — Text feature extraction**: <https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction> — the user-guide section on `CountVectorizer` and `TfidfVectorizer`.
- **sklearn — Classification of text documents using sparse features**: <https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html> — the official text-classification example of Project 3's vectorizer + classifier pattern.
- **UCI SMS Spam Collection**: <https://archive.ics.uci.edu/dataset/228/sms+spam+collection> — the dataset's home page, with the original paper describing how it was collected.

## Skills unlocked

- [ ] I can explain bag-of-words and turn raw text into numbers a model can use.
- [ ] I can write Bayes' rule as a classifier and state the naive independence assumption and why it is wrong but useful.
- [ ] I can explain what breaks without Laplace smoothing and without log-probabilities, and fix both.
- [ ] I built a ~97% spam filter in pure Python and verified it against sklearn.
- [ ] I can make a text classifier explain itself via per-word likelihood ratios.
- [ ] I can explain TF-IDF in one sentence and say when it helps.
- [ ] I can describe an SVM's maximum-margin idea without equations.
- [ ] I choose evaluation metrics based on the real-world cost of each error type.

## Next up

So far the models have learned from labels of every kind; the next lesson takes the labels away entirely and lets the machine find structure on its own: [18 · Unsupervised Learning: k-Means and PCA from Scratch](18-unsupervised-kmeans-pca.md).
