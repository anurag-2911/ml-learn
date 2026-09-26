# 35 · RAG: Build "Chat With My Documents"

**Phase 5 — LLMs** · Estimated time: 1-2 weeks · Prerequisites: [34 · Using LLMs: APIs, Prompting and Structured Output](34-llm-apis-prompting.md), [30 · Embeddings: Meaning as Geometry](../phase-4-nlp-transformers/30-embeddings-word2vec.md), [08 · Linear Algebra by Writing Your Own](../phase-1-math/08-linear-algebra-by-code.md)

> An LLM knows nothing about YOUR notes, YOUR company wiki, or anything written after its training ended — and when you ask anyway, it confidently makes things up. RAG (retrieval-augmented generation) fixes this: search your documents for the relevant passages first, then hand them to the LLM and say "answer from these, and cite your sources." It is the single most-built LLM application in industry right now, and its core is embarrassingly simple — you will build the whole engine yourself in numpy before touching any framework. By the end of this lesson you can point a chatbot at any folder of files and interrogate it.

## What you will build

- **Project 1** — A retrieval engine in pure numpy: a script that chunks your documents, embeds them locally, and finds the most relevant chunk for any question — with a 10-question eval proving it works.
- **Project 2** — A full "chat with my documents" web app (Streamlit or Gradio): upload files, ask questions, get answers with `[source: file.md]` citations, and a bot that says "I don't know" instead of guessing.
- **Project 3** — A RAG lab report: measured comparisons of chunk size and top-k on your own eval, plus your numpy store swapped for a real vector database (Chroma or FAISS).

## Concepts you will learn by doing

- **Why LLMs need retrieval** — knowledge cutoff, private data, and hallucination (confident made-up answers).
- **Text embeddings** — turning a sentence into a vector of numbers where similar meanings land close together (the industrial-strength version of your lesson 30 word2vec).
- **Chunking** — splitting documents into passages small enough to search and stuff into a prompt, and why chunks overlap.
- **Cosine similarity search** — ranking chunks by the angle between vectors (the same cosine you coded in lesson 08).
- **The RAG loop** — embed the query → retrieve top-k chunks → stuff them into the prompt → answer with citations.
- **Vector stores** — what FAISS and Chroma actually do, and why a numpy array is the same thing at small scale.
- **Evaluating RAG** — retrieval hit-rate (did the right chunk come back?) and answer faithfulness (did the answer stick to the sources?).

## Before you start

1. You finished lesson 34 and have a working LLM API setup (a provider account, an API key in an environment variable, and a `chat(prompt)` helper you wrote). Any provider works; check your provider's docs for current model names — do not hardcode assumptions from a tutorial.
2. Activate your venv and install this lesson's tools:

```bash
cd ~/ml/ml-learn
source .venv/bin/activate
pip install sentence-transformers streamlit gradio chromadb faiss-cpu
```

   `sentence-transformers` downloads a small embedding model (~90 MB) on first use and runs it locally on CPU — free, no account, no API key.

3. Create your work folder and a data folder:

```bash
mkdir -p work/35-rag-chat-with-your-docs/data
```

4. Get a document set. The perfect one is sitting right here — this curriculum. Copy 15-20 lesson files in:

```bash
cp phase-0-foundations/*.md phase-1-math/*.md phase-2-classical-ml/*.md work/35-rag-chat-with-your-docs/data/
ls work/35-rag-chat-with-your-docs/data | wc -l   # 20 files — right in the 15-20 range
```

   Your own notes or any folder of `.md`/`.txt` files works too — RAG on documents you know well makes wrong answers easy to spot.

## Project 1 — RAG retrieval from scratch in numpy

**Goal:** Build the retrieval half of RAG with nothing but sentence-transformers and numpy, and prove with a 10-question eval that it finds the right chunk. No frameworks — you need to see that the core of every vector database is ~5 lines of numpy.

**Milestones**

- [ ] Smoke-test embeddings in a throwaway script: embed the sentences "How do I train a neural network?", "What is backpropagation?", and "My cat sleeps all day" with `SentenceTransformer("all-MiniLM-L6-v2")`, then compute all pairwise cosine similarities with your own numpy function (dot product divided by the product of norms — lesson 08). Checkpoint: the two ML sentences score noticeably higher with each other (roughly 0.5+) than either does with the cat sentence (roughly below 0.2). Meaning is now geometry — this is lesson 30's idea, but for whole sentences.
- [ ] Write `chunk_text(text, chunk_size=150, overlap=30)` that splits a document into chunks of ~150 words, each starting 30 words before the previous one ended. Overlap exists so a sentence sitting on a chunk boundary appears whole in at least one chunk. Print the first 3 chunks of one lesson file. Checkpoint: chunk 2 starts with the last ~30 words of chunk 1.
- [ ] Write `build_index(folder)`: loop over every file in `data/`, chunk it, and keep two parallel structures — a list of dicts (`{"text": ..., "source": filename, "chunk_id": i}`) and one numpy matrix of all chunk embeddings (embed a whole list of texts in one `model.encode(list_of_texts)` call; it is much faster). Save with `np.save` and `json.dump`. Checkpoint: for ~18 lesson files you get roughly 200-600 chunks, and the matrix shape is `(num_chunks, 384)` — 384 is this model's embedding size.
- [ ] Write `search(query, k=3)`: embed the query, compute cosine similarity against every row of the matrix in one vectorized expression (no Python loop — lesson 05), and return the top-k chunks with scores. `np.argsort` gives ascending order; you want the other end. Checkpoint: `search("how does gradient descent update weights?")` returns chunks from lesson 09 or 12 with scores around 0.5-0.8.
- [ ] Build your eval set: write `eval.json` with 10 questions you can answer from your documents, each with the filename that contains the answer, e.g. `{"question": "What does the chain rule let backprop do?", "expected_source": "22-micrograd-backpropagation.md"}`. Make them varied: some easy keyword matches, some paraphrased so no shared words appear.
- [ ] Write `evaluate.py`: for each question, check whether any top-3 result comes from the expected file. This metric is called **retrieval hit-rate**: the fraction of questions where the right source appears in the top-k. Print a per-question pass/fail table and the total. Checkpoint: hit-rate at least 8/10. If you are below, read the misses — usually the question is answerable from a different file than you guessed (fix the eval) or your chunks are too small to carry meaning (fix chunking).

<details><summary>Hints</summary>

- Cosine against the whole matrix at once: normalize every embedding to unit length when you build the index (divide each row by its norm), and normalize the query too — then cosine similarity is just `matrix @ query`, one matrix-vector product for all chunks.
- Top-k without sorting everything backwards: `np.argsort(scores)[::-1][:k]`, or look up `np.argpartition` for the efficient version.
- Chunking by words is fine here: `words = text.split()`, then slice `words[start:start+chunk_size]` with `start` advancing by `chunk_size - overlap`. (Real systems count tokens; a token is roughly 3/4 of a word — good enough to reason with.)
- If every score looks weirdly similar (all ~0.9), you probably embedded the chunks with some identical boilerplate attached, or compared un-normalized vectors with dot product instead of cosine.

</details>

**Definition of done:** `python evaluate.py` prints a 10-row table and a hit-rate of 8/10 or better, and you can explain why the numpy matrix + cosine IS a vector store.

## Project 2 — The full RAG app: retrieval + LLM + citations + UI

**Goal:** Complete the loop: retrieved chunks go into an LLM prompt, the answer comes back grounded in your documents with citations, and a web UI makes it usable by anyone. This is the pattern behind every "chat with your PDF" product.

**Milestones**

- [ ] Write `answer(question)` in a plain script first (no UI): call your Project 1 `search()` with k=4, then build a prompt with three parts — instructions, the retrieved chunks each labeled with its source (`[source: 09-calculus-and-gradient-descent.md]\n<chunk text>`), and the question. Send it through your lesson 34 `chat()` helper. This step — pasting retrieved text into the prompt — is all "augmented generation" means.
- [ ] Make the instructions strict, in your own words: answer ONLY from the provided context; cite the source label after each claim; if the context does not contain the answer, reply "I don't know based on the provided documents." This last line is your main defense against hallucination — without it the model fills gaps from its training data and you cannot tell which is which.
- [ ] Test grounding both ways. Ask 3 questions your documents answer and 3 they cannot ("Who won the 2022 World Cup?", "What is the capital of Brazil?", "What does lesson 99 cover?"). Checkpoint: in-scope answers include at least one `[source: ...]` citation each; all 3 out-of-scope questions get the "I don't know" response. If the model answers Brazil anyway, tighten the instructions and test again — prompt, test, tighten is the real workflow.
- [ ] Wrap it in a UI with Streamlit (or Gradio if you prefer — both turn a Python function into a web page in ~20 lines). Minimum: a file uploader accepting `.md`/`.txt`, an "index my files" button that chunks + embeds the uploads, a question box, and an answer area showing the cited sources under the answer. Run it:

```bash
streamlit run app.py
```

   Checkpoint: the app opens in your browser at `localhost:8501`, you upload 3 files, ask a question, and see an answer with sources.
- [ ] Embedding on every rerun is slow, and Streamlit reruns your whole script on every interaction — cache the index (look up `@st.cache_resource` in the Streamlit docs) so uploading once means embedding once. Checkpoint: the second question answers in ~2-5 seconds, not 30+.
- [ ] Show your retrieval, not just your answer: add an expandable "retrieved chunks" section displaying the top-k chunks and their similarity scores. When an answer is bad, this panel tells you instantly whether retrieval failed (wrong chunks) or generation failed (right chunks, wrong answer) — the two failure modes of every RAG system.

<details><summary>Hints</summary>

- Build the context string with a loop and f-strings: `"\n\n".join(f"[source: {c['source']}]\n{c['text']}" for c in chunks)`. Keep the labels on their own line so the model can quote them cleanly.
- Streamlit state resets on every interaction; anything you must keep (the index, chat history) goes in `st.session_state` or a cache decorator.
- Uploaded files arrive as file-like objects: `uploaded_file.read().decode("utf-8")` gets you the text.
- If citations come back malformed or missing, show the model an example of a correctly cited answer inside the instructions — one example beats three rules.

</details>

**Definition of done:** A friend (or you in a fresh terminal) can run `streamlit run app.py`, upload files they choose, and get cited answers — and the app refuses out-of-scope questions in your grounding test.

## Project 3 — The RAG lab: measure, compare, swap

**Goal:** Stop guessing at settings. Run controlled experiments on chunk size and k using your Project 1 eval, then swap the numpy store for a real vector database and understand exactly what it adds (and does not).

**Milestones**

- [ ] Parameterize your indexer so chunk size and k are arguments, then run the four combinations — chunk size 150 vs 600 words (roughly 200 vs 800 tokens) × k=2 vs k=8 — over your 10-question eval. Collect hit-rate for each into one table. Checkpoint: a 4-row results table; expect small chunks + larger k to win on hit-rate, but verify — your documents may disagree.
- [ ] Hit-rate ignores answer quality, so grade faithfulness too: for one strong and one weak config, run all 10 questions end-to-end and mark each answer *faithful* (every claim traceable to a retrieved chunk), *hallucinated* (claims not in the chunks), or *refused*. Do it by hand — reading 20 answers against their sources teaches you more about RAG failure than any metric. Checkpoint: a 10×2 grid of labels and one sentence on where each config failed.
- [ ] Note the tradeoff k=8 exposes: more chunks means more chances the right one is present, but also more irrelevant text ("noise") in the prompt, slower answers, and higher API cost per question. Write down the total prompt length for k=2 vs k=8 on one question.
- [ ] Swap the store: rebuild your index in Chroma (`pip` already done; it runs embedded in your process, no server) or FAISS. Same chunks, same embedding model — only the storage and search change. Checkpoint: Chroma/FAISS returns the same top-1 chunk as your numpy search on at least 9/10 eval questions (tiny score differences are normal; FAISS's default index measures L2 distance, not cosine, unless you normalize — that discovery is part of the exercise).
- [ ] Time both stores on 100 repeated queries (`time.perf_counter`). Checkpoint: at a few hundred chunks the difference is negligible — write one sentence on when a real vector store earns its keep (millions of vectors, filtering by metadata, persistence, concurrent users) and when numpy is honestly fine.
- [ ] Write `findings.md` in your work folder: the results table, the faithfulness grid, your chunk-size/k recommendation for this corpus, and the store comparison. You will reuse this exact experimental habit in lesson 41 when you evaluate deployed models.

<details><summary>Hints</summary>

- Keep experiments honest: change one variable at a time and reuse the identical eval questions — you built the eval before tuning precisely so it can referee.
- Chroma's `collection.add()` can take your precomputed embeddings via the `embeddings=` argument, so you keep using the same sentence-transformers model instead of Chroma's default embedder — otherwise you are comparing two embedding models, not two stores.
- If FAISS's results look wrong, print the scores: large positive numbers that get *worse* with similarity means you are in L2-distance land — normalize vectors and use an inner-product index for cosine behavior.

</details>

**Definition of done:** `findings.md` exists with all three comparisons and a justified recommendation, and your app runs on the store and settings you chose because of data, not defaults.

## Stretch goals

- **Hybrid search:** meaning search misses exact identifiers ("lesson 22", "argsort", error codes). Implement keyword scoring (count query-word overlap per chunk, or search for the classic formula "BM25" and use the `rank-bm25` package), blend it with cosine scores, and re-run your eval — does hit-rate improve on keyword-heavy questions?
- **Re-ranking:** retrieve top-20 with fast cosine, then re-score just those 20 with a cross-encoder (a slower, more accurate relevance model — see the sbert.net docs) and keep the top-4. Measure the hit-rate change.
- **Chat memory:** support follow-up questions ("and how does that relate to overfitting?") by having the LLM rewrite the follow-up into a standalone query using the conversation history, before retrieval.
- **PDF support:** add PDF upload to your app with the `pypdf` package — messy real-world text extraction is half of production RAG work.

## If you get stuck

- **Retrieval returns junk:** debug retrieval alone, never through the LLM. Print the top-5 chunks and scores for the failing question. Top score below ~0.3 usually means no chunk actually covers the topic; right file but useless fragment means chunk size is wrong.
- **Good chunks, bad answer:** now it is a prompting problem — print the exact final prompt string. Nine times out of ten the context is malformed, truncated, or the instructions fight each other.
- **Shape errors in cosine math:** print `matrix.shape`, `query.shape`. A query embedding of shape `(1, 384)` vs `(384,)` is this lesson's favorite bug — `.flatten()` or indexing `[0]` fixes it.
- **First `SentenceTransformer(...)` call hangs:** it is downloading the model from huggingface.co (~90 MB) — let it finish; it is cached afterwards under `~/.cache`.
- Standing advice: read error tracebacks bottom-up (the last line names the real error), print shapes and intermediate values before suspecting anything exotic, ask an AI assistant for a HINT ("what concept am I missing?") rather than the solution, and type every line yourself — muscle memory is the point.

## Resources

- [sentence-transformers documentation](https://www.sbert.net) — the embedding library for Projects 1-3; see its semantic-search examples and the cross-encoder pages for the re-ranking stretch goal.
- [Chroma documentation](https://docs.trychroma.com) — the embedded vector database for Project 3's store swap.
- [FAISS](https://github.com/facebookresearch/faiss) — Meta's similarity-search library; the wiki explains index types and the L2-vs-inner-product distinction you will hit.
- Your LLM provider's own docs (from lesson 34) — for current model names, pricing, and rate limits; never trust a tutorial's hardcoded model string.

## Skills unlocked

- [ ] I can explain why retrieval fixes knowledge cutoff and hallucination, and what RAG stands for.
- [ ] I can chunk documents with overlap and explain the size tradeoff in prompt-space and retrieval terms.
- [ ] I can implement top-k cosine similarity search over an embedding matrix in vectorized numpy.
- [ ] I can build the full RAG loop: embed query → retrieve → stuff context → cited answer.
- [ ] I can make an LLM refuse questions its context does not cover, and verify that it does.
- [ ] I can measure a RAG system with retrieval hit-rate and a faithfulness check, and tune chunk size and k from evidence.
- [ ] I can use a real vector store (Chroma or FAISS) and say precisely when it beats a numpy array.
- [ ] I can diagnose a bad RAG answer as a retrieval failure vs a generation failure.

## Next up

Your bot now knows your documents at answer time — next you change the model itself: [36 · Fine-Tuning Open Models with LoRA](36-fine-tuning-open-models.md).
