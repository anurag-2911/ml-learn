# 35 · RAG: Build "Chat With My Documents"

**Phase 5 — LLMs** · Estimated time: 1-2 weeks · Prerequisites: [34 · Using LLMs: APIs, Prompting and Structured Output](34-llm-apis-prompting.md), [30 · Embeddings: Meaning as Geometry](../phase-4-nlp-transformers/30-embeddings-word2vec.md), [08 · Linear Algebra by Writing Your Own](../phase-1-math/08-linear-algebra-by-code.md)

> An LLM knows nothing about personal notes, a company's internal wiki, or anything written after its training ended, and when asked about them anyway, it confidently makes things up. RAG (retrieval-augmented generation) fixes this: it first searches the documents for the relevant passages, then hands them to the LLM with the instruction "answer from these, and cite your sources." It is the single most-built LLM application in industry today, and its core is surprisingly simple. This lesson builds the whole engine in numpy before touching any framework, combining the embedding idea from lesson 30 with the LLM API setup from lesson 34. By the end, a chatbot can be pointed at any folder of files and questioned about it.

## What this lesson builds

- **Project 1** — A retrieval engine in pure numpy: a script that chunks the documents, embeds them locally, and finds the most relevant chunk for any question, with a 10-question eval that proves it works.
- **Project 2** — A full "chat with my documents" web app (Streamlit or Gradio): upload files, ask questions, get answers with `[source: file.md]` citations, and a bot that says "I don't know" instead of guessing.
- **Project 3** — A RAG lab report: measured comparisons of chunk size and top-k on the Project 1 eval, plus the numpy store swapped for a real vector database (Chroma or FAISS).

## Concepts covered

- **Why LLMs need retrieval**: knowledge cutoff, private data, and hallucination (confident made-up answers).
- **Text embeddings**: turning a sentence into a vector of numbers where similar meanings land close together (the industrial-strength version of the word2vec built in lesson 30).
- **Chunking**: splitting documents into passages small enough to search and stuff into a prompt, and why chunks overlap.
- **Cosine similarity search**: ranking chunks by the angle between vectors (the same cosine coded in lesson 08).
- **The RAG loop**: embed the query → retrieve top-k chunks → stuff them into the prompt → answer with citations.
- **Vector stores**: what FAISS and Chroma actually do, and why a numpy array is the same thing at small scale.
- **Evaluating RAG**: retrieval hit-rate (did the right chunk come back?) and answer faithfulness (did the answer stick to the sources?).

## Before starting

1. Lesson 34 is finished, with a working LLM API setup: a provider account, plus an API key and model name in `work/34-llm-apis-prompting/.env`. `load_dotenv()` only searches the script's own folder and its parents, so copy that file into this lesson's folder right after step 3 (`cp work/34-llm-apis-prompting/.env work/35-rag-chat-with-your-docs/`; the `.env` line in `.gitignore` keeps this copy out of git too). Then turn the API call from lesson 34's `chat.py` into a small function, `chat(prompt)`, that sends one user message and returns the reply text. Any provider works. Check the provider's docs for current model names, and do not hardcode assumptions from a tutorial.
2. Activate the venv and install this lesson's tools:

```bash
cd ~/ml/ml-learn
source .venv/bin/activate
pip install sentence-transformers streamlit gradio chromadb faiss-cpu
```

   `sentence-transformers` downloads a small embedding model (~90 MB) on first use and runs it locally, even on a CPU: free, with no account and no API key. It is built on PyTorch, which was installed in lesson 24. Current PyTorch has no Intel Mac version, so on an Intel Mac this install fails: do this lesson in a free Google Colab notebook, as lesson 01 suggested, and build Project 2's app with Gradio, which runs inside a notebook too.

3. Create the work folder and a data folder:

```bash
mkdir -p work/35-rag-chat-with-your-docs/data
```

4. Get a document set. The perfect one is already here: this curriculum. Copy 15-20 lesson files in:

```bash
cp phase-0-foundations/*.md phase-1-math/*.md phase-2-classical-ml/*.md work/35-rag-chat-with-your-docs/data/
ls work/35-rag-chat-with-your-docs/data | wc -l
```

   The second command counts the files. It should print 20, right in the 15-20 range. Personal notes work too, as does any folder of `.md`/`.txt` files. RAG on familiar documents makes wrong answers easy to spot.

## Project 1 — RAG retrieval from scratch in numpy

**Goal:** Build the retrieval half of RAG with nothing but sentence-transformers and numpy, and prove with a 10-question eval that it finds the right chunk. No frameworks: the point is to see that the core of every vector database is ~5 lines of numpy.

**Milestones**

- [ ] Smoke-test embeddings in a throwaway script: embed the sentences "How do I train a neural network?", "What is backpropagation?", and "My cat sleeps all day" with `SentenceTransformer("all-MiniLM-L6-v2")`, then compute all pairwise cosine similarities with a numpy function written from scratch (dot product divided by the product of norms; see lesson 08). Checkpoint: the two ML sentences score clearly higher with each other (roughly 0.35-0.4) than either does with the cat sentence (roughly 0.1 or below, possibly slightly negative). Meaning is now geometry: this is lesson 30's idea, but for whole sentences.
- [ ] Write `chunk_text(text, chunk_size=150, overlap=30)` that splits a document into chunks of ~150 words, each starting 30 words before the previous one ended. Overlap exists so a sentence sitting on a chunk boundary appears whole in at least one chunk. Print the first 3 chunks of one lesson file. Checkpoint: chunk 2 starts with the last ~30 words of chunk 1.
- [ ] Write `build_index(folder)`: loop over every file in `data/`, chunk it, and keep two parallel structures. One is a list of dicts (`{"text": ..., "source": filename, "chunk_id": i}`). The other is one numpy matrix of all chunk embeddings (embed a whole list of texts in one `model.encode(list_of_texts)` call; it is much faster). Save with `np.save` and `json.dump`. Checkpoint: for the 20 lesson files the index holds roughly 200-600 chunks, and the matrix shape is `(num_chunks, 384)` (384 is this model's embedding size).
- [ ] Write `search(query, k=3)`: embed the query, compute cosine similarity against every row of the matrix in one vectorized expression (no Python loop; see lesson 05), and return the top-k chunks with scores. `np.argsort` gives ascending order, so the best matches are at the other end. Checkpoint: `search("how does gradient descent update weights?")` returns chunks from lesson 09 or 12 with top scores around 0.4-0.5.
- [ ] Build the eval set: write `eval.json` with 10 questions that can be answered from the documents, each with the filename that contains the answer, e.g. `{"question": "What does the chain rule let backprop do?", "expected_source": "09-calculus-and-gradient-descent.md"}`. Make them varied: some easy keyword matches, some paraphrased so no shared words appear.
- [ ] Write `evaluate.py`: for each question, check whether any top-3 result comes from the expected file. This metric is called **retrieval hit-rate**: the fraction of questions where the right source appears in the top-k. Print a per-question pass/fail table and the total. Checkpoint: hit-rate at least 8/10. If it is lower, read the misses. Usually the question can be answered from a different file than the one guessed (fix the eval), or the chunks are too small to carry meaning (fix chunking).

<details><summary>Hints</summary>

- Cosine against the whole matrix at once: normalize every embedding to unit length while building the index (divide each row by its norm), and normalize the query too. Then cosine similarity is just `matrix @ query`, one matrix-vector product for all chunks.
- Top-k without sorting everything backwards: `np.argsort(scores)[::-1][:k]`, or look up `np.argpartition` for the efficient version.
- Chunking by words is fine here: `words = text.split()`, then slice `words[start:start+chunk_size]` with `start` advancing by `chunk_size - overlap`. (Real systems count tokens; a token is roughly 3/4 of a word, which is good enough to reason with.)
- If every score looks oddly similar (all ~0.9), the chunks were probably embedded with some identical boilerplate attached, or un-normalized vectors were compared with dot product instead of cosine.

</details>

**Definition of done:** `python evaluate.py` prints a 10-row table and a hit-rate of 8/10 or better, and the learner can explain why the numpy matrix + cosine *is* a vector store.

## Project 2 — The full RAG app: retrieval + LLM + citations + UI

**Goal:** Complete the loop: retrieved chunks go into an LLM prompt, the answer comes back grounded in the documents with citations, and a web UI makes it usable by anyone. This is the pattern behind every "chat with your PDF" product.

**Milestones**

- [ ] Write `answer(question)` in a plain script first (no UI). Call the Project 1 `search()` with k=4, then build a prompt with three parts: instructions, the retrieved chunks each labeled with its source (`[source: 09-calculus-and-gradient-descent.md]\n<chunk text>`), and the question. Send it through the `chat(prompt)` function from Before starting. This step (pasting retrieved text into the prompt) is all that "augmented generation" means.
- [ ] Make the instructions strict, paraphrasing these rules: answer ONLY from the provided context; cite the source label after each claim; if the context does not contain the answer, reply "I don't know based on the provided documents." The last rule is the main defense against hallucination. Without it, the model fills gaps from its training data, and there is no way to tell which is which.
- [ ] Test grounding both ways. Ask 3 questions that the documents answer and 3 that they cannot ("Who won the 2022 World Cup?", "What is the capital of Brazil?", "What does lesson 99 cover?"). Checkpoint: in-scope answers include at least one `[source: ...]` citation each; all 3 out-of-scope questions get the "I don't know" response. If the model answers Brazil anyway, tighten the instructions and test again. The real workflow is prompt, test, tighten.
- [ ] Wrap it in a UI with Streamlit (Gradio works too; both turn a Python function into a web page in ~20 lines). Minimum: a file uploader accepting `.md`/`.txt`, an "index my files" button that chunks + embeds the uploads, a question box, and an answer area showing the cited sources under the answer. Run it:

```bash
streamlit run app.py
```

   Checkpoint: the app opens in the browser at `localhost:8501` (if no browser window opens by itself, open `http://localhost:8501` by hand; WSL2 forwards localhost to Windows automatically), accepts 3 uploaded files and a question, and shows an answer with sources. A Gradio app starts with `python app.py` instead and serves at `http://localhost:7860`; in Colab, `demo.launch()` shows it inside the notebook.
- [ ] Embedding on every rerun is slow, and Streamlit reruns the whole script on every interaction. Cache the index (look up `@st.cache_resource` in the Streamlit docs) so that uploading once means embedding once. Checkpoint: the second question answers in ~2-5 seconds, not 30+.
- [ ] Show the retrieval, not just the answer: add an expandable "retrieved chunks" section displaying the top-k chunks and their similarity scores. When an answer is bad, this panel shows instantly whether retrieval failed (wrong chunks) or generation failed (right chunks, wrong answer). These are the two failure modes of every RAG system.

<details><summary>Hints</summary>

- Build the context string with a loop and f-strings: `"\n\n".join(f"[source: {c['source']}]\n{c['text']}" for c in chunks)`. Keep the labels on their own line so the model can quote them cleanly.
- Streamlit state resets on every interaction; anything that must be kept (the index, chat history) goes in `st.session_state` or a cache decorator.
- Uploaded files arrive as file-like objects: `uploaded_file.read().decode("utf-8")` returns the text.
- If citations come back malformed or missing, show the model an example of a correctly cited answer inside the instructions. One example beats three rules.

</details>

**Definition of done:** A friend (or the app's author, in a fresh terminal) can start the app (`streamlit run app.py`, or `python app.py` for a Gradio app), upload files they choose, and get cited answers. The app also refuses the out-of-scope questions from the grounding test.

## Project 3 — The RAG lab: measure, compare, swap

**Goal:** Stop guessing at settings. Run controlled experiments on chunk size and k using the Project 1 eval, then swap the numpy store for a real vector database and understand exactly what it adds (and does not).

**Milestones**

- [ ] Parameterize the indexer so chunk size and k are arguments, then run the four combinations over the 10-question eval: chunk size 150 vs 600 words (roughly 270 vs 1,000 tokens on these Markdown files) × k=2 vs k=8. Collect hit-rate for each into one table. Checkpoint: a 4-row results table. Expect small chunks + larger k to win on hit-rate, but verify it, since the documents may disagree. Print `model.max_seq_length` too: all-MiniLM-L6-v2 reads at most 256 tokens and silently ignores the rest, so a 600-word chunk is searched by roughly its first quarter only, while the whole chunk still goes to the LLM. Note this next to the table in `findings.md`, since it can explain why large chunks lose.
- [ ] Hit-rate ignores answer quality, so grade faithfulness too: for one strong and one weak config, run all 10 questions end-to-end and mark each answer *faithful* (every claim traceable to a retrieved chunk), *hallucinated* (claims not in the chunks), or *refused*. Do it by hand: reading 20 answers against their sources teaches more about RAG failure than any metric. Checkpoint: a 10×2 grid of labels and one sentence on where each config failed.
- [ ] Note the tradeoff k=8 exposes: more chunks means more chances the right one is present, but also more irrelevant text ("noise") in the prompt, slower answers, and higher API cost per question. Write down the total prompt length for k=2 vs k=8 on one question.
- [ ] Swap the store: rebuild the index in Chroma (`pip` already done; it runs embedded in the Python process, no server) or FAISS. Same chunks, same embedding model: only the storage and search change. Checkpoint: Chroma/FAISS returns the same top-1 chunk as the numpy search on at least 9/10 eval questions (tiny score differences are normal; FAISS's default index measures L2 distance, not cosine, unless the vectors are normalized, and discovering that is part of the exercise).
- [ ] Time both stores on 100 repeated queries (`time.perf_counter`). Checkpoint: at a few hundred chunks the difference is negligible. Write one sentence on when a real vector store earns its keep (millions of vectors, filtering by metadata, persistence, concurrent users) and when numpy is perfectly fine.
- [ ] Write `findings.md` in the work folder: the results table, the faithfulness grid, a chunk-size/k recommendation for this corpus, and the store comparison. Lesson 41 reuses this exact experimental habit to evaluate deployed models.

<details><summary>Hints</summary>

- Keep experiments honest: change one variable at a time and reuse the identical eval questions. The eval was built before tuning precisely so that it can referee.
- Chroma's `collection.add()` can take the precomputed embeddings via the `embeddings=` argument, so the same sentence-transformers model stays in use instead of Chroma's default embedder. Otherwise the comparison is between two embedding models, not two stores.
- If FAISS's results look wrong, print the scores. Large positive numbers that get *worse* with similarity mean the index measures L2 distance: normalize the vectors and use an inner-product index for cosine behavior.

</details>

**Definition of done:** `findings.md` exists with all three comparisons and a justified recommendation, and the app runs on the store and settings chosen because of data, not defaults.

## Stretch goals

- **Hybrid search:** meaning search misses exact identifiers ("lesson 22", "argsort", error codes). Implement keyword scoring (count query-word overlap per chunk, or search for the classic formula "BM25" and use the `rank-bm25` package), blend it with cosine scores, and re-run the eval. Does hit-rate improve on keyword-heavy questions?
- **Re-ranking:** retrieve top-20 with fast cosine, then re-score just those 20 with a cross-encoder (a slower, more accurate relevance model; see the sbert.net docs) and keep the top-4. Measure the hit-rate change.
- **Chat memory:** support follow-up questions ("and how does that relate to overfitting?") by having the LLM rewrite the follow-up into a standalone query using the conversation history, before retrieval.
- **PDF support:** add PDF upload to the app with the `pypdf` package. Messy real-world text extraction is half of production RAG work.

## Getting unstuck

- **Retrieval returns junk:** debug retrieval alone, never through the LLM. Print the top-5 chunks and scores for the failing question. Top score below ~0.3 usually means no chunk actually covers the topic; right file but useless fragment means chunk size is wrong.
- **Good chunks, bad answer:** now it is a prompting problem, so print the exact final prompt string. Nine times out of ten the context is malformed, truncated, or the instructions fight each other.
- **Shape errors in cosine math:** print `matrix.shape`, `query.shape`. A query embedding of shape `(1, 384)` vs `(384,)` is this lesson's most common bug; `.flatten()` or indexing `[0]` fixes it.
- **First `SentenceTransformer(...)` call hangs:** it is downloading the model from huggingface.co (~90 MB). Let it finish; it is cached afterwards under `~/.cache`.
- Standing advice: read error tracebacks bottom-up (the last line names the real error), print shapes and intermediate values before suspecting anything exotic, ask an AI assistant for a hint ("what concept am I missing?") rather than the solution, and type every line by hand (muscle memory is the point).

## Resources

- [sentence-transformers documentation](https://www.sbert.net) — the embedding library for Projects 1-3; see its semantic-search examples and the cross-encoder pages for the re-ranking stretch goal.
- [Chroma documentation](https://docs.trychroma.com) — the embedded vector database for Project 3's store swap.
- [FAISS](https://github.com/facebookresearch/faiss) — Meta's similarity-search library; the wiki explains index types and the L2-vs-inner-product distinction that comes up in Project 3.
- The LLM provider's own docs (from lesson 34) — for current model names, pricing, and rate limits; never trust a tutorial's hardcoded model string.

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

The bot now knows the documents at answer time, and the next lesson changes the model itself: [36 · Fine-Tuning Open Models with LoRA](36-fine-tuning-open-models.md).
