# 40 · Recommender Systems: Matrix Factorization on MovieLens

**Phase 6 — Special Topics** · Estimated time: 1-2 weeks · Prerequisites: [06 · Pandas](../phase-0-foundations/06-pandas.md), [08 · Linear Algebra by Code](../phase-1-math/08-linear-algebra-by-code.md), [09 · Gradient Descent](../phase-1-math/09-calculus-and-gradient-descent.md), [30 · Embeddings: word2vec](../phase-4-nlp-transformers/30-embeddings-word2vec.md)

> Every time Netflix suggests a show, Spotify builds your Discover Weekly, or YouTube queues the next video, the same core idea is at work: represent every user and every item as a small learned vector, and recommend items whose vectors point the same way as yours. You already built this idea twice — cosine similarity in lesson 08 and embeddings in lessons 28/30 — and now you'll aim it at 100,000 real movie ratings. The payoff is personal: by the end, you will rate 10 movies yourself, retrain the model with *you* inside it, and read your own top-10 recommendations off the screen. This is an optional-track lesson: skip it if you're racing to the capstone, but it's one of the most satisfying builds in the whole curriculum.

## What you will build

- **Project 1 — Neighborhood recommender**: an item-item collaborative filter on MovieLens that answers "people who liked *Toy Story* also liked..." using cosine similarity on a user-item matrix.
- **Project 2 — Matrix factorization from scratch**: a model that learns a 32-dimensional embedding for every user and every movie via SGD, beats a strong baseline on RMSE, and produces **your own** personal top-10 after you add yourself as user.
- **Project 3 — Honest evaluation harness**: a hit-rate@10 benchmark with a leave-last-out split, comparing popularity vs neighborhood CF vs matrix factorization in one table, plus a short written analysis of popularity bias and filter bubbles.

## Concepts you will learn by doing

- **Explicit vs implicit feedback** — ratings a user typed in (explicit) vs behavior you infer preference from, like clicks and watch time (implicit).
- **User-item matrix** — a table with one row per user, one column per item, ratings in the cells — and why it is ~98% empty (sparsity).
- **Collaborative filtering (CF)** — recommending from patterns in *other people's* behavior, with no idea what a movie is "about".
- **Item-item similarity** — cosine similarity between rating columns; the same formula you wrote in lesson 08.
- **Matrix factorization (MF)** — learning a small vector (embedding) per user and per item so that their dot product predicts the rating. Same embedding idea as lessons 28 and 30 — third time it has reappeared.
- **Training on observed entries only** — why you run SGD over the ratings that exist and never treat the empty cells as zeros.
- **Cold start** — why every embedding method fails for a brand-new user or item with no history.
- **Ranking metrics: precision@k** — measuring "were the top k suggestions good?", and why low RMSE is not the same as good recommendations.
- **Popularity bias and filter bubbles** — how recommenders drift toward blockbusters and feed you more of what you already liked.

## Before you start

You need pandas and NumPy (lessons 05-06). PyTorch (lesson 24) is optional — Project 2 works in pure NumPy. If you did phase 0, everything on the `pip install` line below is already installed.

```bash
cd ~/ml/ml-learn
source .venv/bin/activate
pip install numpy pandas matplotlib
mkdir -p work/40-recommender-systems
cd work/40-recommender-systems
```

Download **MovieLens ml-latest-small** (100k ratings, 600 users, 9,000 movies — no account needed) from https://grouplens.org/datasets/movielens/ :

```bash
curl -L -O https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
unzip ml-latest-small.zip
ls ml-latest-small/
```

`ls` should show `ratings.csv`, `movies.csv`, `tags.csv` and `links.csv` (plus a `README.txt`). If the download link ever 404s, curl saves the site's error page under the zip's name and `unzip` fails on it — grab the "ml-latest-small" link from the MovieLens page above instead. You'll only need `ratings.csv` (userId, movieId, rating, timestamp) and `movies.csv` (movieId, title, genres).

## Project 1 — Neighborhood recommender

**Goal:** Build "people who liked X also liked..." with nothing but pandas, a pivot table, and the cosine similarity you wrote in lesson 08. This is item-item collaborative filtering — the algorithm Amazon ran for years.

**Milestones**

- [ ] Load `ratings.csv` with pandas and interrogate it like in lesson 06: how many ratings, users, movies? What does the rating distribution look like? Checkpoint: you should see **100,836 ratings, 610 users**, and a distribution peaking at 4.0 stars.
- [ ] Build the **user-item matrix** with `ratings.pivot_table(index='userId', columns='movieId', values='rating')`. This is explicit feedback: users typed these ratings in on purpose (unlike implicit feedback, e.g. "watched 80% of it"). Checkpoint: shape roughly `(610, 9724)`.
- [ ] Compute the **sparsity**: fraction of cells that are NaN. Checkpoint: about **98.3% empty**. Sit with that number — almost everything you "know" about users is missing, which is the entire difficulty of this field.
- [ ] Write `cosine_sim(a, b)` for two movie columns, using **only the rows where both movies were rated** (mask the NaNs with `.notna()`). A movie pair with fewer than ~10 common raters gives garbage similarity — require a minimum overlap.
- [ ] Write `most_similar(movie_id, k=10)` that scores one movie against all others and returns the top k with titles (merge in `movies.csv`). Loop over columns; a few minutes of runtime is fine at this scale.
- [ ] **Checkpoint (the big one):** `most_similar(1)` — movieId 1 is *Toy Story (1995)* — should return mostly other animation/family films: think *Toy Story 2*, *A Bug's Life*, *Aladdin*, *The Lion King*. If you get obscure films with 3 raters, your minimum-overlap filter is missing.
- [ ] Try 2-3 movies you know (search `movies[movies['title'].str.contains('Matrix')]` to find ids) and sanity-check the neighbors by taste.
- [ ] Note what just happened: the system produced sane genre groupings **without ever reading a genre label**. Pure behavior. That's collaborative filtering.

<details><summary>Hints</summary>

- Center each movie's column by subtracting its mean rating before cosine ("adjusted cosine"). Otherwise the fact that everyone rates everything 3-5 stars makes all movies look similar.
- `common = a.notna() & b.notna()` gives you the co-rater mask; `common.sum()` is the overlap count to threshold on.
- For speed, precompute each column's mean once, and only compare against movies with ≥ 20 total ratings — that alone cuts 9,724 columns to about 1,200.
- NaN poisoning: if a similarity comes out NaN, you divided by a zero norm — return 0 (or skip) when a masked column has no variance.

</details>

**Definition of done:** `most_similar()` works for any movie id, Toy Story's neighbors pass the smell test, and you can explain sparsity and explicit-vs-implicit feedback out loud in one sentence each.

## Project 2 — Matrix factorization from scratch

**Goal:** Replace hand-crafted similarity with *learned* embeddings: a 32-dim vector per user and per movie, trained by SGD so that `user · movie` predicts the rating. Then the payoff milestone: add yourself to the dataset and get your own recommendations.

The prediction model (this is the whole model — the project is training it):

```text
r̂(u, i) = μ + b_u + b_i + p_u · q_i

μ    global mean rating          b_u  user bias (grumpy vs generous rater)
b_i  item bias (acclaimed vs panned movie)
p_u  32-dim user embedding       q_i  32-dim item embedding
```

**Milestones**

- [ ] Split the *ratings* (not the users) 90/10 into train/test with a fixed seed. Every rating is a row `(userId, movieId, rating)` — that list of triples, not the pivot table, is what you train on. This is "SGD on observed entries only": the empty 98% is never touched.
- [ ] Build the **baseline before the model** (lesson 14 discipline): predict `μ` alone, then `μ + b_u + b_i` with biases computed as each user's/item's mean offset from `μ`. Checkpoint: test RMSE around **1.05** for global-mean, around **0.87-0.92** for the bias baseline. Write these numbers down — MF must beat them or it's not learning.
- [ ] Initialize `P` (n_users × 32) and `Q` (n_items × 32) with small random normals (scale ~0.1), plus bias vectors `b_u`, `b_i` at zero. Map raw userId/movieId to 0..n-1 indices first (a dict, or `pd.factorize`).
- [ ] Write the SGD loop: for each training triple, compute the error `e = r - r̂`, then nudge `p_u`, `q_i`, `b_u`, `b_i` in the direction that shrinks `e²`, with L2 regularization (lesson 12's weight decay) pulling them toward zero. Derive the four update rules yourself from `d(e²)/d(p_u)` — it's lesson 09's chain rule, two lines of algebra.
- [ ] Train ~20 epochs with learning rate ~0.01 and regularization ~0.02, shuffling each epoch, printing train and test RMSE per epoch. Checkpoint: test RMSE drops below your bias baseline — expect roughly **0.86-0.88**. If train RMSE keeps falling while test rises, that's overfitting: raise regularization.
- [ ] Plot train vs test RMSE per epoch (lesson 07). Checkpoint: both fall, test flattens out.
- [ ] **The payoff — add yourself.** Pick ~10 movies you have genuinely seen (mix loved and hated ones, that contrast is what positions your embedding) and append them to the ratings as new user id `9999`. Retrain. Predict your score for every movie you *haven't* rated, sort, and print your top-10 with titles. Checkpoint: at least a few of the ten make you think "...yeah, actually." Congratulations — you are now a row in `P`.
- [ ] Look up your nearest-neighbor *users* by cosine on `P`, and your favorite movie's neighbors by cosine on `Q`. Note the echo: word2vec (lesson 30) put similar *words* nearby, makemore/GPT (lessons 28/31) put similar *tokens* nearby, MF puts similar *users and movies* nearby. One idea, entire curriculum.
- [ ] Try predicting for a fabricated user id with **zero** ratings. The model can only output `μ` — it has no embedding worth anything for them. That's the **cold start problem**, and it's why every service interrogates you about your tastes on day one.

<details><summary>Hints</summary>

- Save `q_i`'s old value before updating it, because `p_u`'s update needs the *pre-update* `q_i`. Classic silent bug.
- Exploding RMSE (or NaN) in epoch 1 = learning rate too high or init too large. Drop lr to 0.005.
- Vectorized-NumPy trick if the Python loop feels slow: index whole arrays per epoch (`P[u_idx] * Q[i_idx]` summed on axis 1) — but a plain loop over 90k triples for 20 epochs finishes in a few minutes and is fine.
- Doing it in PyTorch instead? `nn.Embedding(n_users, 32)` for `P` and `Q`, MSE loss, Adam — and note you did *not* have to derive the updates. That's what lesson 22's autograd bought you.

</details>

**Definition of done:** MF beats the bias baseline on held-out RMSE, you have a top-10 list generated *for you personally*, and you can explain cold start and why training touches only observed entries.

## Project 3 — Honest evaluation

**Goal:** RMSE asks "did you guess the star rating?", but users only ever see a top-10 list — so measure that instead. Build hit-rate@10, run a fair three-way bake-off, and stare directly at popularity bias.

**Milestones**

- [ ] Build a **leave-last-out split**: sort each user's ratings by timestamp and hold out their **last** liked movie (rating ≥ 4.0) as the test item; everything earlier is training data. This mimics reality — predict the future from the past — unlike a random split, which lets the model peek ahead in time.
- [ ] Define **hit-rate@10**: recommend 10 movies the user hasn't seen in training; hit-rate@10 = fraction of users whose held-out movie appears in their top-10. (With a single held-out item this equals recall@10; precision@10 would be this divided by 10.) Small numbers are normal — you're finding 1 movie among thousands.
- [ ] Baseline: the **popularity recommender** — everyone gets the same 10 most-rated movies (minus ones they've seen). Checkpoint: hit-rate@10 roughly **0.03-0.10**. Embarrassingly strong for zero intelligence; this is the bar.
- [ ] Wire your Project 1 CF and retrained Project 2 MF into the same harness and produce one table: rows = popularity / item-item CF / MF, columns = hit-rate@10, mean RMSE (where applicable). Checkpoint: the ranking-metric winner is **not guaranteed** to be the RMSE winner — that mismatch is the lesson.
- [ ] Measure **popularity bias**: across all users' MF top-10s, what fraction of recommended movies are in the 100 most-rated? Plot recommendation counts per movie, sorted. Checkpoint: a steep "hockey stick" — a few movies recommended constantly, thousands never recommended at all.
- [ ] Write **3 observations about filter bubbles** in a `NOTES.md` — grounded in *your* outputs, e.g.: did your personal top-10 ever leave the genres you rated? Who never gets recommended? If everyone trains on everyone's clicks and everyone gets shown the same winners, what happens to the training data next year? (This feedback loop is a live research and policy problem — your 20-line bias measurement is a real, honest version of it.)

<details><summary>Hints</summary>

- Exclude each user's *training* movies from their candidate list before taking the top 10, or every model degenerates into re-recommending watched films.
- Users with very few ratings make leave-last-out noisy — it's standard to evaluate only users with ≥ 5 train ratings; just say so in your table.
- Score all candidate movies for one user in a single vectorized op (`Q @ P[u] + b_i + ...`), then `np.argsort` — a Python loop over 9k movies × 600 users will crawl.

</details>

**Definition of done:** one comparison table, one popularity-bias plot, three written observations. You can explain to a friend why "our RMSE improved" can be true while recommendations got more boring.

## Stretch goals

- **Implicit-feedback version:** throw away the star values, keep only "rated = interacted", label unobserved pairs as weak negatives by sampling, and train a binary MF. This is closer to how YouTube/Spotify actually train, since almost nobody rates things.
- **Beat cold start with content:** average the genre one-hots from `movies.csv` into a fallback item vector for movies with < 5 ratings, and blend it with the learned embedding.
- **2-D taste map:** PCA (lesson 18) your item embeddings `Q` down to 2-D and scatter-plot 200 well-known movies with labels. Look for a horror corner and an animation corner.
- **Diversity re-ranking:** penalize each candidate by its max similarity to movies already picked for the top-10 (search for "MMR re-ranking"), and measure how much hit-rate you pay for a less bubbly list.

## If you get stuck

- **All similarities near 1.0** in Project 1 → you forgot to mean-center; ratings all live in 3-5 so raw cosine thinks everything matches.
- **Toy Story's neighbors are random obscure films** → overlap threshold missing or too low; require ≥ 10-20 common raters.
- **RMSE goes to NaN** → learning rate too high, or a bug is updating with the *post-update* `q_i`. Print `e`, `p_u[:3]`, `q_i[:3]` for one triple and step through the math by hand.
- **Index errors like `9724 is out of bounds`** → you used raw movieIds (which go up to 193609) as array indices instead of your 0..n-1 remapping.
- Standing advice: read the traceback bottom-up; print shapes and a few actual values at every step; ask an AI assistant for a **hint, not a solution**; and type every line yourself — the bugs you fix are where the learning is.

## Resources

- [MovieLens datasets](https://grouplens.org/datasets/movielens/) — the dataset page; ml-latest-small for this lesson, ml-25m if you want to stress-test your code later.
- [Google's Recommendation Systems course](https://developers.google.com/machine-learning/recommendation) — short, free walkthrough of the same pipeline (candidate generation → scoring → re-ranking) as it's built in industry; read it *after* Project 2 and enjoy recognizing everything.
- The classic paper behind Project 2 is Koren, Bell & Volinsky, *"Matrix Factorization Techniques for Recommender Systems"* (2009) — from the Netflix Prize era; search for it.

## Skills unlocked

- [ ] I can explain explicit vs implicit feedback and why the user-item matrix is ~98% sparse.
- [ ] I can build an item-item collaborative filter with pandas and cosine similarity, with a co-rater threshold.
- [ ] I can implement matrix factorization with biases and train it by SGD on observed entries only, deriving the updates myself.
- [ ] I can beat a global+user+item-bias baseline on held-out RMSE and prove it with a learning curve.
- [ ] I added myself to a real dataset and generated my own top-10 recommendations.
- [ ] I can implement hit-rate@10 with a leave-last-out split and explain why RMSE alone is misleading.
- [ ] I can measure popularity bias in a recommender and describe how filter bubbles form.
- [ ] I can point to embeddings in lessons 28, 30, 31 and this one, and explain the shared idea in one sentence.

## Next up

Your models work on your laptop — time to make them work for *other people*: experiment tracking, model serving, Docker, and deployment in [41 · MLOps: Track, Serve, Containerize, Deploy](../phase-7-mlops-capstone/41-mlops-ship-your-models.md).
