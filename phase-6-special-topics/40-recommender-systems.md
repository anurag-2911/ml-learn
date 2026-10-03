# 40 · Recommender Systems: Matrix Factorization on MovieLens

**Phase 6 — Special Topics** · Estimated time: 1-2 weeks · Prerequisites: [06 · Pandas](../phase-0-foundations/06-pandas.md), [08 · Linear Algebra by Code](../phase-1-math/08-linear-algebra-by-code.md), [09 · Gradient Descent](../phase-1-math/09-calculus-and-gradient-descent.md), [30 · Embeddings: word2vec](../phase-4-nlp-transformers/30-embeddings-word2vec.md)

> Every time Netflix suggests a show, Spotify builds a listener's Discover Weekly, or YouTube queues the next video, the same core idea is at work: represent every user and every item as a small learned vector, and recommend items whose vectors point the same way as the user's. The course has already built this idea twice: cosine similarity in lesson 08 and embeddings in lessons 28/30. This lesson aims it at 100,000 real movie ratings and builds up to a personal payoff: rate 10 movies, retrain the model with those ratings inside it, and read the resulting top-10 recommendations off the screen. This is an optional-track lesson: skip it if the goal is to reach the capstone quickly.

## What this lesson builds

- **Project 1 — Neighborhood recommender**: an item-item collaborative filter on MovieLens that answers "people who liked *Toy Story* also liked..." using cosine similarity on a user-item matrix.
- **Project 2 — Matrix factorization from scratch**: a model that learns a 32-dimensional embedding for every user and every movie via SGD, beats a strong baseline on RMSE, and produces a **personal** top-10 after the learner is added as a user.
- **Project 3 — Honest evaluation harness**: a hit-rate@10 benchmark with a leave-last-out split, comparing popularity vs neighborhood CF vs matrix factorization in one table, plus a short written analysis of popularity bias and filter bubbles.

## Concepts covered

- **Explicit vs implicit feedback**: ratings a user typed in (explicit) vs preference inferred from behavior, like clicks and watch time (implicit).
- **User-item matrix**: a table (one row per user, one column per item, ratings in the cells), and why it is ~98% empty (sparsity).
- **Collaborative filtering (CF)**: recommending from patterns in *other people's* behavior, with no idea what a movie is "about".
- **Item-item similarity**: cosine similarity between rating columns, the same formula written in lesson 08.
- **Matrix factorization (MF)**: learning a small vector (embedding) per user and per item so that their dot product predicts the rating. It is the same embedding idea as in lessons 28 and 30, now appearing for the third time.
- **Training on observed entries only**: why SGD runs over the ratings that exist and never treats the empty cells as zeros.
- **Cold start**: why every embedding method fails for a brand-new user or item with no history.
- **Ranking metrics (precision@k)**: measuring "were the top k suggestions good?", and why low RMSE is not the same as good recommendations.
- **Popularity bias and filter bubbles**: how recommenders drift toward blockbusters and feed users more of what they already liked.

## Before starting

This lesson needs pandas and NumPy (lessons 05-06). PyTorch (lesson 24) is optional: Project 2 works in pure NumPy. If phase 0 is complete, everything on the `pip install` line below is already installed.

```bash
cd ~/ml/ml-learn
source .venv/bin/activate
pip install numpy pandas matplotlib
mkdir -p work/40-recommender-systems
cd work/40-recommender-systems
```

Download **MovieLens ml-latest-small** (100k ratings, 600 users, 9,000 movies; no account needed) from https://grouplens.org/datasets/movielens/ :

```bash
curl -L -O https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
unzip ml-latest-small.zip
ls ml-latest-small/
```

`ls` should show `ratings.csv`, `movies.csv`, `tags.csv` and `links.csv` (plus a `README.txt`). If the download link ever returns a 404 error, curl saves the site's error page under the zip's name and `unzip` fails on it. In that case, use the "ml-latest-small" link from the MovieLens page above instead. The lesson only needs `ratings.csv` (userId, movieId, rating, timestamp) and `movies.csv` (movieId, title, genres).

## Project 1 — Neighborhood recommender

**Goal:** Build "people who liked X also liked..." with nothing but pandas, a pivot table, and the cosine similarity written in lesson 08. This is item-item collaborative filtering, the algorithm Amazon ran for years.

**Milestones**

- [ ] Load `ratings.csv` with pandas and ask it questions as in lesson 06: how many ratings, users and movies are there? What does the rating distribution look like? Checkpoint: **100,836 ratings, 610 users**, and a distribution that peaks at 4.0 stars.
- [ ] Build the **user-item matrix** with `ratings.pivot_table(index='userId', columns='movieId', values='rating')`. This is explicit feedback: users typed these ratings in on purpose (unlike implicit feedback, e.g. "watched 80% of it"). Checkpoint: shape roughly `(610, 9724)`.
- [ ] Compute the **sparsity**: the fraction of cells that are NaN. Checkpoint: about **98.3% empty**. That number is worth a pause: almost everything the system would want to know about users is missing, which is the entire difficulty of this field.
- [ ] Write `cosine_sim(a, b)` for two movie columns, using **only the rows where both movies were rated** (mask the NaNs with `.notna()`). A movie pair with few common raters gives a noisy, meaningless similarity, so require a minimum overlap of about 50 common raters.
- [ ] Write `most_similar(movie_id, k=10)` that scores one movie against all others and returns the top k with titles (merge in `movies.csv`). Loop over columns; a few minutes of runtime is fine at this scale.
- [ ] **Checkpoint (the main one):** `most_similar(1)`, where movieId 1 is *Toy Story (1995)*, should return mostly other animation and family films, such as *Toy Story 2*, *The Incredibles*, *Aladdin* and *Finding Nemo*. If it returns unrelated films such as *Dirty Harry* or *Twilight* that share only 10-20 raters with *Toy Story*, the minimum-overlap threshold is missing or too low.
- [ ] Try 2-3 familiar movies (search `movies[movies['title'].str.contains('Matrix')]` to find ids) and sanity-check the neighbors by taste.
- [ ] Note what just happened: the system produced sensible genre groupings **without ever reading a genre label**, from behavior alone. That is collaborative filtering.

<details><summary>Hints</summary>

- Center each movie's column by subtracting its mean rating before cosine (a Pearson-style similarity; the textbook "adjusted cosine" subtracts each user's mean instead, and either works here). Otherwise the fact that everyone rates everything 3-5 stars makes all movies look similar.
- `common = a.notna() & b.notna()` gives the co-rater mask, and `common.sum()` is the overlap count to threshold on.
- For speed, precompute each column's mean once, and only compare against movies with ≥ 20 total ratings. That alone cuts 9,724 columns to about 1,300.
- NaN poisoning: if a similarity comes out NaN, the code divided by a zero norm. Return 0 (or skip) when a masked column has no variance.

</details>

**Definition of done:** `most_similar()` works for any movie id, Toy Story's neighbors look right, and sparsity and explicit-vs-implicit feedback can each be explained out loud in one sentence.

## Project 2 — Matrix factorization from scratch

**Goal:** Replace hand-crafted similarity with *learned* embeddings: a 32-dim vector per user and per movie, trained by SGD so that `user · movie` predicts the rating. Then comes the payoff milestone: join the dataset as a new user and get personal recommendations.

The prediction model (this is the whole model; the project is about training it):

```text
r̂(u, i) = μ + b_u + b_i + p_u · q_i

μ    global mean rating          b_u  user bias (grumpy vs generous rater)
b_i  item bias (acclaimed vs panned movie)
p_u  32-dim user embedding       q_i  32-dim item embedding
```

**Milestones**

- [ ] Split the *ratings* (not the users) 90/10 into train/test with a fixed seed. Every rating is a row `(userId, movieId, rating)`, and training uses that list of triples, not the pivot table. This is "SGD on observed entries only": the empty 98% is never touched.
- [ ] Build the **baseline before the model** (lesson 13 discipline): predict `μ` alone, then `μ + b_u + b_i` with biases computed as each user's/item's mean offset from `μ`. Checkpoint: test RMSE around **1.05** for the global mean, around **0.87-0.92** for the bias baseline. Write these numbers down: MF must beat them, or it is not learning.
- [ ] Initialize `P` (n_users × 32) and `Q` (n_items × 32) with small random normals (scale ~0.1), plus bias vectors `b_u`, `b_i` at zero. Map raw userId/movieId to 0..n-1 indices first (a dict, or `pd.factorize`).
- [ ] Write the SGD loop: for each training triple, compute the error `e = r - r̂`, then nudge `p_u`, `q_i`, `b_u`, `b_i` in the direction that shrinks `e²`, with L2 regularization (the weight penalty of lesson 19's Ridge, called weight decay in lesson 25) pulling them toward zero. Derive the four update rules by hand from `d(e²)/d(p_u)`. It is lesson 09's chain rule: two lines of algebra.
- [ ] Train ~20 epochs with learning rate ~0.01 and regularization ~0.05, shuffling each epoch, printing train and test RMSE per epoch. Checkpoint: test RMSE drops below the bias baseline, to roughly **0.86-0.88**. If train RMSE keeps falling while test rises, that is overfitting: raise regularization.
- [ ] Plot train vs test RMSE per epoch (lesson 07). Checkpoint: both fall, test flattens out.
- [ ] **The payoff: join the dataset.** Pick ~10 movies that were genuinely watched, mixing loved and hated ones (that contrast is what positions the new user's embedding), and append them to the ratings as new user id `9999`. Retrain. Predict a score for every movie the new user *hasn't* rated, sort, and print the top-10 with titles. Checkpoint: at least a few of the ten prompt the reaction "...yeah, actually." The person behind those ratings is now a row in `P`.
- [ ] Look up the new user's nearest-neighbor *users* by cosine on `P`, and a favorite movie's neighbors by cosine on `Q`. Note the echo: word2vec (lesson 30) put similar *words* nearby, makemore/GPT (lessons 28/31) put similar *tokens* nearby, and MF puts similar *users and movies* nearby. One idea runs through the entire curriculum.
- [ ] Try predicting for a made-up user id with **zero** ratings. The model can only output about `μ + b_i`, because it has no useful bias or embedding for that user: every new user gets the same list of generally well-rated movies. That is the **cold start problem**, and it is why every service quizzes new users about their tastes on day one.

<details><summary>Hints</summary>

- Save `q_i`'s old value before updating it, because `p_u`'s update needs the *pre-update* `q_i`. It is a classic silent bug.
- Exploding RMSE (or NaN) in epoch 1 means the learning rate is too high or the init is too large. Drop lr to 0.005.
- Vectorized-NumPy trick if the Python loop feels slow: index whole arrays per epoch (`P[u_idx] * Q[i_idx]` summed on axis 1). But a plain loop over 90k triples for 20 epochs finishes in a few minutes and is fine.
- Doing it in PyTorch instead? Use `nn.Embedding(n_users, 32)` for `P` and `nn.Embedding(n_items, 32)` for `Q`, MSE loss and Adam, and note that the updates did *not* have to be derived. That is the benefit of lesson 22's autograd.

</details>

**Definition of done:** MF beats the bias baseline on held-out RMSE, a *personal* top-10 list has been generated, and two ideas can be explained: cold start, and why training touches only observed entries.

## Project 3 — Honest evaluation

**Goal:** RMSE asks "did the model guess the star rating?", but users only ever see a top-10 list, so measure that instead. Build hit-rate@10, run a fair three-way comparison, and look directly at popularity bias.

**Milestones**

- [ ] Build a **leave-last-out split**: sort each user's ratings by timestamp and hold out their **last** liked movie (rating ≥ 4.0) as the test item; everything earlier is training data. This mimics reality (predict the future from the past), unlike a random split, which lets the model peek ahead in time.
- [ ] Define **hit-rate@10**: recommend 10 movies the user hasn't seen in training; hit-rate@10 is the fraction of users whose held-out movie appears in their top-10. (With a single held-out item this equals recall@10; precision@10 would be this divided by 10.) Small numbers are normal: the task is to find 1 movie among thousands.
- [ ] Baseline: the **popularity recommender**, where everyone gets the same 10 most-rated movies (minus ones they've seen). Checkpoint: hit-rate@10 roughly **0.03-0.10**. That is embarrassingly strong for zero intelligence, and it sets the bar.
- [ ] Rebuild the Project 1 CF and retrain the Project 2 MF on the leave-last-out training data only (otherwise the held-out movies leak into them), then wire both into the same harness. For CF, score each candidate movie by the sum of its similarities to the user's liked training movies, and compute the item-item similarities once, as a matrix, instead of calling `most_similar` for every user. Produce one table: rows = popularity / item-item CF / MF, columns = hit-rate@10, mean RMSE (where applicable). Checkpoint: the ranking-metric winner is **not guaranteed** to be the RMSE winner, and that mismatch is the lesson.
- [ ] Measure **popularity bias**: across all users' MF top-10s, what fraction of recommended movies are in the 100 most-rated? Plot recommendation counts per movie, sorted. Checkpoint: a steep "hockey stick", with a few movies recommended constantly and thousands never recommended at all.
- [ ] Write **3 observations about filter bubbles** in a `NOTES.md`, grounded in *this lesson's* outputs. For example: did the personal top-10 ever leave the genres of the rated movies? Who never gets recommended? If everyone trains on everyone's clicks and everyone gets shown the same winners, what happens to the training data next year? (This feedback loop is a live research and policy problem, and the 20-line bias measurement here is a real, honest version of it.)

<details><summary>Hints</summary>

- Exclude each user's *training* movies from their candidate list before taking the top 10, or every model degenerates into re-recommending watched films.
- Users with very few ratings make leave-last-out noisy. It is standard to evaluate only users with ≥ 5 train ratings; just say so in the table.
- Score all candidate movies for one user in a single vectorized operation (`Q @ P[u] + b_i + ...`), then `np.argsort`. A Python loop over 9k movies × 600 users will be very slow.

</details>

**Definition of done:** one comparison table, one popularity-bias plot, three written observations, and the ability to explain to a friend why "our RMSE improved" can be true while recommendations got more boring.

## Stretch goals

- **Implicit-feedback version:** throw away the star values, keep only "rated = interacted", label unobserved pairs as weak negatives by sampling, and train a binary MF. This is closer to how YouTube/Spotify actually train, since almost nobody rates things.
- **Beat cold start with content:** average the genre one-hots from `movies.csv` into a fallback item vector for movies with < 5 ratings, and blend it with the learned embedding.
- **2-D taste map:** reduce the item embeddings `Q` to 2-D with PCA (lesson 18) and scatter-plot 200 well-known movies with labels. Look for a horror corner and an animation corner.
- **Diversity re-ranking:** penalize each candidate by its max similarity to movies already picked for the top-10 (search for "MMR re-ranking"), and measure how much hit-rate a less bubbly list costs.

## Getting unstuck

- **All similarities near 1.0** in Project 1 → the columns were not mean-centered. Ratings all live in 3-5, so raw cosine thinks everything matches.
- **Toy Story's neighbors are random films** → the overlap threshold is missing or too low; require about 50 common raters.
- **RMSE goes to NaN** → learning rate too high, or a bug is updating with the *post-update* `q_i`. Print `e`, `p_u[:3]`, `q_i[:3]` for one triple and step through the math by hand.
- **Index errors like `index 193609 is out of bounds for axis 0 with size 9724`** → the code used raw movieIds (which go up to 193609) as array indices instead of the 0..n-1 remapping.
- Standing advice: read the traceback bottom-up; print shapes and a few actual values at every step; ask an AI assistant for a **hint, not a solution**; and type every line by hand, because the bugs fixed along the way are where the learning is.

## Resources

- [MovieLens datasets](https://grouplens.org/datasets/movielens/) — the dataset page; ml-latest-small for this lesson, ml-25m for anyone who wants to stress-test the code later.
- [Google's Recommendation Systems course](https://developers.google.com/machine-learning/recommendation) — a short, free walkthrough of the same pipeline (candidate generation → scoring → re-ranking) as it is built in industry. Read it *after* Project 2, when every part of it will look familiar.
- The classic paper behind Project 2 is Koren, Bell & Volinsky, *"Matrix Factorization Techniques for Recommender Systems"* (2009), from the Netflix Prize era; search for it.

## Skills unlocked

- [ ] I can explain explicit vs implicit feedback and why the user-item matrix is ~98% sparse.
- [ ] I can build an item-item collaborative filter with pandas and cosine similarity, with a co-rater threshold.
- [ ] I can implement matrix factorization with biases and train it by SGD on observed entries only, deriving the updates by hand.
- [ ] I can beat a global+user+item-bias baseline on held-out RMSE and prove it with a learning curve.
- [ ] I added myself to a real dataset and generated my own top-10 recommendations.
- [ ] I can implement hit-rate@10 with a leave-last-out split and explain why RMSE alone is misleading.
- [ ] I can measure popularity bias in a recommender and describe how filter bubbles form.
- [ ] I can point to embeddings in lessons 28, 30, 31 and this one, and explain the shared idea in one sentence.

## Next up

The models so far work on a laptop, and the next lesson makes them work for *other people* with experiment tracking, model serving, Docker and deployment: [41 · MLOps: Track, Serve, Containerize, Deploy](../phase-7-mlops-capstone/41-mlops-ship-your-models.md).
