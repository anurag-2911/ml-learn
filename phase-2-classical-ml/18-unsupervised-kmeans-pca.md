# 18 · Unsupervised Learning: k-Means and PCA from Scratch

**Phase 2 — Classical Machine Learning** · Estimated time: 1-2 weeks · Prerequisites: [05 · NumPy](../phase-0-foundations/05-numpy.md), [07 · Data Visualization](../phase-0-foundations/07-data-visualization.md), [08 · Linear Algebra by Code](../phase-1-math/08-linear-algebra-by-code.md), [11 · k-Nearest Neighbors](11-first-model-knn.md)

> Every model you have built so far was told the right answer during training — that is **supervised learning**. This lesson removes the answers. In **unsupervised learning** the data has no labels, and the job is to find hidden structure on your own: groups, patterns, directions that matter. You will build the two workhorses of this world from scratch — k-means (finds groups) and PCA (finds directions) — and use them for two genuinely cool things: compressing the colors of a real photo, and flattening 64-dimensional handwritten digits onto a 2D map where the digits sort themselves into islands. The photo posterizer is the most shareable thing you have built so far. Post it.

## What you will build

- **A k-means clusterer in pure NumPy** — with an animation (a series of plots) showing the centroids walking into the middle of each data blob, plus an elbow plot for choosing k.
- **A photo posterizer** — your own k-means run on the pixels of a real photo, producing versions of the image painted with only 4, 8, and 16 colors. This is a real compression technique called color quantization.
- **A PCA implementation and a 2D map of handwritten digits** — center, covariance, eigenvectors, project. Then the same map made with t-SNE, a tool you use rather than build.

## Concepts you will learn by doing

- **Supervised vs unsupervised** — learning with answer labels vs finding structure without them.
- **k-means** — repeat two steps: *assign* each point to its nearest centroid, *update* each centroid to the mean of its points.
- **Centroid** — the center point of a cluster (just the average of its members).
- **Inertia** — total squared distance from every point to its assigned centroid; k-means tries to make this small.
- **Choosing k with the elbow method** — plot inertia for k = 1, 2, 3, ... and look for the bend.
- **k-means as compression** — replace millions of pixel colors with k representative colors.
- **PCA (Principal Component Analysis)** — find the directions along which the data varies the most.
- **Covariance matrix + eigenvectors** — the machinery behind PCA, using `np.linalg.eigh` (this is lesson 08's eigenvector intuition finally paying off).
- **Dimensionality reduction** — squashing many columns down to 2 so you can *see* your data.
- **t-SNE / UMAP** — fancier map-making tools you will use as black boxes for now.

## Before you start

Check you can do these (from earlier lessons): compute pairwise distances with NumPy broadcasting (lesson 11), make scatter plots with colors (lesson 07), and explain in one sentence what an eigenvector is (lesson 08 — "a direction a matrix only stretches, never rotates").

Activate your venv and install what's needed (from the repo root):

```bash
source .venv/bin/activate
pip install numpy matplotlib scikit-learn pillow
```

Pillow is a library for opening image files as arrays of numbers.

Create your work folder:

```bash
mkdir -p work/18-unsupervised-kmeans-pca
cd work/18-unsupervised-kmeans-pca
```

**Dataset downloads: none.** Project 1 generates fake data, Project 2 uses any photo you already own, and Project 3's digits ship inside scikit-learn. Put a photo of your own in your work folder as `photo.jpg`. To drag one in, open the folder in your file manager: `open .` on macOS, `explorer.exe .` on Windows (WSL2), `xdg-open .` on Linux. Or copy it in the terminal: on macOS and Linux, `cp ~/Pictures/YOUR-PHOTO.jpg photo.jpg`; on Windows (WSL2) your Windows files are under `/mnt/c/Users/`, so `cp /mnt/c/Users/YOUR-WINDOWS-NAME/Pictures/YOUR-PHOTO.jpg photo.jpg`.

Pick a colorful photo (a sunset, a market, a parrot) — the effect is much more dramatic than on a grey screenshot.

## Project 1 — k-means from scratch

**Goal:** Implement the k-means assign/update loop in NumPy and *watch* it converge by plotting every iteration on 2D blob data.

**Milestones**

- [ ] Make practice data you know the answer to: `from sklearn.datasets import make_blobs` then `X, y_true = make_blobs(n_samples=500, centers=4, cluster_std=0.9, random_state=42)`. `X` is 500 points in 2D; ignore `y_true` while clustering — pretending labels don't exist is the whole point. Scatter-plot `X`. Checkpoint: you should see 4 distinct blobs.
- [ ] Write `init_centroids(X, k, rng)`: pick k *distinct* random rows of X as the starting centroids. Use `rng.choice(len(X), size=k, replace=False)`. Starting from real data points guarantees no centroid begins in empty space.
- [ ] Write `assign(X, centroids)`: for every point, find the index of its nearest centroid. This is the same distance computation you wrote for kNN in lesson 11 — squared Euclidean distance is fine (skip the square root; it doesn't change who is nearest). Return an array of 500 cluster indices. Checkpoint: `assign(X, centroids).shape == (500,)` and the values are all in `0..k-1`.
- [ ] Write `update(X, labels, k)`: the new centroid j is `X[labels == j].mean(axis=0)`. Handle the edge case where a cluster has zero points (see Hints).
- [ ] Write `inertia(X, labels, centroids)`: sum of squared distances from each point to *its own* centroid.
- [ ] Wire the loop: init, then repeat assign → update, saving a scatter plot each iteration — points colored by cluster label, centroids drawn as big black X markers — to `frames/iter_00.png`, `iter_01.png`, ... Stop when the labels stop changing (compare with `np.array_equal`). Checkpoint: it converges in under ~15 iterations, printed inertia goes *down every single iteration*, and in the final frame the black X's sit in the middle of each blob.
- [ ] Flip through the frames in order (open the folder in VS Code and arrow-key through them). You are watching an optimization algorithm think. Optional: stitch them into a GIF (see Hints).
- [ ] Run k-means with a *bad* seed a few times (change `rng`). Notice the final inertia differs run to run — k-means only finds a *local* best. Fix: run it 5 times and keep the run with lowest inertia (this is what sklearn's `n_init` does).
- [ ] **Elbow method:** loop k from 1 to 10, run your k-means for each, and plot k vs final inertia. Checkpoint: the curve drops steeply then flattens, with a visible bend (the "elbow") at k=4 — the true number of blobs, discovered without ever peeking at `y_true`.

<details><summary>Hints</summary>

- Distances without loops: `d = ((X[:, None, :] - centroids[None, :, :])**2).sum(axis=2)` gives a `(n_points, k)` matrix; then `labels = d.argmin(axis=1)` and inertia is `d[np.arange(len(X)), labels].sum()`.
- Empty cluster: if `labels == j` matches nothing, `.mean(axis=0)` returns NaN and poisons everything after. When it happens, re-seed that centroid to a random data point instead.
- Use `rng = np.random.default_rng(0)` and pass it around, so runs are reproducible while you debug.
- GIF in three lines with Pillow: open the frame PNGs with `Image.open`, then `frames[0].save("kmeans.gif", save_all=True, append_images=frames[1:], duration=400, loop=0)`.

</details>

**Definition of done:** Your k-means converges on the blobs with centroids centered in each blob, inertia decreases monotonically, and your elbow plot points at k=4.

## Project 2 — Posterize a photo

**Goal:** Run *your* k-means from Project 1 on the pixels of a real photo — each pixel is just a point in 3D (R, G, B) space — and repaint the photo using only the k centroid colors. You are building a color-quantization compressor.

**Milestones**

- [ ] Load the photo as a NumPy array:

  ```python
  from PIL import Image
  import numpy as np
  img = np.asarray(Image.open("photo.jpg").convert("RGB"))
  print(img.shape, img.dtype)   # (height, width, 3) uint8
  ```

  Checkpoint: shape is `(H, W, 3)` and values run 0-255.
- [ ] Reshape it into a dataset: `pixels = img.reshape(-1, 3).astype(np.float64)`. A 1000×800 photo is now 800,000 "points" in 3D — same shape of problem as the blobs, just more rows and one more column. Your Project 1 code should run on it *unchanged*.
- [ ] Speed trick: fitting on 800k points is slow, and unnecessary. Fit the centroids on a random sample of ~10,000 pixels (`rng.choice`), then do one final `assign` pass over *all* pixels using those centroids.
- [ ] Rebuild the image: `new_pixels = centroids[labels]` (fancy indexing — look how much work that one line does), then reshape back to `(H, W, 3)`, convert with `.clip(0,255).astype(np.uint8)`, and save with `Image.fromarray(...).save("posterized_k8.png")`.
- [ ] Do it for k = 4, 8, 16. Checkpoint: k=4 looks like a bold screen-printed poster, k=8 looks stylized, and k=16 looks almost like the original until you zoom into a sky or a skin-tone gradient.
- [ ] Make one comparison figure with matplotlib: original plus the three posterized versions in a 2×2 grid (`plt.subplots(2, 2)`, `ax.imshow(...)`, title each with its k). Save it as `comparison.png`.
- [ ] Appreciate the compression: the original stores 24 bits per pixel; with k=16 you only need 4 bits per pixel plus a tiny 16-color palette — 6× smaller. Print those numbers for your photo.
- [ ] **Share it.** This is the most post-worthy artifact of the course so far — the 2×2 grid makes a great "I built this from scratch" post. Commit the images to your repo too.

<details><summary>Hints</summary>

- If assigning all 800k pixels at once eats your RAM (the broadcasted distance matrix is `n_pixels × k × 3`), assign in chunks of ~100k pixels with a plain Python loop over slices.
- Ugly muddy colors usually mean you fit on uint8 data and integer division mangled the means — cast to float *before* clustering, back to uint8 only at save time.
- If the output looks like static noise, you probably indexed `centroids[labels]` with labels from a *different* run than the centroids. Fit once, keep both together.

</details>

**Definition of done:** Three posterized images plus a labeled 2×2 comparison grid, all produced by the k-means *you* wrote, saved in your work folder.

## Project 3 — PCA from scratch

**Goal:** Implement PCA in four lines of linear algebra and use it to project 64-dimensional handwritten digits onto 2D, where the ten digits separate into visible clusters — structure found without labels.

**Milestones**

- [ ] Load the data: `from sklearn.datasets import load_digits` then `digits = load_digits()`. `digits.data` is `(1797, 64)` — each row is an 8×8 grayscale digit image flattened into 64 numbers. Plot a few with `plt.imshow(digits.data[i].reshape(8, 8), cmap="gray")` to meet the data. Checkpoint: you can recognize the digits (squint — they're tiny).
- [ ] You cannot scatter-plot 64 dimensions. PCA's promise: find the 2 directions in 64D space along which the data spreads out the most, and shadow everything onto them. "Direction of maximal variance" = the line along which the cloud of points is most stretched out.
- [ ] Step 1 — **center**: `Xc = X - X.mean(axis=0)`. PCA measures spread around the middle, so the middle must be at zero.
- [ ] Step 2 — **covariance matrix**: `C = np.cov(Xc, rowvar=False)` — a 64×64 table where entry (i, j) says how features i and j vary together. Checkpoint: `C.shape == (64, 64)` and `np.allclose(C, C.T)` is True (it's symmetric).
- [ ] Step 3 — **eigendecomposition**: `eigvals, eigvecs = np.linalg.eigh(C)` (`eigh` is the version for symmetric matrices — faster and returns real numbers). Lesson 08 payoff: the eigenvectors of the covariance matrix *are* the directions of maximal variance, and each eigenvalue is the amount of variance along its eigenvector. Careful: `eigh` sorts eigenvalues *ascending*, so the best directions are the **last** columns.
- [ ] Step 4 — **project**: take the top-2 eigenvectors as a `(64, 2)` matrix `W`, then `X2 = Xc @ W`. Sixty-four numbers per digit have become two. Checkpoint: `X2.shape == (1797, 2)`.
- [ ] Scatter-plot `X2` colored by the true digit: `plt.scatter(X2[:,0], X2[:,1], c=digits.target, cmap="tab10", s=8)` plus `plt.colorbar()`. Note we use labels only to *color the picture*, never to compute the projection. Checkpoint: clear islands — 0s clumped together far from 1s, 6s in their own region. Some digits smear into each other (4/9, 3/8/5 are lookalikes); that's honest, not a bug.
- [ ] Print the **explained variance**: `eigvals[-2:].sum() / eigvals.sum()`. Checkpoint: roughly 0.2 — two dimensions carry about a fifth of the total variance, yet the map is already readable.
- [ ] Sanity-check against the professionals: `from sklearn.decomposition import PCA; X2_sk = PCA(n_components=2).fit_transform(X)`. Checkpoint: sklearn's scatter matches yours up to a possible mirror flip of either axis (an eigenvector times −1 is an equally valid eigenvector).
- [ ] Now try **t-SNE**, a nonlinear map-maker designed purely for pretty 2D visualization: `from sklearn.manifold import TSNE; X2_t = TSNE(n_components=2, random_state=0).fit_transform(X)` (takes ~a minute). Plot it the same way. Checkpoint: ten tight, well-separated islands — noticeably crisper than PCA. **UMAP** (`pip install umap-learn`) is a faster modern alternative; try it if curious (not on an Intel Mac: numba, a library it needs, has no current Intel Mac version; use Google Colab, which has UMAP preinstalled). Use these as tools — you will not implement them.
- [ ] One-sentence journal entry: when would you reach for PCA (fast, linear, keeps global structure, reusable as a preprocessing step) vs t-SNE/UMAP (slower, for visualization only)? You'll use exactly this trick to draw maps of *word meanings* in [lesson 30](../phase-4-nlp-transformers/30-embeddings-word2vec.md).

<details><summary>Hints</summary>

- Wrong-looking blob with no structure? Most common cause: forgetting to center, or projecting the *uncentered* X onto W. Center once, use `Xc` everywhere.
- Grabbing the top eigenvectors: `W = eigvecs[:, -2:][:, ::-1]` takes the last two columns and puts the biggest first. Columns are eigenvectors, not rows.
- Your plot mirrored vs sklearn's? Sign flips are expected; a *rotated or squashed* plot is not — that means you took eigenvectors from the small end.

</details>

**Definition of done:** Your from-scratch 2D digits map shows visibly separated digit clusters, matches sklearn PCA up to sign flips, and you produced a t-SNE map alongside it.

## Stretch goals

- **k-means++ initialization:** instead of uniformly random starting centroids, pick each new one with probability proportional to its squared distance from the nearest already-chosen centroid. Implement it and compare final inertia over 10 runs vs plain random init.
- **PCA as compression:** project the digits onto the top-k components and reconstruct (`Xc @ W @ W.T + mean`). Show the same digit reconstructed from 2, 8, 16, 32 components — watch it sharpen.
- **Cluster the digits:** run your k-means with k=10 directly on the 64D digits data (no labels!), then check how well clusters align with true digits. Show each cluster's centroid as an 8×8 image — several look like ghostly average digits.
- **Posterizer CLI:** wrap Project 2 in a script — `python posterize.py photo.jpg 8` — using `argparse` from lesson 04, so friends can run it.

## If you get stuck

- **Shape bugs dominate this lesson.** Print `.shape` at every step of the distance broadcast; the `(n, k)` distance matrix, `(n,)` labels, and `(k, d)` centroids must line up or `argmin`/indexing silently does the wrong thing on the wrong axis.
- **NaNs appearing?** Almost always an empty cluster (Project 1 hint) or a division by zero. `np.isnan(arr).any()` finds them; then trace backwards to the first operation that produced one.
- **Everything in one cluster?** Your update step probably averages over the wrong axis — check that each centroid is the mean of *its* points, shape `(d,)`, not a mean over features.
- Standing advice: read the traceback bottom-up (the last line names the actual error), reproduce bugs on 10 points instead of 800,000, ask your AI assistant for a *hint* ("why might my inertia increase between iterations?") rather than the solution, and type every line yourself — muscle memory is the point.

## Resources

- **StatQuest: K-means clustering** (YouTube — search "StatQuest K-means clustering") — the assign/update loop drawn out step by step; watch before or during Project 1.
- **StatQuest: Principal Component Analysis (PCA), Step-by-Step** (YouTube — search for it) — the best visual explanation of variance directions there is; watch before Project 3.
- [scikit-learn clustering guide](https://scikit-learn.org/stable/modules/clustering.html) — the pro-grade map of clustering algorithms; skim after Project 1 to see what exists beyond k-means and how your inertia/elbow vocabulary matches theirs.
- [scikit-learn homepage](https://scikit-learn.org) — reference docs for `KMeans`, `PCA`, and `TSNE` when comparing against your implementations.

## Skills unlocked

- [ ] I can explain the difference between supervised and unsupervised learning in one sentence.
- [ ] I can implement the k-means assign/update loop in NumPy and explain why inertia never increases.
- [ ] I can choose a reasonable k using an elbow plot.
- [ ] I can turn any image into a dataset of RGB points and back, and use k-means as a compressor.
- [ ] I can implement PCA with centering, `np.cov`, and `np.linalg.eigh`, and explain what an eigenvector of the covariance matrix means.
- [ ] I can project high-dimensional data to 2D and interpret the resulting scatter plot honestly (including what "explained variance" tells me).
- [ ] I know when to reach for t-SNE/UMAP instead of PCA, and that I should use them, not fear them.

## Next up

Your models are only as good as the features you feed them — next you learn to craft those features and chain every preprocessing step into leak-proof pipelines: [19 · Feature Engineering and Pipelines](19-feature-engineering-pipelines.md).
