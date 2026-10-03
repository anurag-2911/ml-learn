# 18 · Unsupervised Learning: k-Means and PCA from Scratch

**Phase 2 — Classical Machine Learning** · Estimated time: 1-2 weeks · Prerequisites: [05 · NumPy](../phase-0-foundations/05-numpy.md), [07 · Data Visualization](../phase-0-foundations/07-data-visualization.md), [08 · Linear Algebra by Code](../phase-1-math/08-linear-algebra-by-code.md), [11 · k-Nearest Neighbors](11-first-model-knn.md)

> Every model built in the course so far was told the right answer during training; that is **supervised learning**. This lesson removes the answers: in **unsupervised learning** the data has no labels, and the job is to find hidden structure without them, such as groups, patterns, and directions that matter. The lesson builds the two workhorses of unsupervised learning from scratch: k-means, which finds groups, and PCA, which finds directions. It then uses them to compress the colors of a real photo and to flatten 64-dimensional handwritten digits onto a 2D map where the digits sort themselves into islands. The photo posterizer is the most shareable project in the course so far, and it is worth posting.

## What this lesson builds

- **A k-means clusterer in pure NumPy**, with an animation (a series of plots) showing the centroids walking into the middle of each data blob, plus an elbow plot for choosing k.
- **A photo posterizer**: the from-scratch k-means, run on the pixels of a real photo to produce versions of the image painted with only 4, 8, and 16 colors. This is a real compression technique called color quantization.
- **A PCA implementation and a 2D map of handwritten digits**: center, covariance, eigenvectors, project. Then the same map made with t-SNE, a tool to use rather than build.

## Concepts covered

- **Supervised vs unsupervised**: learning with answer labels vs finding structure without them.
- **k-means**: repeat two steps. *Assign* each point to its nearest centroid, and *update* each centroid to the mean of its points.
- **Centroid**: the center point of a cluster (just the average of its members).
- **Inertia**: the total squared distance from every point to its assigned centroid; k-means tries to make this small.
- **Choosing k with the elbow method**: plot inertia for k = 1, 2, 3, ... and look for the bend.
- **k-means as compression**: replace millions of pixel colors with k representative colors.
- **PCA (Principal Component Analysis)**: find the directions along which the data varies the most.
- **Covariance matrix + eigenvectors**: the machinery behind PCA, using `np.linalg.eigh` (this is lesson 08's eigenvector intuition finally paying off).
- **Dimensionality reduction**: squashing many columns down to 2 so that the data can be *seen*.
- **t-SNE / UMAP**: fancier map-making tools, used as black boxes for now.

## Before starting

Check that these skills from earlier lessons are solid: computing the distances from one point to every row of a matrix with NumPy broadcasting (lesson 11), making scatter plots with colors (lesson 07), and explaining in one sentence what an eigenvector is (lesson 08: "a direction a matrix only stretches, never rotates").

Activate the venv and install what is needed (from the repo root):

```bash
source .venv/bin/activate
pip install numpy matplotlib scikit-learn pillow
```

Pillow is a library for opening image files as arrays of numbers.

Create the work folder for this lesson:

```bash
mkdir -p work/18-unsupervised-kmeans-pca
cd work/18-unsupervised-kmeans-pca
```

**Dataset downloads: none.** Project 1 generates fake data, Project 2 uses any photo already on hand, and Project 3's digits ship inside scikit-learn. Put a personal photo in the work folder as `photo.jpg`. To drag one in, open the folder in the file manager: `open .` on macOS, `explorer.exe .` on Windows (WSL2), `xdg-open .` on Linux. Or copy it in the terminal: on macOS and Linux, `cp ~/Pictures/YOUR-PHOTO.jpg photo.jpg`; on Windows (WSL2), Windows files are under `/mnt/c/Users/` (`ls /mnt/c/Users/` lists the Windows account names), so `cp /mnt/c/Users/YOUR-WINDOWS-NAME/Pictures/YOUR-PHOTO.jpg photo.jpg`. When OneDrive backs up the Pictures folder, the photos are in `/mnt/c/Users/YOUR-WINDOWS-NAME/OneDrive/Pictures/` instead.

Pick a colorful photo (a sunset, a market, a parrot). The effect is much more dramatic than on a grey screenshot.

The fork on GitHub is public, and a phone photo often records where it was taken (GPS data inside the file). Keep the original out of git: in this folder, run `printf 'photo.jpg\n' > .gitignore`. Checkpoint: `git check-ignore photo.jpg` prints `photo.jpg`.

## Project 1 — k-means from scratch

**Goal:** Implement the k-means assign/update loop in NumPy and *watch* it converge by plotting every iteration on 2D blob data.

**Milestones**

- [ ] Make practice data with a known answer: `from sklearn.datasets import make_blobs` then `X, y_true = make_blobs(n_samples=500, centers=4, cluster_std=0.9, random_state=42)`. `X` is 500 points in 2D. Ignore `y_true` while clustering; pretending the labels do not exist is the whole point. Scatter-plot `X`. Checkpoint: the plot shows 4 distinct blobs.
- [ ] Write `init_centroids(X, k, rng)`: pick k *distinct* random rows of X as the starting centroids. Use `rng.choice(len(X), size=k, replace=False)`. Starting from real data points guarantees no centroid begins in empty space.
- [ ] Write `assign(X, centroids)`: for every point, find the index of its nearest centroid. This is the same distance computation written for kNN in lesson 11. Squared Euclidean distance is fine (skip the square root; it does not change which centroid is nearest). Return an array of 500 cluster indices. Checkpoint: `assign(X, centroids).shape == (500,)` and the values are all in `0..k-1`.
- [ ] Write `update(X, labels, k)`: the new centroid j is `X[labels == j].mean(axis=0)`. Handle the edge case where a cluster has zero points (see Hints).
- [ ] Write `inertia(X, labels, centroids)`: sum of squared distances from each point to *its own* centroid.
- [ ] Wire the loop: init, then repeat assign → update, saving a scatter plot each iteration (points colored by cluster label, centroids drawn as big black X markers) to `frames/iter_00.png`, `iter_01.png`, ... (create the folder first: `import os` then `os.makedirs("frames", exist_ok=True)`, because `savefig` does not create folders). Stop when the labels stop changing (compare with `np.array_equal`). Checkpoint: it converges in under ~15 iterations, the printed inertia goes *down every single iteration*, and in the final frame the black X's sit in the middle of each blob.
- [ ] Flip through the frames in order (open the folder in VS Code and arrow-key through them). They show an optimization algorithm thinking. Optional: stitch them into a GIF (see Hints).
- [ ] Run k-means with a *bad* seed a few times (change `rng`). Notice that the final inertia differs from run to run: k-means only finds a *local* best. Fix: run it 5 times and keep the run with the lowest inertia (this is what sklearn's `n_init` does).
- [ ] **Elbow method:** loop k from 1 to 10, run the best-of-5 k-means from the previous milestone for each k, and plot k vs the lowest final inertia (a single run per k can get stuck in a local best and put false bumps and bends in the curve). Checkpoint: the curve drops steeply, then flattens, with a visible bend (the "elbow") at k=4. That is the true number of blobs, discovered without ever peeking at `y_true`.

<details><summary>Hints</summary>

- Distances without loops: `d = ((X[:, None, :] - centroids[None, :, :])**2).sum(axis=2)` gives a `(n_points, k)` matrix; then `labels = d.argmin(axis=1)` and inertia is `d[np.arange(len(X)), labels].sum()`.
- Empty cluster: if `labels == j` matches nothing, `.mean(axis=0)` returns NaN and poisons everything after. When it happens, re-seed that centroid to a random data point instead.
- Use `rng = np.random.default_rng(22)` and pass it around, so runs are reproducible during debugging. If the final frame shows two X's inside one blob and another X stuck between blobs, the code is probably fine: k-means has settled in a *local* best, which a later milestone explains (seed 0 does this on this data). Try another seed.
- GIF in three lines with Pillow: open the frame PNGs with `Image.open`, then `frames[0].save("kmeans.gif", save_all=True, append_images=frames[1:], duration=400, loop=0)`.

</details>

**Definition of done:** The k-means converges on the blobs with centroids centered in each blob, inertia decreases monotonically, and the elbow plot points at k=4.

## Project 2 — Posterize a photo

**Goal:** Run the k-means *built in Project 1* on the pixels of a real photo, where each pixel is just a point in 3D (R, G, B) space, and repaint the photo using only the k centroid colors. The result is a color-quantization compressor.

**Milestones**

- [ ] Load the photo as a NumPy array:

  ```python
  from PIL import Image, ImageOps
  import numpy as np
  img = np.asarray(ImageOps.exif_transpose(Image.open("photo.jpg")).convert("RGB"))
  print(img.shape, img.dtype)   # (height, width, 3) uint8
  ```

  Checkpoint: shape is `(H, W, 3)` and values run 0-255.
- [ ] Reshape it into a dataset: `pixels = img.reshape(-1, 3).astype(np.float64)`. A 1000×800 photo is now 800,000 "points" in 3D: the same shape of problem as the blobs, just with more rows and one more column. The Project 1 code should run on it *unchanged*.
- [ ] Speed trick: fitting on 800k points is slow, and unnecessary. Fit the centroids on a random sample of ~10,000 pixels (`rng.choice`), then do one final `assign` pass over *all* pixels using those centroids.
- [ ] Rebuild the image: `new_pixels = centroids[labels]` (fancy indexing; that one line does a lot of work), then reshape back to `(H, W, 3)`, convert with `.clip(0,255).astype(np.uint8)`, and save with `Image.fromarray(...).save("posterized_k8.png")`.
- [ ] Do it for k = 4, 8, 16. Checkpoint: k=4 looks like a bold screen-printed poster, k=8 looks stylized, and k=16 looks almost like the original, except when zoomed in on a sky or a skin-tone gradient.
- [ ] Make one comparison figure with matplotlib: original plus the three posterized versions in a 2×2 grid (`plt.subplots(2, 2)`, `ax.imshow(...)`, title each with its k). Save it as `comparison.png`.
- [ ] Appreciate the compression: the original stores 24 bits per pixel; with k=16, only 4 bits per pixel are needed, plus a tiny 16-color palette. That is 6× smaller. Print those numbers for the photo.
- [ ] **Share it.** This is the most post-worthy artifact of the course so far, and the 2×2 grid works well as an "I built this from scratch" post. Commit the posterized images and `comparison.png` (the original `photo.jpg` stays ignored). They show the photo's content, so commit them only if the photo is fine to make public.

<details><summary>Hints</summary>

- If assigning all 800k pixels at once uses up all the RAM (the broadcasted distance matrix is `n_pixels × k × 3`), assign in chunks of ~100k pixels with a plain Python loop over slices.
- Ugly muddy colors usually mean that k-means was fit on uint8 data: uint8 arithmetic wraps around at 256 (10 - 20 gives 246, not -10), so the distances are wrong and pixels go to the wrong centroid. Cast to float *before* clustering, and back to uint8 only at save time.
- If the output looks like static noise, `centroids[labels]` was probably indexed with labels from a *different* run than the centroids. Fit once, keep both together.

</details>

**Definition of done:** Three posterized images plus a labeled 2×2 comparison grid, all produced by the *from-scratch* k-means and saved in the work folder.

## Project 3 — PCA from scratch

**Goal:** Implement PCA in four lines of linear algebra and use it to project 64-dimensional handwritten digits onto 2D, where the ten digits separate into visible clusters: structure found without labels.

**Milestones**

- [ ] Load the data: `from sklearn.datasets import load_digits`, then `digits = load_digits()` and `X = digits.data`. `digits.data` is `(1797, 64)`: each row is an 8×8 grayscale digit image flattened into 64 numbers. Plot a few with `plt.imshow(digits.data[i].reshape(8, 8), cmap="gray")` to meet the data. Checkpoint: the digits are recognizable (they are tiny, so squint).
- [ ] A scatter plot cannot show 64 dimensions. PCA's promise: find the 2 directions in 64D space along which the data spreads out the most, and shadow everything onto them. "Direction of maximal variance" = the line along which the cloud of points is most stretched out.
- [ ] Step 1 (**center**): `Xc = X - X.mean(axis=0)`. PCA measures spread around the middle, so the middle must be at zero.
- [ ] Step 2 (**covariance matrix**): `C = np.cov(Xc, rowvar=False)`, a 64×64 table where entry (i, j) says how features i and j vary together. Checkpoint: `C.shape == (64, 64)` and `np.allclose(C, C.T)` is True (it is symmetric).
- [ ] Step 3 (**eigendecomposition**): `eigvals, eigvecs = np.linalg.eigh(C)` (`eigh` is the version for symmetric matrices; it is faster and returns real numbers). Lesson 08 payoff: the eigenvectors of the covariance matrix *are* the directions of maximal variance, and each eigenvalue is the amount of variance along its eigenvector. Careful: `eigh` sorts eigenvalues *ascending*, so the best directions are the **last** columns.
- [ ] Step 4 (**project**): take the top-2 eigenvectors as a `(64, 2)` matrix `W`, then `X2 = Xc @ W`. Sixty-four numbers per digit have become two. Checkpoint: `X2.shape == (1797, 2)`.
- [ ] Scatter-plot `X2` colored by the true digit: `plt.scatter(X2[:,0], X2[:,1], c=digits.target, cmap="tab10", s=8)` plus `plt.colorbar()`. Note that the labels are used only to *color the picture*, never to compute the projection. Checkpoint: clear islands (0s clumped together far from 1s, 6s in their own region). Some digits smear into each other (5 and 8 overlap heavily in the middle, and 3 blends into 2 and 9); that is honest, not a bug.
- [ ] Print the **explained variance**: `eigvals[-2:].sum() / eigvals.sum()`. Checkpoint: roughly 0.29. Two dimensions carry between a quarter and a third of the total variance, yet the map is already readable.
- [ ] Sanity-check against the professionals: `from sklearn.decomposition import PCA; X2_sk = PCA(n_components=2).fit_transform(X)`. Checkpoint: sklearn's scatter matches the from-scratch one up to a possible mirror flip of either axis (an eigenvector times −1 is an equally valid eigenvector).
- [ ] Now try **t-SNE**, a nonlinear map-maker designed purely for pretty 2D visualization: `from sklearn.manifold import TSNE; X2_t = TSNE(n_components=2, random_state=0).fit_transform(X)` (takes ~a minute). Plot it the same way. Checkpoint: ten tight, well-separated islands, noticeably crisper than PCA. **UMAP** (`pip install umap-learn`) is a faster modern alternative; try it if curious (not on an Intel Mac: numba, a library it needs, has no current Intel Mac version; use Google Colab, which has UMAP preinstalled). Use these as tools; this lesson does not implement them.
- [ ] One-sentence journal entry: when to reach for PCA (fast, linear, keeps global structure, reusable as a preprocessing step) vs t-SNE/UMAP (slower, for visualization only). This exact trick is used again in [lesson 30](../phase-4-nlp-transformers/30-embeddings-word2vec.md) to draw maps of *word meanings*.

<details><summary>Hints</summary>

- Wrong-looking blob with no structure? Most common cause: forgetting to center, or projecting the *uncentered* X onto W. Center once, use `Xc` everywhere.
- Grabbing the top eigenvectors: `W = eigvecs[:, -2:][:, ::-1]` takes the last two columns and puts the biggest first. Columns are eigenvectors, not rows.
- Is the plot mirrored compared with sklearn's? Sign flips are expected. A *rotated or squashed* plot is not: it means the eigenvectors were taken from the small end.

</details>

**Definition of done:** The from-scratch 2D digits map shows visibly separated digit clusters and matches sklearn PCA up to sign flips, with a t-SNE map produced alongside it.

## Stretch goals

- **k-means++ initialization:** instead of uniformly random starting centroids, pick each new one with probability proportional to its squared distance from the nearest already-chosen centroid. Implement it and compare final inertia over 10 runs vs plain random init.
- **PCA as compression:** project the digits onto the top-k components and reconstruct (`Xc @ W @ W.T + mean`). Show the same digit reconstructed from 2, 8, 16, 32 components, and watch it sharpen.
- **Cluster the digits:** run the from-scratch k-means with k=10 directly on the 64D digits data (no labels), then check how well clusters align with true digits. Show each cluster's centroid as an 8×8 image; several look like ghostly average digits.
- **Posterizer CLI:** wrap Project 2 in a script (`python posterize.py photo.jpg 8`) using `argparse` (Python's built-in library for reading command-line options), so friends can run it.

## Getting unstuck

- **Shape bugs dominate this lesson.** Print `.shape` at every step of the distance broadcast; the `(n, k)` distance matrix, `(n,)` labels, and `(k, d)` centroids must line up or `argmin`/indexing silently does the wrong thing on the wrong axis.
- **NaNs appearing?** Almost always an empty cluster (Project 1 hint) or a division by zero. `np.isnan(arr).any()` finds them; then trace backwards to the first operation that produced one.
- **Everything in one cluster?** The update step probably averages over the wrong axis. Check that each centroid is the mean of *its* points, shape `(d,)`, not a mean over features.
- Standing advice: read the traceback bottom-up (the last line names the actual error), reproduce bugs on 10 points instead of 800,000, ask an AI assistant for a *hint* ("why might my inertia increase between iterations?") rather than the solution, and type every line by hand, since muscle memory is the point.

## Resources

- **StatQuest: K-means clustering** (YouTube: search "StatQuest K-means clustering") — the assign/update loop drawn out step by step; watch before or during Project 1.
- **StatQuest: Principal Component Analysis (PCA), Step-by-Step** (YouTube: search for it) — the best visual explanation of variance directions there is; watch before Project 3.
- [scikit-learn clustering guide](https://scikit-learn.org/stable/modules/clustering.html) — the pro-grade map of clustering algorithms; skim after Project 1 to see what exists beyond k-means and how the inertia of this lesson matches their definition (they call it the within-cluster sum-of-squares).
- [scikit-learn homepage](https://scikit-learn.org) — reference docs for `KMeans`, `PCA`, and `TSNE` when comparing against the from-scratch implementations.

## Skills unlocked

- [ ] I can explain the difference between supervised and unsupervised learning in one sentence.
- [ ] I can implement the k-means assign/update loop in NumPy and explain why inertia never increases.
- [ ] I can choose a reasonable k using an elbow plot.
- [ ] I can turn any image into a dataset of RGB points and back, and use k-means as a compressor.
- [ ] I can implement PCA with centering, `np.cov`, and `np.linalg.eigh`, and explain what an eigenvector of the covariance matrix means.
- [ ] I can project high-dimensional data to 2D and interpret the resulting scatter plot honestly (including what "explained variance" tells me).
- [ ] I know when to reach for t-SNE/UMAP instead of PCA, and that I should use them, not fear them.

## Next up

A model is only as good as the features fed into it, so the next lesson covers crafting those features and chaining every preprocessing step into leak-proof pipelines: [19 · Feature Engineering and Pipelines](19-feature-engineering-pipelines.md).
