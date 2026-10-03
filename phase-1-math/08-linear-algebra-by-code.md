# 08 · Linear Algebra by Writing Your Own

**Phase 1 — Math** · Estimated time: 1-2 weeks · Prerequisites: [04 · Classes, Files, JSON and Errors](../phase-0-foundations/04-python-oop-files-errors.md), [05 · NumPy: Thinking in Arrays](../phase-0-foundations/05-numpy.md), [07 · Data Visualization](../phase-0-foundations/07-data-visualization.md)

> Linear algebra is the language every ML model speaks. A dataset is a matrix, a neural network layer is a matrix multiplication, and "how similar are these two things?" is a dot product. Most people meet this subject as a wall of symbols; this lesson presents it as code written by hand. It builds a tiny linear algebra library, uses matrices to spin and stretch pictures, and builds the exact similarity math that powers recommender systems and LLM embeddings later in this curriculum.

## What this lesson builds

- **minialg:** `Vector` and `Matrix` classes written from scratch in `minialg.py`, with add, scale, dot, matmul, transpose and inverse, plus a test file that proves every result matches NumPy.
- **Picture transformer:** a script that draws a shape, applies 2×2 rotation/scale/shear matrices to it, and plots before/after with matplotlib.
- **Similarity engine:** a mini "who is similar to whom" ranker, with movies as feature vectors, cosine similarity between every pair and a printed ranking.

## Concepts covered

- A **vector** is a list of numbers that can be pictured as an arrow, and also as one row of a dataset.
- The **dot product** measures how much two vectors point the same way: it is similarity, and it is projection.
- A **matrix** is a machine that transforms space: feed in a vector, get a moved vector out.
- **Matrix multiplication** chains transformations, and the shapes rule (m×n times n×p) is not arbitrary.
- The **identity matrix** does nothing; the **inverse** undoes what a matrix did.
- **Cosine similarity** compares direction while ignoring length. It is the core trick of embeddings and recommenders.
- **Eigen-intuition**: some directions survive a transformation unrotated. Just the picture, no formulas yet.
- Big connections: **dataset = matrix** (rows = examples, columns = features) and **neural network layer = matrix multiply**.

## Before starting

Check that these skills from earlier lessons are solid: writing a class with `__init__` and methods (lesson 04 built `Vector2D`, and this lesson extends that idea), creating NumPy arrays and doing elementwise arithmetic on them (lesson 05), and making a basic matplotlib plot (lesson 07).

```bash
cd ~/ml/ml-learn
source .venv/bin/activate
pip install numpy matplotlib
mkdir -p work/08-linear-algebra-by-code
cd work/08-linear-algebra-by-code
```

NumPy and matplotlib are probably already installed from phase 0, in which case pip just reports "Requirement already satisfied". No datasets are needed; the projects create all their own data.

**Companion videos:** watch 3Blue1Brown's *Essence of Linear Algebra* alongside the projects, not before them. Suggested pairing: episodes on vectors and the dot product with Project 1, episodes on linear transformations, matrix multiplication and inverses with Project 2, and the eigenvectors episode after Project 2's last milestone. Build first, then watch the episode. The video will then explain the ideas behind code that is already written.

## Project 1 — minialg: build Vector and Matrix from scratch

**Goal:** Implement the core operations of linear algebra by hand in `minialg.py`, verifying every single one against NumPy. The `@` sign also appears here for the first time: it is Python's matrix-multiplication operator (NumPy supports it on arrays), and this project implements it from scratch. By the end, nothing NumPy does with `dot` or `@` will be a mystery.

**Milestones**

- [ ] Create `minialg.py` with a `Vector` class that wraps a plain Python list of numbers: `v = Vector([1, 2, 3])`. Give it `__len__` and a `__repr__` so printing shows `Vector([1, 2, 3])`. Checkpoint: `print(Vector([1, 2, 3]))` shows exactly that.
- [ ] Add `__add__`, `__sub__`, and `scale(c)` (multiply every component by a number `c`). Adding vectors means adding component by component. Picture it as placing one arrow at the tip of the other. Checkpoint: `Vector([1, 2]) + Vector([3, 4])` gives `Vector([4, 6])`.
- [ ] Implement `dot(other)`: multiply matching components and sum the results. This one number shows how aligned two arrows are: big positive = same direction, zero = perpendicular, negative = opposite. Checkpoint: `Vector([1, 0]).dot(Vector([0, 1]))` is `0` (perpendicular arrows).
- [ ] Implement `norm()` (the vector's length) using only minialg's own `dot`: length is the square root of `v.dot(v)`. Checkpoint: `Vector([3, 4]).norm()` is `5.0` (a 3-4-5 triangle).
- [ ] Create a `Matrix` class that wraps a list of rows: `Matrix([[1, 2], [3, 4]])`. Add a `shape` property returning `(rows, cols)` and validate in `__init__` that all rows have equal length (raise `ValueError` if not). Checkpoint: `Matrix([[1, 2, 3], [4, 5, 6]]).shape` is `(2, 3)`.
- [ ] Add `transpose()`: flip rows into columns. Checkpoint: transposing a `(2, 3)` matrix gives shape `(3, 2)`, and transposing twice gives back the original.
- [ ] Implement matrix × vector: the output's component `i` is *row i dotted with the vector*. Reuse `Vector.dot`: matrix–vector multiplication is just a stack of dot products. Checkpoint: `Matrix([[1, 0], [0, 1]])` times any vector returns that vector unchanged.
- [ ] Implement `__matmul__` (the `@` operator) for matrix × matrix: entry `(i, j)` is row `i` of the left matrix dotted with column `j` of the right. Now the shapes rule explains itself. A dot product needs both vectors to be the same length, so the left matrix's row length (its columns, n) must equal the right matrix's column length (its rows, n): `(m, n) @ (n, p)` → `(m, p)`. Raise `ValueError` with a helpful message on mismatch. Checkpoint: `(2,3) @ (3,2)` gives shape `(2,2)`; `(2,3) @ (2,3)` raises that error.
- [ ] Add a class method `Matrix.identity(n)`: 1s on the diagonal, 0s elsewhere (the "do nothing" transformation). Checkpoint: `Matrix.identity(2) @ A` equals `A` for any 2×2 `A`.
- [ ] Add `inverse()` for 2×2 matrices using the closed formula: for `[[a, b], [c, d]]`, compute the determinant `det = a*d - b*c`, then the inverse is `(1/det) * [[d, -b], [-c, a]]`. The inverse is the "undo" matrix. If `det == 0`, the matrix squashes space flat and cannot be undone, so raise `ValueError`. The number of independent directions that survive a matrix is its **rank**: an invertible 2×2 matrix keeps both (rank 2), while `[[1, 2], [2, 4]]` squashes the whole plane onto one line (rank 1). A product `(m, r) @ (r, p)` never has a rank above `r`, so two thin matrices multiply into a big but *low-rank* matrix (`np.linalg.matrix_rank` counts it; lesson 36 builds LoRA fine-tuning on this idea). Checkpoint: `A @ A.inverse()` is (approximately) the identity.
- [ ] Write `test_minialg.py`: for each operation, build the same data as a NumPy array and `assert np.allclose(your_result_as_list, numpy_result)`. Use random matrices via `np.random.rand(3, 3).tolist()` (and `np.random.rand(2, 2).tolist()` for the 2×2 `inverse()`) so the tests are not limited to hand-picked easy cases. Checkpoint: `python test_minialg.py` prints `all tests passed`.

<details><summary>Hints</summary>

- Dunder methods make classes feel native: `__add__` powers `+`, `__matmul__` powers `@`, `__eq__` powers `==`. Lesson 04 used this pattern for `Vector2D`.
- A *property* is a method read like an attribute: put `@property` on the line directly above `def shape(self):`, and `m.shape` then works without parentheses, like NumPy's `arr.shape`. A *class method* is called on the class itself: put `@classmethod` above `def identity(cls, n):` and build the result with `cls(...)`. Written above a `def`, `@` marks a *decorator*, which has nothing to do with the `@` operator.
- `transpose` in one line: `list(zip(*self.rows))` flips rows and columns (then convert the tuples back to lists).
- Never compare floats with `==`: after arithmetic, `0.30000000000000004 != 0.3`. Use `math.isclose(a, b, abs_tol=1e-9)` for single numbers and `np.allclose` in tests. The `abs_tol` part matters whenever one side is `0`: with the defaults, `math.isclose(6e-17, 0)` is `False`, while `np.allclose` already includes a small absolute tolerance.
- For matmul, grabbing *column j* of the right matrix is the awkward part. Two options: index it as `[row[j] for row in other.rows]`, or transpose the right matrix first so its columns become easy-to-grab rows.

</details>

**Definition of done:** `test_minialg.py` passes with every operation (add, sub, scale, dot, norm, transpose, matvec, matmul, identity, 2×2 inverse) verified against NumPy, including on random inputs.

## Project 2 — Transform pictures with matrices

**Goal:** Make "a matrix transforms space" physically visible: build a shape out of points, multiply it by 2×2 matrices, and watch it rotate, stretch and lean on a matplotlib plot.

**Milestones**

- [ ] In `transform.py`, define a shape as a NumPy array of shape `(2, N)`: row 0 holds x-coordinates, row 1 holds y-coordinates. Use an asymmetric shape like the letter **F** or a house with a chimney (roughly 8-12 points), so rotations and flips are obvious. The picture is literally a matrix where each *column* is a point-vector.
- [ ] Plot it: `plt.plot(points[0], points[1], marker="o")`, plus `plt.axis("equal")` so squares look square, and `plt.grid(True)`. Checkpoint: the plot shows the shape, undistorted.
- [ ] Build a rotation matrix: `R = [[cos θ, -sin θ], [sin θ, cos θ]]` with `np.cos`/`np.sin` (angles are in **radians**, so use `np.radians(90)`). Transform every point at once with one multiplication: `new_points = R @ points`. That is the whole trick: one matmul moves the entire picture. Checkpoint: after rotating 90°, a point that was at `(1, 0)` is now at `(0, 1)`.
- [ ] Make a before/after figure with `fig, (ax1, ax2) = plt.subplots(1, 2)`: draw with `ax1.plot(...)` and `ax2.plot(...)`, and call `ax1.axis("equal")`, `ax2.axis("equal")`, `ax1.grid(True)` and `ax2.grid(True)` (a plain `plt.axis` or `plt.grid` call reaches only the right panel). Reuse this layout, with a fresh figure each time, for every experiment below. Checkpoint: original on the left, rotated shape on the right, both undistorted.
- [ ] Try a scaling matrix `[[2, 0], [0, 0.5]]` (twice as wide, half as tall) and a shear `[[1, 1], [0, 1]]` (leans the shape sideways like italic text). Checkpoint: the shear turns vertical lines into slanted lines but keeps horizontal lines horizontal.
- [ ] Compose transformations by multiplying matrices: `(R @ S) @ points` means "scale first, then rotate". The matrix nearest the points acts first. Plot `R @ S` versus `S @ R` applied to the shape. Checkpoint: the two results are visibly different. Order matters; matmul is not commutative.
- [ ] Full circle: starting from the original shape, apply a 15° rotation 24 times in a loop (24 × 15 = 360). Checkpoint: `np.allclose(result, original)` is `True`, since rotating by 360 degrees returns the original (within floating-point fuzz).
- [ ] Undo with the inverse: pick any transform `M` with nonzero determinant, compute `np.linalg.inv(M)` (and check it against the minialg 2×2 `inverse()`), then apply `M` followed by `M⁻¹`. Checkpoint: the shape comes back where it started: `np.allclose(result, original)` is `True` (equal within floating-point fuzz, as in the full-circle loop).
- [ ] **Eigen-intuition finale.** Make 100 unit vectors pointing in all directions (angles 0 to 2π; stack `cos` and `sin` into a `(2, 100)` array). Apply the shear `[[1, 1], [0, 1]]` and draw each input arrow and its output arrow. Most arrows get *turned*. Hunt for the direction that stays on its own line: the output points exactly the way the input went in (or exactly the opposite way), only maybe longer or shorter. That surviving direction is an **eigenvector** ("the direction the transform does not rotate"), and its stretch factor is the **eigenvalue** (negative when the arrow comes out flipped). Checkpoint: for this shear, arrows along the x-axis survive unrotated. That is all the eigen-theory needed for now; PCA in [lesson 18](../phase-2-classical-ml/18-unsupervised-kmeans-pca.md) builds on this picture.

<details><summary>Hints</summary>

- If the rotated shape looks wrong, check radians vs degrees first: `np.sin(90)` is *not* 1.
- If the shape looks distorted rather than rotated, `plt.axis("equal")` is probably missing; matplotlib stretches axes independently by default.
- Keep points as `(2, N)` with columns as points, so `M @ points` just works. If the points are stored as `(N, 2)`, either transpose or use `(M @ points.T).T`.
- Write every matrix as a NumPy array, for example `R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])` and `S = np.array([[2, 0], [0, 0.5]])`. `R @ points` happens to work with a plain-list `R` because `points` is an array, but `R @ S` with two plain lists raises `TypeError: unsupported operand type(s) for @: 'list' and 'list'`.
- For the arrows in the eigen milestone, plain `plt.plot` lines from the origin are the simplest choice. `plt.quiver(np.zeros(100), np.zeros(100), U[0], U[1], angles="xy", scale_units="xy", scale=1)`, with `U` the `(2, 100)` array of unit vectors (and a second call for the sheared ones), draws all arrows at once at their true length (without those three keyword arguments, quiver shrinks them to a blob), but it never widens the view to fit the arrow tips: set `plt.xlim(-2, 2)`, `plt.ylim(-2, 2)` and `plt.gca().set_aspect("equal")` by hand (`plt.axis("equal")` shrinks the view back to the origin).

</details>

**Definition of done:** One script produces before/after plots for rotation, scaling, shear and a composition; the 24×15° loop reproduces the original shape; and the eigenvector's direction can be pointed out on the shear plot.

## Project 3 — Who is similar to whom

**Goal:** Represent things as feature vectors and rank their similarity with cosine similarity. This is the identical mechanism behind recommender systems and LLM embeddings, just with human-readable vectors.

**Milestones**

- [ ] In `similarity.py`, invent features and score 6-8 movies (or songs) by hand in a dict, each value a `minialg.Vector` (from the library built in Project 1). Example features: `action, romance, comedy, scifi`, each 0.0-1.0. `"Die Hard": Vector([0.9, 0.1, 0.3, 0.0])`. Each movie is now a point in 4-dimensional space. It cannot be drawn, but every minialg operation works unchanged.
- [ ] Write `cosine_similarity(v, w)` using only minialg's `dot` and `norm`: `v.dot(w) / (v.norm() * w.norm())`. Dividing by the lengths keeps only *direction*, so it measures taste-profile, not intensity: a mild action movie and an extreme one still point the same way. The result ranges from 1 (same direction) through 0 (unrelated) to -1 (opposite). Checkpoint: for any nonzero vector `v`, `math.isclose(cosine_similarity(v, v), 1.0)` is `True`, and so is `math.isclose(cosine_similarity(v, v.scale(10)), 1.0)`. Printed directly, the value may show as `0.9999999999999998` or `1.0000000000000002`; that is floating-point rounding, not a bug. (A zero vector has no direction, and the division raises `ZeroDivisionError`.)
- [ ] Compute similarity for every pair and print a labeled table (nested loops over the dict; `f"{sim:.2f}"` keeps it readable). Checkpoint: the table is symmetric and its diagonal is all 1.00.
- [ ] For each movie, print its top-2 most similar others, sorted descending (exclude itself). Checkpoint: the ranking matches common sense. The action movies cluster together, and so do the romances. If not, look at the feature scores: the math is honestly reporting what was encoded, which is a first taste of "model quality depends on features".
- [ ] Now the industrial version. Stack all movies into one NumPy array `X` of shape `(n_movies, n_features)`. **This is what a dataset is**: rows = examples, columns = features; every CSV loaded in this curriculum becomes exactly this matrix. Divide each row by its norm, then compute `S = X @ X.T`. One matmul just computed *all* pairwise cosine similarities at once. Checkpoint: `S` matches the pairwise-loop table to within `np.allclose`.
- [ ] Add a comment block at the top of the file that paraphrases these points: a neural network layer computes `X @ W`, the same operation used in the previous milestone, with `W` learned instead of hand-written; and embeddings ([lesson 30](../phase-4-nlp-transformers/30-embeddings-word2vec.md)), RAG search ([lesson 35](../phase-5-llms/35-rag-chat-with-your-docs.md)) and recommenders ([lesson 40](../phase-6-special-topics/40-recommender-systems.md)) are this project with learned features and millions of rows.

<details><summary>Hints</summary>

- To get top-2, build a list of `(similarity, name)` tuples and use `sorted(pairs, reverse=True)` (the skills from lesson 03).
- Row-normalizing in NumPy: `X / np.linalg.norm(X, axis=1, keepdims=True)`. Without `keepdims=True`, the shapes will not line up for the division. Print them and see.
- If two movies score 1.00 but feel different, their feature *ratios* are identical or nearly so (the table rounds to two decimals, so anything from 0.995 up prints as 1.00). Add a feature that separates them.

</details>

**Definition of done:** The script prints a sensible top-2 ranking for every movie, and the one-matmul NumPy similarity matrix agrees with the minialg pairwise loop.

## Stretch goals

- **Translation via 3×3.** A 2×2 matrix can never *slide* a shape (the origin is stuck). Add a third "always 1" coordinate to each point and use 3×3 matrices (search for "homogeneous coordinates"). This is how all computer graphics works.
- **Transform a real photo.** Load an image as an array (`plt.imread`), treat pixel coordinates as vectors, and rotate the image by mapping each *output* pixel back through the inverse matrix to find its color.
- **General inverse.** Implement Gauss-Jordan elimination in minialg to invert any n×n matrix, verified against `np.linalg.inv`.
- **Matmul three ways.** Compute the same product as (a) rows dot columns, (b) a combination of the left matrix's columns and (c) a sum of outer products, then assert that all three agree. Each view will pay off later in deep learning.

## Getting unstuck

- **Shape errors are the #1 bug in all of ML**, so start practicing now: print `A.shape` and `B.shape` right before the failing line and check that the inner numbers match, as in `(m, n) @ (n, p)`. Say it out loud.
- **Wrong numbers, no error?** First test on inputs that are easy to verify mentally: identity matrices, `Vector([1, 0])`, 90° rotations. Only then trust random tests.
- Floating-point rounding (`0.9999999` instead of `1`) is normal; that is what `np.allclose` and `math.isclose` are for.
- Read tracebacks bottom-up: the last line names the error, the lines above show where.
- Stuck over 30 minutes? Ask an AI assistant for a **hint** (for example, "give me a hint about why my matmul gives the wrong shape, don't write the code"), never for the solution.
- Type every line by hand. No pasting. Typing the code is part of remembering it.

## Resources

- **3Blue1Brown — Essence of Linear Algebra**: search YouTube for "3Blue1Brown Essence of Linear Algebra". The series gives the best visual intuition for this subject. Pair its episodes with the projects as described above.
- **Immersive Linear Algebra** ([immersivemath.com/ila/index.html](https://immersivemath.com/ila/index.html)): a free interactive book with draggable figures, good for replaying dot products and transformations with the mouse.
- **NumPy documentation** ([numpy.org](https://numpy.org)): the reference for the `np.linalg` functions used to check the work in this lesson.

## Skills unlocked

- [ ] I can explain a vector both as an arrow and as a row of data.
- [ ] I can compute a dot product by hand and say what its sign and size mean.
- [ ] I implemented matrix multiplication myself and can explain why `(m, n) @ (n, p)` is the rule.
- [ ] I can predict what a 2×2 matrix does to a picture before running the code.
- [ ] I can explain what identity and inverse matrices do, and when an inverse does not exist.
- [ ] I can rank items by cosine similarity and explain why length is divided out.
- [ ] I can point at a transformed picture and identify the direction that survived (the eigenvector).
- [ ] I can say what "dataset = matrix" and "layer = matrix multiply" mean in one sentence each.

## Next up

Matrices move data; the next lesson covers what moves *matrices* (the derivative) and uses it to build gradient descent, the engine that trains linear models and every neural network in the rest of this curriculum: [09 · Calculus You Can Run: Gradient Descent](09-calculus-and-gradient-descent.md).
