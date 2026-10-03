# 05 · NumPy: Thinking in Arrays

**Phase 0 — Foundations** · Estimated time: 1 week · Prerequisites: [02 · Python Basics](02-python-basics.md), [03 · Data Structures](03-python-data-structures.md), [04 · Classes, Files, JSON and Errors](04-python-oop-files-errors.md)

> Every piece of ML you will ever build — every image, every dataset, every neural network weight — is stored and manipulated as a NumPy-style array. This week you make the single most important mental shift in this whole curriculum: you stop writing loops that touch one number at a time and start writing operations that transform millions of numbers at once. To prove it is not just a style preference, you will edit a real photo with pure math, win a 50–100x speed race against your own loop code, and build Conway's Game of Life on a 2D grid. When arrays feel natural, everything after this lesson gets easier.

## What you will build

- **Project 1 — Images are just arrays:** a photo-editing script that grayscales, flips, crops, brightens and inverts a real photo using only array operations, saving each result as an image file you can open and admire.
- **Project 2 — Speed race:** a benchmark script that computes the same sum-of-squares on 10 million numbers with a Python loop and with NumPy, and prints a timed scoreboard showing NumPy winning by 50–100x.
- **Project 3 — Conway's Game of Life:** a self-running 2D world where cells live and die by simple rules, updated entirely with array slicing — no per-cell loops — and animated in your terminal or with matplotlib.

## Concepts you will learn by doing

- **ndarray** — NumPy's core object: a grid of numbers, all the same type, from 1D lists to 2D tables to 3D image stacks.
- **shape and dtype** — the size of each dimension, and the number type stored inside (e.g. `uint8`, `float64`).
- **indexing and slicing in 2D** — grabbing rows, columns, and rectangular chunks with `arr[rows, cols]`.
- **elementwise operations** — `a + b`, `a * 2`, `a > 5` applied to every element at once, no loop written by you.
- **broadcasting** — NumPy's rules for combining arrays of different shapes (the trickiest and most useful idea this week).
- **aggregations and axis** — `sum`, `mean`, `max`, `argmax`, and what `axis=0` vs `axis=1` actually means.
- **boolean masking** — using an array of True/False to select or modify only the elements you care about.
- **np.random** — generating random arrays for simulations and, later, for initializing neural networks.
- **vectorization** — why NumPy is 50–100x faster than Python loops, and how to rewrite loop-thinking as array-thinking.

## Before you start

Check your foundation. You should be comfortable with lists, `for` loops, functions, and running a script from the terminal (all from lessons 02–04). If not, go back — this lesson leans on them hard.

Activate your venv (the isolated Python environment you made in lesson 01) and install this week's packages:

```bash
cd ~/ml/ml-learn
source .venv/bin/activate
pip install numpy pillow matplotlib
```

- **NumPy** is the array library. **Pillow** loads and saves image files. **matplotlib** draws plots (a preview of lesson 07 — here we only use it to display grids).

Create your work folder:

```bash
mkdir -p work/05-numpy
cd work/05-numpy
```

You also need one photo for Project 1 — any JPEG of yours works (a pet, a meal, a holiday shot; ideally at least 500×500 pixels). Put it in this folder as `photo.jpg`. To drag one in, open the folder in your file manager: `open .` on macOS, `explorer.exe .` on Windows (WSL2), `xdg-open .` on Linux. Or copy it in the terminal: on macOS and Linux, `cp ~/Pictures/YOUR-PHOTO.jpg photo.jpg`; on Windows (WSL2) your Windows files are under `/mnt/c/Users/`, so `cp /mnt/c/Users/YOUR-WINDOWS-NAME/Pictures/YOUR-PHOTO.jpg photo.jpg`.

Finally, spend 30–60 minutes skimming the [NumPy absolute beginners guide](https://numpy.org/doc/stable/user/absolute_beginners.html) with a Python prompt open, typing every example. Skim now, refer back all week.

## Project 1 — Images are just arrays

**Goal:** Discover that a photo is literally a 3D array of numbers — height × width × 3 color channels — and edit it with nothing but array math. No loops allowed anywhere in this project.

**Milestones**

- [ ] Create `images.py`. Load your photo and convert it to an array:

  ```python
  from PIL import Image
  import numpy as np

  img = np.array(Image.open("photo.jpg"))
  print(img.shape, img.dtype)
  ```

  **Checkpoint:** you should see something like `(1024, 768, 3) uint8` — height, width, and 3 values (red, green, blue) per pixel, each an integer from 0 to 255. `uint8` means "unsigned 8-bit integer": a whole number from 0 to 255.
- [ ] Poke at single pixels: print `img[0, 0]` (top-left pixel, 3 numbers) and `img[0, 0, 0]` (just its red value). Print `img[:5, :5, 0]` — the red channel of the top-left 5×5 corner. This is 2D/3D indexing: one index (or slice) per dimension, separated by commas.
- [ ] Write a helper `save(arr, name)` that does `Image.fromarray(arr).save(name)` so you can eyeball every result. You will call it after each step. Open the saved files in VS Code's file explorer.
- [ ] **Crop:** slicing a 2D array works like slicing a list, once per axis: `img[100:400, 200:500]` keeps rows 100–399 and columns 200–499. Crop an interesting region of your photo and save it. **Checkpoint:** the saved image is the region you aimed for (expect a few tries — rows come first, columns second, which surprises everyone).
- [ ] **Flip:** a slice step of `-1` reverses an axis. Flip your photo upside down, then mirror it left-right, using only slicing. Save both. **Checkpoint:** the mirrored image reads like your photo in a mirror.
- [ ] **Invert:** `255 - img` subtracts every one of the millions of values from 255 in one line — that is an elementwise operation. Save it. **Checkpoint:** it looks like an old film negative.
- [ ] **Brighten:** naively `img + 80` overflows — 250 + 80 wraps around to 74 in `uint8`, causing weird dark speckles in bright areas. The fix is a dtype round-trip: convert to a bigger type, add, cap at 255 with `np.clip`, convert back:

  ```python
  bright = np.clip(img.astype(np.int64) + 80, 0, 255).astype(np.uint8)
  ```

  Save it. **Checkpoint:** brighter photo, no weird speckles. You just learned why dtype matters.
- [ ] **Grayscale, and meet broadcasting.** A grayscale pixel is a weighted mix of red, green and blue: `0.299*R + 0.587*G + 0.114*B` (the classic luminance formula). Compute it for all pixels at once by multiplying the whole image by a 3-number array of weights, then summing away the color axis. Study this comment until it clicks:

  ```python
  # BROADCASTING: how NumPy combines arrays of different shapes.
  # It lines the shapes up from the RIGHT, and two dimensions are
  # compatible if they are equal, or one of them is 1 (or missing).
  #
  #   img:              (1024, 768, 3)
  #   weights:                     (3,)
  #                     ---------------
  #   result:           (1024, 768, 3)
  #
  # The (3,) array is treated as if copied out to (1024, 768, 3):
  # every pixel's [R, G, B] gets multiplied by [0.299, 0.587, 0.114].
  # No copy actually happens - NumPy just pretends. That's broadcasting.
  #
  # Then .sum(axis=-1) adds up the last axis (the 3 color values),
  # collapsing (1024, 768, 3) down to (1024, 768): one gray value per pixel.
  ```

  Convert back to `uint8` before saving. **Checkpoint:** a proper black-and-white version of your photo (a smaller shape too — print it).
- [ ] **Boolean masking finale:** `img.sum(axis=-1) > 400` gives a `(height, width)` array of True/False — a **mask** marking bright pixels. Use it to index the image (`img[mask] = ...`) and turn all bright pixels pure red. Save it. **Checkpoint:** bright regions of your photo (sky, lights, white walls) are now red.

<details><summary>Hints</summary>

- Crop confusion: `img[a:b, c:d]` means rows a–b (vertical) then columns c–d (horizontal). If your crop looks transposed, you swapped them.
- `Image.fromarray` wants `uint8`. If saving throws a type error or produces garbage, print `arr.dtype` — after any math involving floats you must `.astype(np.uint8)` (clip first).
- For grayscale, `img * np.array([0.299, 0.587, 0.114])` broadcasts exactly as the comment shows; then sum along the last axis.
- Masking a color image: a `(h, w)` boolean mask used on a `(h, w, 3)` image selects whole pixels, so `img[mask] = [255, 0, 0]` sets each selected pixel to red — broadcasting again.

</details>

**Definition of done:** `images.py` saves cropped, flipped, mirrored, inverted, brightened, grayscale and red-masked versions of your photo, contains zero `for` loops, and you can explain the broadcasting comment out loud.

## Project 2 — Speed race

**Goal:** Prove with a stopwatch that vectorized NumPy demolishes Python loops, and understand why. This number — 50–100x — is the reason all of ML is built on arrays.

**Milestones**

- [ ] Create `speed_race.py`. Make the contestants' data: `data_list = list(range(10_000_000))` and `data_array = np.arange(10_000_000)` (`np.arange` is NumPy's `range` that returns an ndarray directly).
- [ ] Write `sum_of_squares_loop(numbers)` the way lesson-02 you would: a `for` loop accumulating `total += x * x`. Pure Python, no NumPy inside.
- [ ] Write `sum_of_squares_numpy(arr)` as a single expression: square the array elementwise, then `.sum()`. **Checkpoint:** both functions return the same value on `range(100)` / `np.arange(100)`: 328350.
- [ ] Time both on the 10-million-element data with `time.perf_counter()` (a high-precision clock from Python's built-in `time` module):

  ```python
  import time
  start = time.perf_counter()
  result = sum_of_squares_loop(data_list)
  elapsed = time.perf_counter() - start
  ```

  Print each contestant's time and the ratio. **Checkpoint:** the loop takes on the order of a second; NumPy takes milliseconds; the printed ratio is roughly 50–100x (anywhere from 30x to 300x is normal depending on your machine).
- [ ] One gotcha, and it is a good one: the two totals do NOT match. Print both and look. The true sum of squares here is about 3.3 × 10²⁰, but `np.arange` gives an `int64` array (check `data_array.dtype`), and `int64` maxes out near 9.2 × 10¹⁸ — so NumPy's total silently wraps around, exactly like the `uint8` speckles in Project 1, just at 64-bit scale. The Python loop is immune: plain Python ints grow as big as they need. Fix the NumPy side by computing in floats — `(data_array.astype(np.float64) ** 2).sum()` — and compare it to `float(loop_total)`. **Checkpoint:** the two float totals agree to about 15 significant digits (a difference in the last digit or two is ordinary float rounding, not overflow).
- [ ] Add a third contestant: a Python loop *over a NumPy array* (`for x in data_array: ...`). **Checkpoint:** it is the *slowest* of the three — looping in Python throws away all of NumPy's speed. Moral: never loop over an array's elements.
- [ ] In a comment at the bottom, explain the result in your own words. The short version: your Python loop makes the interpreter examine one number at a time, with type-checking overhead per element; `(arr ** 2).sum()` hands the whole array to compiled C code that rips through memory in bulk. Replacing per-element Python loops with whole-array operations is called **vectorization**.

<details><summary>Hints</summary>

- Run several timings back to back (or wrap the timing in a small helper that repeats 3 times and takes the minimum) — the first run can be slower while things warm up.
- If the loop version takes so long you get bored, drop to 1 million elements, get everything working, then scale back up.
- Ratio wildly off? Make sure you are not accidentally timing the creation of the data, only the computation.

</details>

**Definition of done:** `speed_race.py` prints three labeled timings and the NumPy-vs-loop ratio; the ratio is at least 30x; your closing comment correctly names vectorization as the reason.

## Project 3 — Conway's Game of Life

**Goal:** Build a famous zero-player game: a 2D grid of cells where each generation, cells live or die based on how many of their 8 neighbors are alive. The catch that makes it a NumPy boss fight: compute every cell's neighbor count *simultaneously* with array operations — no per-cell loops.

The rules (from mathematician John Conway, 1970): a live cell with 2 or 3 live neighbors survives; a dead cell with exactly 3 live neighbors becomes alive; every other cell is dead next generation.

**Milestones**

- [ ] Create `life.py`. Make a random starting world: a 30×40 grid of 0s (dead) and 1s (alive), about 25% alive. `np.random.random((30, 40)) < 0.25` gives booleans; `.astype(np.uint8)` makes them 0/1. `np.random` is NumPy's random-array toolbox — you will meet it again initializing neural networks. **Checkpoint:** `grid.shape` is `(30, 40)` and `grid.mean()` is near 0.25 (why does the mean of 0s and 1s equal the fraction alive? Think about it — this trick appears constantly in ML).
- [ ] Print the world readably: a function that prints each row as characters, e.g. `'█'` for alive, `'·'` for dead. Loops are allowed for *printing* — the no-loop rule applies to the update math.
- [ ] The core: `count_neighbors(grid)` returns a same-shaped array where each entry is that cell's number of live neighbors. The vectorized idea: shifting the whole grid one step in each of the 8 directions and summing the shifts counts every cell's neighbors at once. Pad the grid with a border of zeros first (`np.pad(grid, 1)`) so edge cells simply see dead space outside; then each direction is a `(30, 40)` slice of the padded `(32, 42)` array. **Checkpoint:** on a tiny 4×4 test grid with one live cell in the middle, the count array shows 1 in the 8 surrounding cells and 0 in that cell itself.
- [ ] Write `step(grid)` that applies Conway's rules using only comparisons and boolean logic on whole arrays — combine masks with `&` (and) and `|` (or), e.g. survivors are `(grid == 1) & ((n == 2) | (n == 3))`. Note NumPy needs `&`/`|` with parentheses, not Python's `and`/`or`. Return the new grid; never modify the old one mid-computation (every cell's fate depends on the *old* neighbor counts).
- [ ] Test with a known pattern: a **blinker** — three live cells in a horizontal row on an otherwise dead grid. **Checkpoint:** after one step it becomes a vertical row of three, after two steps horizontal again, forever. If this works, your rules are provably correct.
- [ ] Run it: loop 50 generations, printing each and pausing with `time.sleep(0.1)`. Clearing the terminal between frames (`print("\033[2J\033[H", end="")`) makes it a real animation. **Checkpoint:** the world visibly evolves — flickering chaos settles into stable blocks, oscillators, and maybe a glider crawling diagonally across the grid.
- [ ] Seed a **glider** by hand (search "game of life glider" for the 5-cell pattern) on an empty grid. **Checkpoint:** it walks one cell diagonally every 4 generations.

<details><summary>Hints</summary>

- The padded-slice trick, half-revealed: with `p = np.pad(grid, 1)`, the "north-west neighbor" of every cell is `p[:-2, :-2]`, and `p[1:-1, 1:-1]` is the grid itself. Work out the other 7 slices on paper by drawing a 4×4 grid and its 6×6 padded version. Sum the 8 shifted slices, not the center.
- Wrong neighbor counts? Print shapes first — every one of the 8 slices must be exactly `grid.shape`. Then test on the single-live-cell 4×4 grid where you can check by hand.
- Blinker stuck or exploding? Classic cause: updating the grid in place while still reading it. Build the new grid entirely from the old one.
- Prefer graphics over terminal art? `matplotlib.pyplot.imshow(grid)` draws the grid as an image; call `plt.pause(0.1)` between generations. On macOS a window opens straight away; on Windows (WSL2) and Linux none opens until you install tkinter, the toolkit matplotlib uses for windows there: `sudo apt install -y python3.13-tk`.

</details>

**Definition of done:** `life.py` animates 50 generations from a random seed with a loop-free `step`, the blinker oscillates with period 2, and a glider travels.

## Stretch goals

- **Sepia filter:** in Project 1, multiply the image by a 3×3 color-mixing matrix (search "sepia matrix") for that old-photograph look — a preview of the matrix multiplication you will build in [lesson 08](../phase-1-math/08-linear-algebra-by-code.md).
- **Wrap-around world:** change Life's edges so gliders exiting right re-enter left (look at `np.roll` — it makes the neighbor counting even cleaner than padding).
- **Race extension:** add contestants to Project 2 — Python's built-in `sum` with a generator expression, and NumPy's `np.dot(arr, arr)`. Predict the ranking before running.
- **Axis drills:** make a random `(5, 3)` array of "exam scores" (5 students, 3 subjects); compute each student's mean, each subject's mean, and each subject's best student (`argmax`, which returns the *index* of the maximum) — choosing the right `axis` each time without guessing.

## If you get stuck

- **Print `.shape` and `.dtype` before anything else.** Ninety percent of NumPy bugs are one of those two. Make it a reflex now — it stays your #1 debugging move through the deep-learning phases.
- **Shape mismatch errors** ("operands could not be broadcast together with shapes ...") are telling you the two shapes, aligned from the right, have a dimension that is neither equal nor 1. Write the shapes one above the other, right-aligned, like the Project 1 comment.
- **The axis rule, once and for all:** `axis=N` means *axis N is the one that gets collapsed*. For a 2D array, `axis=0` collapses the rows, leaving one result **per column**; `axis=1` collapses the columns, leaving one result **per row**. Memorable version: **the axis you name is the axis that disappears.** Check: `(30, 40)` summed with `axis=0` → shape `(40,)`.
- **Shrink the problem.** Debug every 2D operation on a 4×4 array you can verify by hand before running it on a photo or a 10-million-element benchmark.
- Standing advice: read the error message bottom-up (the last line names the problem, the lines above say where); print shapes and values rather than staring; ask an AI assistant for a **hint** and never a solution; and type all code yourself — no pasting.

## Resources

- [NumPy absolute beginners guide](https://numpy.org/doc/stable/user/absolute_beginners.html) — the one tutorial to actually work through; covers every concept in this lesson.
- [NumPy learn page](https://numpy.org/learn/) — curated books, videos and next-step tutorials when you want more depth or a different angle.

## Skills unlocked

- [ ] I can create arrays (`np.array`, `np.arange`, `np.zeros`, `np.random`) and read their `shape` and `dtype` at a glance.
- [ ] I can slice rows, columns and rectangular regions out of a 2D array, and reverse an axis with a negative step.
- [ ] I can predict the result shape when two differently-shaped arrays are combined, using the broadcasting rules.
- [ ] I can say what `axis=0` vs `axis=1` does to a 2D array without running it.
- [ ] I can select and modify elements with a boolean mask, combining conditions with `&` and `|`.
- [ ] I can explain why `uint8` arithmetic overflows and how `astype` + `np.clip` fixes it.
- [ ] I can spot a slow Python loop over data and rewrite it as a vectorized array operation.
- [ ] I updated an entire 2D grid simultaneously using shifted slices — no per-cell loop.

## Next up

Arrays hold the numbers; next you learn the tool that gives rows and columns *names* and lets you interrogate real datasets — [06 · Pandas: Interrogating Real Datasets](06-pandas.md).
