# 05 · NumPy: Thinking in Arrays

**Phase 0 — Foundations** · Estimated time: 1 week · Prerequisites: [02 · Python Basics](02-python-basics.md), [03 · Data Structures](03-python-data-structures.md), [04 · Classes, Files, JSON and Errors](04-python-oop-files-errors.md)

> In machine learning, every image, every dataset and every neural network weight is stored and manipulated as a NumPy-style array. This lesson teaches the single most important mental shift in the whole curriculum: moving from loops that touch one number at a time to operations that transform millions of numbers at once. To prove that this is not just a style preference, the three projects edit a real photo with pure math, run a speed race in which NumPy beats a hand-written Python loop by 50–100x, and build Conway's Game of Life on a 2D grid. When arrays feel natural, everything after this lesson gets easier.

## What this lesson builds

- **Project 1 — Images are just arrays:** a photo-editing script that grayscales, flips, crops, brightens and inverts a real photo using only array operations, saving each result as an image file to open and view.
- **Project 2 — Speed race:** a benchmark script that computes the same sum-of-squares on 10 million numbers with a Python loop and with NumPy, and prints a timed scoreboard showing NumPy winning by 50–100x.
- **Project 3 — Conway's Game of Life:** a self-running 2D world where cells live and die by simple rules, updated entirely with array slicing (no per-cell loops) and animated in the terminal or with matplotlib.

## Concepts covered

- **ndarray**: NumPy's core object, a grid of numbers all of the same type, from 1D lists to 2D tables to 3D image stacks.
- **shape and dtype**: the size of each dimension, and the type of number stored inside (for example `uint8` or `float64`).
- **indexing and slicing in 2D**: selecting rows, columns and rectangular chunks with `arr[rows, cols]`.
- **elementwise operations**: `a + b`, `a * 2` and `a > 5` applied to every element at once, without a hand-written loop.
- **broadcasting**: NumPy's rules for combining arrays of different shapes (the trickiest and most useful idea in this lesson).
- **aggregations and axis**: `sum`, `mean`, `max` and `argmax`, and what `axis=0` and `axis=1` actually mean.
- **boolean masking**: using an array of True/False values to select or modify only the elements that matter.
- **np.random**: generating random arrays for simulations and, later, for initializing neural networks.
- **vectorization**: why NumPy is 50–100x faster than Python loops, and how to rewrite loop-thinking as array-thinking.

## Before starting

Check the foundations first. This lesson relies heavily on lists, `for` loops, functions and running a script from the terminal, all from lessons 02–04. If any of these do not feel comfortable yet, go back to those lessons.

Activate the venv (the isolated Python environment made in lesson 01) and install the packages for this lesson:

```bash
cd ~/ml/ml-learn
source .venv/bin/activate
pip install numpy pillow matplotlib
```

- **NumPy** is the array library. **Pillow** loads and saves image files. **matplotlib** draws plots (a preview of lesson 07; this lesson only uses it to display grids).

Create the work folder for this lesson:

```bash
mkdir -p work/05-numpy
cd work/05-numpy
```

Project 1 also needs one photo. Any personal JPEG works (a pet, a meal, a holiday shot; ideally at least 500×500 pixels). Put it in this folder as `photo.jpg`. To drag one in, open the folder in the file manager: `open .` on macOS, `explorer.exe .` on Windows (WSL2), `xdg-open .` on Linux. Or copy it in the terminal: on macOS and Linux, `cp ~/Pictures/YOUR-PHOTO.jpg photo.jpg`; on Windows (WSL2), Windows files are under `/mnt/c/Users/`, so `cp /mnt/c/Users/YOUR-WINDOWS-NAME/Pictures/YOUR-PHOTO.jpg photo.jpg`.

Finally, spend 30–60 minutes skimming the [NumPy absolute beginners guide](https://numpy.org/doc/stable/user/absolute_beginners.html) with a Python prompt open, typing every example. Skim it now, and refer back to it all week.

## Project 1 — Images are just arrays

**Goal:** Discover that a photo is literally a 3D array of numbers (height × width × 3 color channels) and edit it with nothing but array math. No loops are allowed anywhere in this project.

**Milestones**

- [ ] Create `images.py`. Load the photo and convert it to an array:

  ```python
  from PIL import Image
  import numpy as np

  img = np.array(Image.open("photo.jpg"))
  print(img.shape, img.dtype)
  ```

  **Checkpoint:** the script prints something like `(1024, 768, 3) uint8`, meaning height, width, and 3 values (red, green, blue) per pixel, each an integer from 0 to 255. `uint8` means "unsigned 8-bit integer": a whole number from 0 to 255.
- [ ] Look at single pixels: print `img[0, 0]` (top-left pixel, 3 numbers) and `img[0, 0, 0]` (just its red value). Print `img[:5, :5, 0]`, the red channel of the top-left 5×5 corner. This is 2D/3D indexing: one index (or slice) per dimension, separated by commas.
- [ ] Write a helper `save(arr, name)` that does `Image.fromarray(arr).save(name)`, so that every result can be checked by eye. Call it after each step. Open the saved files in VS Code's file explorer.
- [ ] **Crop:** slicing a 2D array works like slicing a list, once per axis: `img[100:400, 200:500]` keeps rows 100–399 and columns 200–499. Crop an interesting region of the photo and save it. **Checkpoint:** the saved image shows the intended region (expect a few tries: rows come first and columns second, which surprises most people).
- [ ] **Flip:** a slice step of `-1` reverses an axis. Flip the photo upside down, then mirror it left-right, using only slicing. Save both. **Checkpoint:** the mirrored image looks like the photo seen in a mirror.
- [ ] **Invert:** `255 - img` subtracts every one of the millions of values from 255 in one line. That is an elementwise operation. Save it. **Checkpoint:** it looks like an old film negative.
- [ ] **Brighten:** done naively, `img + 80` overflows: 250 + 80 wraps around to 74 in `uint8`, causing strange dark speckles in bright areas. The fix is a dtype round-trip: convert to a bigger type, add, cap at 255 with `np.clip`, and convert back:

  ```python
  bright = np.clip(img.astype(np.int64) + 80, 0, 255).astype(np.uint8)
  ```

  Save it. **Checkpoint:** the photo is brighter, with no strange speckles. This step shows why dtype matters.
- [ ] **Grayscale, and a first look at broadcasting.** A grayscale pixel is a weighted mix of red, green and blue: `0.299*R + 0.587*G + 0.114*B` (the classic luminance formula). Compute it for all pixels at once by multiplying the whole image by a 3-number array of weights, then summing away the color axis. Study this comment until it makes sense:

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

  Convert back to `uint8` before saving. **Checkpoint:** a proper black-and-white version of the photo, with a smaller shape too (print it).
- [ ] **Boolean masking finale:** `img.sum(axis=-1) > 400` gives a `(height, width)` array of True/False values, a **mask** that marks the bright pixels. Use it to index the image (`img[mask] = ...`) and turn all bright pixels pure red. Save it. **Checkpoint:** the bright regions of the photo (sky, lights, white walls) are now red.

<details><summary>Hints</summary>

- Crop confusion: `img[a:b, c:d]` means rows a–b (vertical), then columns c–d (horizontal). If the crop looks transposed, the two were swapped.
- `Image.fromarray` needs `uint8`. If saving throws a type error or produces garbage, print `arr.dtype`. After any math involving floats, the array must be converted with `.astype(np.uint8)` (clip first).
- For grayscale, `img * np.array([0.299, 0.587, 0.114])` broadcasts exactly as the comment shows; then sum along the last axis.
- Masking a color image: a `(h, w)` boolean mask used on a `(h, w, 3)` image selects whole pixels, so `img[mask] = [255, 0, 0]` sets each selected pixel to red. That is broadcasting again.

</details>

**Definition of done:** `images.py` saves cropped, flipped, mirrored, inverted, brightened, grayscale and red-masked versions of the photo and contains zero `for` loops, and the broadcasting comment can be explained out loud.

## Project 2 — Speed race

**Goal:** Prove with a stopwatch that vectorized NumPy is far faster than Python loops, and understand why. This number, 50–100x, is the reason all of ML is built on arrays.

**Milestones**

- [ ] Create `speed_race.py`. Make the contestants' data: `data_list = list(range(10_000_000))` and `data_array = np.arange(10_000_000)` (`np.arange` is NumPy's `range` that returns an ndarray directly).
- [ ] Write `sum_of_squares_loop(numbers)` in the style of lesson 02: a `for` loop accumulating `total += x * x`. Pure Python, no NumPy inside.
- [ ] Write `sum_of_squares_numpy(arr)` as a single expression: square the array elementwise, then `.sum()`. **Checkpoint:** both functions return the same value on `range(100)` / `np.arange(100)`: 328350.
- [ ] Time both on the 10-million-element data with `time.perf_counter()` (a high-precision clock from Python's built-in `time` module):

  ```python
  import time
  start = time.perf_counter()
  result = sum_of_squares_loop(data_list)
  elapsed = time.perf_counter() - start
  ```

  Print each contestant's time and the ratio. **Checkpoint:** the loop takes on the order of a second; NumPy takes milliseconds; the printed ratio is roughly 50–100x (anywhere from 30x to 300x is normal, depending on the computer).
- [ ] There is one catch, and it is a useful one: the two totals do *not* match. Print both and compare them. The true sum of squares here is about 3.3 × 10²⁰, but `np.arange` gives an `int64` array (check `data_array.dtype`), and `int64` maxes out near 9.2 × 10¹⁸. So NumPy's total silently wraps around, exactly like the `uint8` speckles in Project 1, just at 64-bit scale. The Python loop is immune: plain Python ints grow as big as they need to. Fix the NumPy side by computing in floats with `(data_array.astype(np.float64) ** 2).sum()`, and compare the result to `float(loop_total)`. **Checkpoint:** the two float totals agree to about 15 significant digits (a difference in the last digit or two is ordinary float rounding, not overflow).
- [ ] Add a third contestant: a Python loop *over a NumPy array* (`for x in data_array: ...`). **Checkpoint:** it is the *slowest* of the three, because looping in Python throws away all of NumPy's speed. Moral: never loop over an array's elements.
- [ ] In a comment at the bottom, explain the result in the learner's own words. The short version: the Python loop makes the interpreter examine one number at a time, with type-checking overhead per element; `(arr ** 2).sum()` hands the whole array to compiled C code that races through memory in bulk. Replacing per-element Python loops with whole-array operations is called **vectorization**.

<details><summary>Hints</summary>

- Run several timings back to back (or wrap the timing in a small helper that repeats 3 times and takes the minimum). The first run can be slower while things warm up.
- If the loop version is too slow to wait for, drop to 1 million elements, get everything working, then scale back up.
- Ratio wildly off? Make sure the timing does not accidentally include the creation of the data; time only the computation.

</details>

**Definition of done:** `speed_race.py` prints three labeled timings and the NumPy-vs-loop ratio; the ratio is at least 30x; the closing comment correctly names vectorization as the reason.

## Project 3 — Conway's Game of Life

**Goal:** Build a famous zero-player game: a 2D grid of cells where each generation, cells live or die based on how many of their 8 neighbors are alive. The catch that makes it a real NumPy challenge: compute every cell's neighbor count *simultaneously* using array operations, with no per-cell loops.

The rules (from mathematician John Conway, 1970): a live cell with 2 or 3 live neighbors survives; a dead cell with exactly 3 live neighbors becomes alive; every other cell is dead next generation.

**Milestones**

- [ ] Create `life.py`. Make a random starting world: a 30×40 grid of 0s (dead) and 1s (alive), about 25% alive. `np.random.random((30, 40)) < 0.25` gives booleans; `.astype(np.uint8)` makes them 0/1. `np.random` is NumPy's random-array toolbox; it comes back later for initializing neural networks. **Checkpoint:** `grid.shape` is `(30, 40)` and `grid.mean()` is near 0.25 (why does the mean of 0s and 1s equal the fraction alive? Think it through: this trick appears constantly in ML).
- [ ] Print the world in a readable form: write a function that prints each row as characters, for example `'█'` for alive and `'·'` for dead. Loops are allowed for *printing*; the no-loop rule applies to the update math.
- [ ] The core: `count_neighbors(grid)` returns a same-shaped array where each entry is that cell's number of live neighbors. The vectorized idea: shifting the whole grid one step in each of the 8 directions and summing the shifts counts every cell's neighbors at once. Pad the grid with a border of zeros first (`np.pad(grid, 1)`) so edge cells simply see dead space outside; then each direction is a `(30, 40)` slice of the padded `(32, 42)` array. **Checkpoint:** on a tiny 4×4 test grid with one live cell in the middle, the count array shows 1 in the 8 surrounding cells and 0 in that cell itself.
- [ ] Write a `step(grid)` function that applies Conway's rules using only comparisons and boolean logic on whole arrays. Combine masks with `&` (and) and `|` (or); for example, survivors are `(grid == 1) & ((n == 2) | (n == 3))`. Note that NumPy needs `&`/`|` with parentheses, not Python's `and`/`or`. Return the new grid; never modify the old one mid-computation (every cell's fate depends on the *old* neighbor counts).
- [ ] Test with a known pattern: a **blinker**, three live cells in a horizontal row on an otherwise dead grid. **Checkpoint:** after one step it becomes a vertical row of three, after two steps horizontal again, forever. If this works, the rules in `step` are provably correct.
- [ ] Run it: loop 50 generations, printing each and pausing with `time.sleep(0.1)`. Clearing the terminal between frames (`print("\033[2J\033[H", end="")`) makes it a real animation. **Checkpoint:** the world visibly evolves: flickering chaos settles into stable blocks, oscillators, and maybe a glider crawling diagonally across the grid.
- [ ] Seed a **glider** by hand (search "game of life glider" for the 5-cell pattern) on an empty grid. **Checkpoint:** it walks one cell diagonally every 4 generations.

<details><summary>Hints</summary>

- The padded-slice trick, half-revealed: with `p = np.pad(grid, 1)`, the "north-west neighbor" of every cell is `p[:-2, :-2]`, and `p[1:-1, 1:-1]` is the grid itself. Work out the other 7 slices on paper by drawing a 4×4 grid and its 6×6 padded version. Sum the 8 shifted slices, not the center.
- Wrong neighbor counts? Print the shapes first: every one of the 8 slices must be exactly `grid.shape`. Then test on the single-live-cell 4×4 grid, where the counts can be checked by hand.
- Blinker stuck or exploding? Classic cause: updating the grid in place while still reading it. Build the new grid entirely from the old one.
- Prefer graphics over terminal art? `matplotlib.pyplot.imshow(grid)` draws the grid as an image; call `plt.pause(0.1)` between generations. On macOS a window opens straight away. On Windows (WSL2) and Linux, matplotlib draws its windows with the tkinter toolkit, and no window opens until tkinter is installed: `sudo apt install -y python3.13-tk`.

</details>

**Definition of done:** `life.py` animates 50 generations from a random seed with a loop-free `step`, the blinker oscillates with period 2, and a glider travels.

## Stretch goals

- **Sepia filter:** in Project 1, multiply the image by a 3×3 color-mixing matrix (search "sepia matrix") for that old-photograph look. It is a preview of the matrix multiplication that [lesson 08](../phase-1-math/08-linear-algebra-by-code.md) builds.
- **Wrap-around world:** change Life's edges so gliders exiting right re-enter left (look at `np.roll`; it makes the neighbor counting even cleaner than padding).
- **Race extension:** add contestants to Project 2: Python's built-in `sum` with a generator expression, and NumPy's `np.dot(arr, arr)`. Predict the ranking before running.
- **Axis drills:** make a random `(5, 3)` array of "exam scores" (5 students, 3 subjects); compute each student's mean, each subject's mean, and each subject's best student (`argmax`, which returns the *index* of the maximum). Choose the right `axis` each time without guessing.

## Getting unstuck

- **Print `.shape` and `.dtype` before anything else.** Ninety percent of NumPy bugs are one of those two. Make it a reflex now: it stays the #1 debugging move through the deep-learning phases.
- **Shape mismatch errors** ("operands could not be broadcast together with shapes ...") mean that the two shapes, aligned from the right, have a dimension that is neither equal nor 1. Write the shapes one above the other, right-aligned, like the Project 1 comment.
- **The axis rule, once and for all:** `axis=N` means *axis N is the one that gets collapsed*. For a 2D array, `axis=0` collapses the rows, leaving one result **per column**; `axis=1` collapses the columns, leaving one result **per row**. Memorable version: **the named axis is the axis that disappears.** Check: `(30, 40)` summed with `axis=0` → shape `(40,)`.
- **Shrink the problem.** Debug every 2D operation on a 4×4 array that can be verified by hand before running it on a photo or a 10-million-element benchmark.
- Standing advice: read the error message bottom-up (the last line names the problem, the lines above say where); print shapes and values rather than staring; ask an AI assistant for a **hint** and never a solution; and type all code by hand, with no pasting.

## Resources

- [NumPy absolute beginners guide](https://numpy.org/doc/stable/user/absolute_beginners.html) — the one tutorial to actually work through; covers every concept in this lesson.
- [NumPy learn page](https://numpy.org/learn/) — curated books, videos and next-step tutorials for more depth or a different angle.

## Skills unlocked

- [ ] I can create arrays (`np.array`, `np.arange`, `np.zeros`, `np.random`) and read their `shape` and `dtype` at a glance.
- [ ] I can slice rows, columns and rectangular regions out of a 2D array, and reverse an axis with a negative step.
- [ ] I can predict the result shape when two differently-shaped arrays are combined, using the broadcasting rules.
- [ ] I can say what `axis=0` vs `axis=1` does to a 2D array without running it.
- [ ] I can select and modify elements with a boolean mask, combining conditions with `&` and `|`.
- [ ] I can explain why `uint8` arithmetic overflows and how `astype` + `np.clip` fixes it.
- [ ] I can spot a slow Python loop over data and rewrite it as a vectorized array operation.
- [ ] I updated an entire 2D grid simultaneously using shifted slices, with no per-cell loop.

## Next up

Arrays hold the numbers; the next lesson introduces the tool that gives rows and columns *names* and is used to interrogate real datasets: [06 · Pandas: Interrogating Real Datasets](06-pandas.md).
