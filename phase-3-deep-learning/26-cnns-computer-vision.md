# 26 · Convolutional Neural Networks: Teaching Machines to See

**Phase 3 — Deep Learning** · Estimated time: 2 weeks · Prerequisites: [24 · PyTorch Fundamentals](24-pytorch-fundamentals.md), [25 · The Dark Arts of Training Deep Networks](25-training-deep-nets.md), [05 · NumPy](../phase-0-foundations/05-numpy.md)

> Your MNIST network from lesson 23 treated an image as a flat list of pixels — it had no idea that pixel (5,5) sits next to pixel (5,6). Convolutional neural networks (CNNs) fix that, and they are the reason computers can now recognize faces, read street signs, and spot tumors. In this lesson you will first implement convolution yourself in NumPy and *watch* a 3×3 grid of numbers find every edge in a photo — the single best aha moment in deep learning. Then you will build a CNN in PyTorch that classifies real color photos (CIFAR-10) at 80%+ accuracy, and finally open it up to see what it actually learned.

## What you will build

- **Project 1 — Convolution from scratch:** a `conv2d` function in pure NumPy, plus a script that applies hand-made edge/blur/sharpen kernels to a real photo and saves the results as images.
- **Project 2 — CIFAR-10 in PyTorch:** a small CNN (3 conv blocks + classifier head) trained with data augmentation to 80%+ test accuracy on 32×32 color photos, next to an MLP baseline that proves why convolution matters.
- **Project 3 — See what it sees:** visualizations of your trained network's learned filters and feature maps, plus a confusion matrix showing exactly which classes it mixes up.

## Concepts you will learn by doing

- Why MLPs waste parameters on images: a fully connected layer relearns "cat ear" separately for every position, because it has no idea the image is 2D.
- Convolution: sliding a small grid of weights (a *filter* or *kernel*) across an image, computing a weighted sum at each spot — one pattern detector reused everywhere.
- Classic hand-made kernels: edge detection, blur, sharpen — proof that 9 numbers can "see" something.
- Stride (how far the filter jumps each step), padding (extra border pixels so edges aren't lost), and channels (red/green/blue in, many feature maps out).
- Pooling: shrinking feature maps by keeping the strongest activation in each small region.
- The conv → ReLU → pool stack, and how each layer sees a bigger patch of the original image (the *receptive field*).
- Data augmentation: randomly flipping/cropping training images so the network can't memorize exact pixels.
- Reading modern architecture diagrams: LeNet → VGG → ResNet, and what a *skip connection* is.

## Before you start

Check you can do the following (all from earlier lessons): train a PyTorch model with a training loop, DataLoader, and the overfit-one-batch trick (lessons 24–25); write NumPy code with slicing and broadcasting (lesson 5).

Then activate the venv you made in lesson 01, install this lesson's packages and create your work folder:

```bash
cd ~/ml/ml-learn
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install matplotlib pillow numpy
mkdir -p work/26-cnns-computer-vision
cd work/26-cnns-computer-vision
```

As in lesson 24, torch and torchvision come from PyTorch's own package index; if lesson 24 already installed them (even a GPU build), that line leaves them as they are. That index has no matplotlib, so the other packages get a plain `pip install` line of their own. On an Intel Mac the torch line fails, because current PyTorch has no Intel Mac version: do Project 1 on your Mac and Projects 2 and 3 in Google Colab (see **Free GPU** below).

**Get a photo for Project 1.** Any photo with clear shapes works — one of your own is more fun. Put it in this folder, `work/26-cnns-computer-vision`, as `photo.jpg`. To drag one in, open the folder in your file manager: `open .` on macOS, `explorer.exe .` on Windows (WSL2), `xdg-open .` on Linux. Or copy it in the terminal: on macOS and Linux, `cp ~/Pictures/YOUR-PHOTO.jpg photo.jpg`; on Windows (WSL2) your Windows files are under `/mnt/c/Users/`, so `cp /mnt/c/Users/YOUR-WINDOWS-NAME/Pictures/YOUR-PHOTO.jpg photo.jpg`.

**CIFAR-10** (60,000 tiny 32×32 photos in 10 classes: plane, car, bird, cat, deer, dog, frog, horse, ship, truck) downloads automatically via `torchvision.datasets.CIFAR10(root="data", download=True)` — no account needed, ~170 MB.

**Free GPU, 2-minute setup (first lesson where it genuinely helps).** Training Project 2 takes roughly 1–3 hours on a laptop CPU but ~10–15 minutes on a free cloud GPU. Two options, both need a (free) account:

- **Google Colab** (needs a Google account): go to colab.research.google.com → New notebook → menu *Runtime → Change runtime type → T4 GPU*. Paste your training script into a cell.
- **Kaggle** (needs a Kaggle account — you'll want one for lesson 20 anyway): kaggle.com → Create → Notebook → right-hand *Settings → Accelerator → GPU*.

In your script, one line makes it portable: `device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"`, then move the model and each batch with `.to(device)`. It picks an NVIDIA GPU (`cuda`, as on Colab and Kaggle), else the built-in GPU of an Apple Silicon Mac (`mps`, macOS 14 or later), else the CPU. Develop locally with 2–3 epochs; do the real 20–30-epoch run on the GPU.

## Project 1 — Convolution from scratch

**Goal:** Implement 2D convolution yourself in NumPy, apply classic hand-designed 3×3 kernels to a real photo, and *look* at the outputs. Do not skip this — Project 2 makes no sense without it.

**Milestones**

- [ ] Load your photo as a grayscale NumPy array. Starter:

  ```python
  from PIL import Image
  import numpy as np
  img = np.array(Image.open("photo.jpg").convert("L"), dtype=np.float64)
  print(img.shape, img.min(), img.max())   # e.g. (768, 1024) 0.0 255.0
  ```

  Checkpoint: you should see a 2D shape and pixel values between 0 and 255.
- [ ] Write `conv2d(image, kernel)` that slides the kernel over the image. Loops are absolutely fine — clarity beats speed here. For a `(H, W)` image and `(kh, kw)` kernel, the output shape is `(H - kh + 1, W - kw + 1)` (the filter can't hang off the edge). Test it on a tiny 5×5 array of zeros with a single 1 in the middle, using a 3×3 kernel of all ones. Checkpoint: the output is 3×3 and the center row/column pattern shows the kernel "stamped" around the 1.
- [ ] Apply these three classics to your photo and save each result with `Image.fromarray(...)` (clip values to 0–255 and convert to `uint8` first):

  ```python
  edge    = np.array([[-1,-1,-1], [-1, 8,-1], [-1,-1,-1]])
  blur    = np.ones((3, 3)) / 9.0
  sharpen = np.array([[ 0,-1, 0], [-1, 5,-1], [ 0,-1, 0]])
  ```

  Checkpoint: the edge output clearly outlines every shape in the photo on a mostly-black background; blur is visibly softer; sharpen is visibly crisper.
- [ ] Try the Sobel kernel `[[-1,0,1],[-2,0,2],[-1,0,1]]`, which responds only to *vertical* edges. Checkpoint: vertical lines in your photo glow; horizontal lines mostly vanish. Sit with this: 9 numbers just detected "vertical edge" everywhere in the image at once.
- [ ] Add a `stride` parameter (jump `stride` pixels per step) and a `padding` parameter (wrap the image with `np.pad` before convolving). Checkpoint: with `padding=1, stride=1` and a 3×3 kernel, the output shape equals the input shape; with `stride=2` it is about half.
- [ ] Write the punchline in a comment at the top of your script, in your own words: *you* designed these kernels by hand — a CNN puts kernels exactly like these into `nn.Conv2d` layers and **learns their 9 numbers by gradient descent**, discovering whichever detectors best reduce the loss.

<details><summary>Hints</summary>

- The core of `conv2d` is two nested loops over output positions; at each position, slice a `kh × kw` patch with `image[i:i+kh, j:j+kw]` and compute `np.sum(patch * kernel)`.
- If your output is a scrambled mess, check you are slicing `[row, column]` — mixing up i/j is the classic bug. Print `patch.shape` inside the loop once.
- Saving looks wrong / all white? Convolution outputs can go below 0 and above 255. Use `np.clip(out, 0, 255).astype(np.uint8)` before `Image.fromarray`.
- Slow on a huge photo? Resize first: `Image.open("photo.jpg").convert("L").resize((400, 300))`.

</details>

**Definition of done:** your own `conv2d` (with stride and padding) produces a saved edge-detected image where shapes are clearly outlined, and you can explain in one sentence what a CNN does differently with these kernels.

## Project 2 — CIFAR-10 in PyTorch

**Goal:** Beat real color photos with a small CNN. Start from your lesson-24 training-loop template, prove an MLP plateaus around ~50%, then push a 3-block CNN past 80% test accuracy.

**Milestones**

- [ ] Load CIFAR-10 with `torchvision.datasets.CIFAR10` and wrap train/test sets in DataLoaders (batch size 64–128). Look at a grid of 16 training images with their labels using matplotlib. Checkpoint: you can see (and mostly guess) the 10 classes; images are 3×32×32 tensors.
- [ ] Normalize with the standard CIFAR-10 stats, via `transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])`.
- [ ] **Baseline MLP:** flatten each image (`3*32*32 = 3072` inputs) into 2–3 `nn.Linear` layers, train ~10 epochs. Checkpoint: test accuracy stalls around 50–55% no matter what you tweak. This number is your "why CNNs" evidence.
- [ ] Build the CNN — three conv blocks then a classifier head, roughly:

  ```
  block: Conv2d(3→32, kernel 3, padding 1) → ReLU → MaxPool2d(2)   # 32×32 → 16×16
  block: Conv2d(32→64, ...)               → ReLU → MaxPool2d(2)   # 16×16 → 8×8
  block: Conv2d(64→128, ...)              → ReLU → MaxPool2d(2)   # 8×8  → 4×4
  head:  Flatten → Linear(128*4*4 → 256) → ReLU → Linear(256 → 10)
  ```

  Each block halves the spatial size while adding channels: from pixels toward "is there a wheel here?". Checkpoint: `sum(p.numel() for p in model.parameters())` is well under the MLP's count, and a forward pass on one batch outputs shape `(batch, 10)`.
- [ ] **Overfit one batch** (lesson 25 ritual) before any long run. Checkpoint: loss on a single batch drops below 0.01 within a few hundred steps. Also sanity-check the very first loss ≈ `ln(10)` ≈ 2.30 — random guessing over 10 classes.
- [ ] Train ~15 epochs without augmentation, plotting train and test accuracy per epoch. Checkpoint: train accuracy runs well above test (a visible gap) — the network is memorizing. That gap is overfitting, and augmentation is the cure.
- [ ] Add augmentation to the **training** transform only (never the test set): `transforms.RandomCrop(32, padding=4)` and `transforms.RandomHorizontalFlip()` before `ToTensor`. Each epoch now sees slightly shifted/mirrored variants, so exact-pixel memorization stops paying off.
- [ ] Full run: 20–30 epochs (on the free GPU if local is slow), optionally dropping the learning rate ×10 around epoch 20. Checkpoint: **test accuracy ≥ 80%**. Save weights with `torch.save(model.state_dict(), "cifar_cnn.pt")` — Project 3 and lesson 27 need them.

<details><summary>Hints</summary>

- Shape-mismatch error at the first `Linear`? Print the tensor's shape right after the last pool. `Linear` input size must equal `channels * height * width` at that point.
- Stuck at ~10% accuracy (random)? Usual suspects: forgot `optimizer.zero_grad()`, learning rate 10–100× too high, or normalizing train but not test.
- Stuck at 70–75%? Train longer, confirm both augmentations are active, add a second conv per block (conv→ReLU→conv→ReLU→pool), or try a small weight decay (e.g. `5e-4`).
- On Colab/Kaggle, `num_workers=2` in the DataLoader speeds up loading; if it misbehaves, set it back to 0.

</details>

**Definition of done:** a table or plot comparing MLP (~50%) vs CNN-no-aug vs CNN-with-aug, the CNN at 80%+ test accuracy, and saved weights on disk.

## Project 3 — See what it sees

**Goal:** Open the black box: look at the kernels your CNN learned, watch an image transform as it flows through the layers, and find out which classes it confuses.

**Milestones**

- [ ] Reload your trained model (`model.load_state_dict(torch.load("cifar_cnn.pt", map_location="cpu"))`, then `model.eval()`). `map_location="cpu"` lets weights saved on a Colab or Kaggle GPU load on a computer without an NVIDIA GPU, such as a Mac.
- [ ] **Learned filters:** the first conv layer's weights are a tensor of shape `(32, 3, 3, 3)` — 32 learned RGB kernels, direct cousins of your Project 1 kernels. Normalize each to 0–1 and show all 32 in a matplotlib grid. Checkpoint: several filters look like oriented edges or color-contrast blobs — nobody designed them; gradient descent did.
- [ ] **Feature maps:** run one test image through the network layer by layer and plot a handful of channels after each conv block (grab intermediate outputs by calling the blocks manually, or with a *forward hook* — a function PyTorch calls with a layer's output; search "pytorch forward hook"). Checkpoint: early maps look like edge-detected versions of the image; deeper maps are smaller and increasingly abstract blobs.
- [ ] **Confusion matrix:** a 10×10 grid where cell (i, j) counts test images of true class i predicted as class j (your lesson-14 metrics library, or `sklearn.metrics.confusion_matrix`). Plot with `plt.imshow` and class-name tick labels. Checkpoint: a bright diagonal, with the biggest off-diagonal glow at cat↔dog — the classic confusion.
- [ ] Pull up 9 misclassified images with predicted vs true labels. Checkpoint: some are genuinely hard even for you — a lesson in what "80% accuracy" actually means.
- [ ] **Stretch (start of the ResNet idea):** make your network deeper (6+ conv layers) and watch it train *worse* — deep plain stacks degrade. Now add a skip connection: each block computes `output = relu(conv(x) + x)`, giving gradients a shortcut past the layers (channel counts must match — use 1×1 convs or keep channels constant). Checkpoint: the deeper skip-connected version trains at least as well as your 3-block CNN. You have rediscovered ResNet's core trick; the CS231n notes have the architecture story from LeNet → VGG → ResNet.

<details><summary>Hints</summary>

- Filters look like noise? Normalize each 3×3×3 filter independently: `(w - w.min()) / (w.max() - w.min())`, and use `plt.imshow(..., interpolation="nearest")` so 3×3 pixels aren't smoothed away.
- To reach layers inside an `nn.Sequential`, index it: `model.features[0]` is the first layer. `print(model)` shows the tree.
- Wrap evaluation code in `with torch.no_grad():` and move tensors back with `.cpu()` before converting to NumPy for plotting.

</details>

**Definition of done:** three saved figures — filter grid, feature-map progression, confusion matrix — and one paragraph in your notes on what the network learned and where it fails.

## Stretch goals

- Add `nn.BatchNorm2d` after each conv (before ReLU) and measure the effect on training speed and final accuracy — this plus skip connections is most of the modern recipe.
- Push past 90% on CIFAR-10: deeper skip-connected network, more augmentation, learning-rate schedule (`torch.optim.lr_scheduler.OneCycleLR`), ~50+ epochs on GPU.
- Vectorize your NumPy `conv2d` (im2col: unroll patches into a matrix, convolve via one matmul) and benchmark against the loop version.
- Try CIFAR-100 (same download mechanism, 100 classes) and see how far your architecture gets.

## If you get stuck

- **Read the error bottom-up** — the last lines name the failing operation, and in CNNs it is almost always a shape mismatch.
- **Print shapes obsessively.** Temporarily add `print(x.shape)` between layers in `forward`. Know the conv arithmetic: output size = `(input + 2*padding - kernel) / stride + 1`.
- **Trust the ladder:** overfit one batch → short local run → full GPU run. Never debug a 30-epoch run; debug a 50-step one.
- If training diverges (loss → NaN), your learning rate is too high or normalization is missing.
- Ask an AI assistant for a **hint**, not a solution ("why might my CNN be stuck at 10% on CIFAR-10?" — and don't paste its code).
- Type all code yourself. Copy-pasting a CNN teaches nothing; typing one teaches everything.

## Resources

- [CS231n Convolutional Networks notes](https://cs231n.github.io/convolutional-networks/) — the free gold-standard write-up of conv layers, arithmetic, pooling, and the LeNet→VGG→ResNet lineage; read alongside Projects 1–2.
- 3Blue1Brown, *"But what is a convolution?"* — beautiful visual intuition for Project 1; search the title on youtube.com.
- [PyTorch `nn.Conv2d` docs](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) — the exact meaning of every argument you pass in Project 2.
- [torchvision transforms docs](https://pytorch.org/vision/stable/transforms.html) — reference for the augmentation pipeline.

## Skills unlocked

- [ ] I can explain why a fully connected network wastes parameters on images and what convolution reuses.
- [ ] I can implement 2D convolution with stride and padding in NumPy and predict output shapes.
- [ ] I can explain what edge/blur/sharpen kernels do — and that a CNN *learns* its kernels.
- [ ] I can build and train a multi-block CNN in PyTorch and beat 80% on CIFAR-10.
- [ ] I can use data augmentation to shrink the train/test gap and say why it works.
- [ ] I can visualize learned filters and feature maps, and read a confusion matrix.
- [ ] I can describe what a skip connection is and the problem ResNet solved.
- [ ] I can run training on a free cloud GPU with device-portable code.

## Next up

Your CNN learned to see from scratch in two weeks — next you'll borrow a network pre-trained on a million images and bend it to your own photos in an afternoon: [27 · Transfer Learning: A Custom Image Classifier + Demo](27-transfer-learning-vision-project.md).
