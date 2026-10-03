# 38 · Generative Models: Autoencoders, GANs and Diffusion

**Phase 6 — Special Topics** · Estimated time: 2-3 weeks · Prerequisites: [24 · PyTorch Fundamentals](../phase-3-deep-learning/24-pytorch-fundamentals.md), [25 · The Dark Arts of Training Deep Networks](../phase-3-deep-learning/25-training-deep-nets.md), [26 · Convolutional Neural Networks](../phase-3-deep-learning/26-cnns-computer-vision.md)

> Every model built in the earlier lessons answers questions such as "which digit is this?" or "what comes next?". This lesson flips the direction and builds models that *create*: an autoencoder that compresses digits into a 2D space where one digit can morph into another, a forger network trained against a detective network until the forger draws convincing digits, and a toy diffusion model that turns pure random noise into a spiral, step by step. These three projects are the ideas behind Stable Diffusion, Midjourney and DALL-E, shrunk to a size where every moving part is visible. Two results stand out: decoding the straight line between a "3" and an "8", so that one digit melts into the other, and a cloud of random static collapsing into a perfect spiral. This is an optional-track lesson, but those two results alone make it worth doing.

## What this lesson builds

- **Project 1 — Autoencoder on MNIST**: a network that squeezes each digit image through a 2-number bottleneck and back, plus a scatter plot of that 2D "latent space" (the digits form clusters), a morph animation between two digits, and a denoising version that cleans up noisy images.
- **Project 2 — DCGAN on MNIST/FashionMNIST**: a generator and discriminator trained as adversaries, producing a training-progress GIF of fake digits going from static to legible, plus a documented failure (mode collapse) and the fix.
- **Project 3 — Toy diffusion on 2D point clouds**: a forward noising schedule, a small MLP that learns to predict noise, and a sampling loop whose animation shows scattered noise condensing into a spiral.

## Concepts covered

- **Generative vs discriminative**: a discriminative model draws boundaries between existing things; a generative model learns the shape of the data itself, well enough to make new samples.
- **Autoencoder**: a network trained to compress its input and reconstruct it; the compressed middle is forced to keep only what matters.
- **Latent space**: the compressed coordinate system in the middle of an autoencoder, where similar inputs land near each other.
- **Latent interpolation and arithmetic**: walking a line between two points in latent space and decoding each step; meaning becomes geometry that math can work on.
- **VAE idea (intuition only)**: a variational autoencoder trains the latent space to be smooth and gap-free, so *any* sampled point decodes to something sensible.
- **GAN** (generative adversarial network): a forger (generator) and a detective (discriminator) trained against each other until the forgeries are convincing.
- **Adversarial instability and mode collapse**: why GAN training is a knife-edge, and what it looks like when the generator gives up and draws the same thing forever.
- **Diffusion**: a model learns to remove a little noise; then it starts from pure noise and removes a little, hundreds of times, until a sample appears.
- **Why diffusion won**: its training objective is a plain, stable regression (predict the noise), so it scales where GANs wobble.

## Before starting

Check that PyTorch from lesson 24 runs, then install this lesson's extras inside the repo venv:

```bash
cd ~/ml/ml-learn
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install matplotlib imageio scikit-learn
mkdir -p work/38-generative-models
cd work/38-generative-models
```

As in lesson 24, torch and torchvision come from PyTorch's own package index. If lesson 24 already installed them (even a GPU build), that line leaves them as they are. That index carries no matplotlib, imageio or scikit-learn, so they get a plain `pip install` line of their own. Current PyTorch has no Intel Mac version, so on an Intel Mac do this lesson in a free Google Colab notebook, as lesson 01 suggested.

No dataset needs to be downloaded by hand: `torchvision` fetches MNIST and FashionMNIST itself (no account, ~12 MB each). Test it:

```python
from torchvision import datasets, transforms
ds = datasets.MNIST(root="data", download=True, transform=transforms.ToTensor())
print(len(ds), ds[0][0].shape)   # 60000 torch.Size([1, 28, 28])
```

All three projects train fine on a CPU in minutes to an hour. Only the stretch goals call for a GPU (free Colab works).

## Project 1 — Autoencoder on MNIST

**Goal:** Build a network that compresses every 784-pixel digit down to just **2 numbers** and reconstructs it. Then explore the 2D latent space that emerges: clusters, morphs, and denoising.

**Milestones**

- [ ] Write `autoencoder.py`. Load MNIST and flatten each image to a 784-long vector (values 0-1). Build two networks: an **encoder** `784 → 128 → 2` and a **decoder** `2 → 128 → 784` with a `Sigmoid` on the output (a function that squashes values into 0-1, matching the pixel range). An autoencoder is just these two glued together.
- [ ] Train it with the simplest possible objective: the output should equal the input. Use `nn.MSELoss()` between reconstruction and original, Adam with `lr=1e-3`, batch size 128, ~20 epochs. Note what is strange here: **there are no labels**. The image is its own target. Checkpoint: loss falls steadily and ends around 0.03-0.05.
- [ ] Plot 8 test images above their reconstructions (`plt.subplots(2, 8)`). Checkpoint: the reconstructions are blurry, but most digits are still recognizable. Ten thousand pixels of information survived a 2-number bottleneck. That is what "keep only what matters" means.
- [ ] The key result: encode the whole test set to get 10,000 points of shape `(10000, 2)`, and scatter-plot them colored by digit label (`plt.scatter(z[:,0], z[:,1], c=labels, cmap="tab10", s=2)` plus `plt.colorbar()`). Checkpoint: **digits form visible clusters**. Nobody told the network what a "7" is, yet all the 7s landed close together. That map is the latent space.
- [ ] Latent morph: encode one "3" and one "8" to get points `z_a` and `z_b`. Compute 10 evenly spaced points on the line between them (`z_a + t*(z_b - z_a)` for `t` in 0…1) and decode each. Plot the 10 outputs in a row. Checkpoint: a smooth morph. The 3 grows a loop and *becomes* the 8, passing through plausible in-between digits. The row is a walk through the space of digits.
- [ ] Denoising autoencoder: retrain, but corrupt each input with `noisy = (x + 0.4*torch.randn_like(x)).clamp(0,1)` while the target stays the **clean** `x`. Feed the trained model noisy test images. Checkpoint: static-covered digits come out clean. Keep this idea for later: *a network can learn to remove noise*. That one idea is the seed of Project 3 and of Stable Diffusion.
- [ ] In `notes.md`, answer: why do morph midpoints sometimes look like a *different* valid digit? And why might a random latent point decode to garbage? (That second question is exactly the gap a **VAE** fixes: it adds a training nudge that keeps the latent space smooth and gap-free, so every point decodes to something sensible. Intuition is enough here; there is no need to implement one.)

<details><summary>Hints</summary>

- Keep the encoder and decoder as separate `nn.Sequential` modules inside one `nn.Module`. The scatter plot and the morph need to call them independently.
- Flattening: `transforms.Lambda(lambda x: x.view(-1))` in the transform, or `x.view(x.size(0), -1)` in the training loop. Pick one, not both.
- Blurry reconstructions are *correct*, not a bug: 2 dimensions cannot hold stroke-level detail. Try latent size 32 once to see near-perfect reconstructions, then go back to 2 (the plots need 2D).
- Wrap evaluation code in `with torch.no_grad():` and call `.cpu().numpy()` before handing tensors to matplotlib.

</details>

**Definition of done:** the reconstruction figure, the clustered latent scatter plot, a 10-frame morph strip and a working denoiser are all committed, along with the notes.

## Project 2 — DCGAN on MNIST/FashionMNIST

**Goal:** Train two networks as adversaries: a generator (forger) that turns random noise into images, and a discriminator (detective) that judges real vs fake. Capture the forger's improvement as a GIF. Expect turbulence: documenting one failure and one fix is part of the project.

**Milestones**

- [ ] Read the "what is a GAN" opening of the [PyTorch DCGAN tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html). Stop before the code, because this project builds the 28×28 version from scratch. A **DCGAN** is simply a GAN whose two networks are CNNs (deep convolutional GAN).
- [ ] Write `dcgan.py` with the two players. **Generator**: a 100-number noise vector → `ConvTranspose2d` layers (a convolution that *upsamples*, growing the image instead of shrinking it) → a `1×28×28` image, `Tanh` output. **Discriminator**: the mirror image (`Conv2d` layers from lesson 26 → a single sigmoid score, real or fake). Use `BatchNorm2d` in both (per the DCGAN recipe). Scale real images to [-1, 1] with `transforms.Normalize((0.5,), (0.5,))` to match `Tanh`. Checkpoint: `G(torch.randn(64, 100, 1, 1))` returns shape `(64, 1, 28, 28)` before any training.
- [ ] Write the alternating training loop with `nn.BCELoss()`. This loop is the whole adversarial game: **(1)** discriminator step: maximize its accuracy on a real batch (target 1) and a fake batch (target 0); **(2)** generator step: generate fakes and train them toward the discriminator saying *real* (target 1). Use two optimizers, both Adam, with `lr=2e-4` and `betas=(0.5, 0.999)`. These exact numbers are famous folklore; change them and expect unstable training.
- [ ] Before the loop, create `fixed_noise = torch.randn(64, 100, 1, 1)` once. After every epoch, run it through the generator and save the 8×8 grid (`torchvision.utils.save_image(fakes, f"epoch_{e:02d}.png", normalize=True)`). Because the noise is the same every time, the grids show the *same 64 imaginary digits* sharpening across epochs.
- [ ] Train for ~25 epochs on MNIST (~20-40 min on a CPU; FashionMNIST is a drop-in swap for drawing handbags instead of digits). Log both losses each epoch. Checkpoint: by epoch 1 the grid is structured static, by ~5 there are ghostly digit-shapes, by ~25 most of the 64 are legible. The losses **oscillate without converging**. That is what an adversarial game looks like, not a sign of broken code. There is a real problem only if D's loss slides to ~0 (detective too strong: G gets no signal) or G's loss explodes.
- [ ] Make the training GIF:

  ```python
  import imageio.v2 as imageio, glob
  frames = [imageio.imread(f) for f in sorted(glob.glob("epoch_*.png"))]
  imageio.mimsave("training.gif", frames, duration=0.3)
  ```

  Checkpoint: noise → digits, in one looping animation.
- [ ] Break it on purpose. Raise both learning rates to `1e-3` (or train D twice per G step) and rerun a few epochs. The aim is to trigger **mode collapse**: the generator discovers one output that fools D and produces only that, so the grid shows 64 near-identical digits. Save that grid as `mode_collapse.png`. This is the GAN disease, and it is why the field moved on.
- [ ] Apply a fix and recover: go back to `2e-4` and add **label smoothing** (train D on real images with target 0.9 instead of 1.0). This stops D from becoming overconfident and keeps gradients flowing to G. In `notes.md`, write three lines on symptom → cause → fix.

<details><summary>Hints</summary>

- Getting `ConvTranspose2d` to land exactly on 28×28 is fiddly. An easy path: project noise to `7×7×128` with a Linear layer + reshape, then two stride-2 transposed convs take the size 7→14→28.
- In the discriminator step, `.detach()` the fake images so that D's loss does not backprop into G. In the generator step, do not detach. This one line is the classic GAN bug.
- Each network updates only itself: `d_optimizer.step()` after D's loss, `g_optimizer.step()` after G's. Call the right `zero_grad()` before each.
- Nothing legible by epoch 10? Check in order: images normalized to [-1,1]? `betas=(0.5, 0.999)` on both optimizers? BatchNorm present? The DCGAN recipe genuinely needs all three.

</details>

**Definition of done:** `training.gif` shows noise becoming digits; `mode_collapse.png` plus notes document one failure and the fix that recovered it.

## Project 3 — Toy diffusion on 2D point clouds

**Goal:** Build a complete diffusion model (forward noising, a noise-predicting MLP and an iterative sampler) on 2D points instead of images, so that every stage can be watched as a scatter plot. The math is the same as in Stable Diffusion, at a thousandth of the size.

**Milestones**

- [ ] Make a dataset of ~2,000 2D points shaped like a spiral (or `sklearn.datasets.make_moons(2000, noise=0.05)` for two moons). Spiral recipe: `t = torch.rand(n)*3*torch.pi; x = t*torch.cos(t); y = t*torch.sin(t)`, stack, add a pinch of noise, then normalize to roughly [-1, 1]. Scatter-plot it. Checkpoint: a clean spiral. This point cloud *is* the "data distribution": each point plays the role that one image plays for Stable Diffusion.
- [ ] Build the **forward process**: destruction on a schedule. Pick `T = 200` steps and a linear **beta schedule** (how much noise each step adds):

  ```python
  T = 200
  betas = torch.linspace(1e-4, 0.02, T)
  alphas = 1.0 - betas
  alpha_bar = torch.cumprod(alphas, dim=0)   # cumulative signal remaining at each step
  ```

  The shortcut formula jumps to any step t in one line: `x_t = sqrt(alpha_bar[t]) * x0 + sqrt(1 - alpha_bar[t]) * noise`, where `noise = torch.randn_like(x0)`. Plot the cloud at t = 0, 50, 100, 199. Checkpoint: spiral → fuzzy spiral → vague swirl → pure Gaussian blob. The structure dissolves into noise.
- [ ] Build the model: a small MLP that takes a noisy point *and* its timestep, and predicts **the noise that was added**. The input is `(x, y, t/T)` (3 numbers), the hidden layers are ~128 wide with ReLU, and the output is 2 numbers. That is the whole model. It does not guess the clean point; it predicts the *noise*. Learning to spot the noise and learning the data's shape turn out to be the same skill.
- [ ] Training loop: notice how boring it is, and that this boringness is the main point. Sample a batch of clean points, a random `t` per point (`torch.randint(0, T, ...)`) and fresh noise; form `x_t` with the shortcut formula; loss = `mse_loss(model(x_t, t), noise)`. This is plain, stable regression: no adversary, no knife-edge, no mode collapse. **This is why diffusion won.** Train for ~3,000 steps with Adam, `lr=1e-3` (about a minute on a CPU). Checkpoint: the loss falls from ~1.0 toward ~0.3-0.5 and flattens.
- [ ] Write the **sampler**: creation by iterated cleanup. Start from `x = torch.randn(1000, 2)` and step t = T-1 → 0: predict the noise, subtract the right fraction of it, and (for every step except the last) re-add a smaller dose of fresh noise. The per-step update is the one formula worth copying from a reference: take "Algorithm 2 (Sampling)" from [Lilian Weng's diffusion post](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) (skim it for the pictures and boxed algorithms; ignore the derivations). Checkpoint: the 1,000 sampled points form a spiral. **The model created the spiral out of pure noise.**
- [ ] Make the collapse animation: during sampling, snapshot the cloud every ~10 steps into a scatter-plot PNG, then stitch the snapshots together with imageio as in Project 2. Checkpoint: `diffusion.gif` shows formless static condensing into a spiral. This is the second standout animation of the lesson.
- [ ] In `notes.md`, connect the three projects: how is the denoising autoencoder (Project 1) like one *single* step of diffusion? Why is "predict the noise, many small steps" so much more stable than "fool the detective" (Project 2)? Then scale this mental model up: swap 2D points for images and the MLP for a U-Net-style CNN, and run diffusion in an autoencoder's latent space instead of pixels. That is *latent diffusion*, i.e. Stable Diffusion. The three projects have now built every ingredient at toy scale.

<details><summary>Hints</summary>

- A common source of shape bugs: `alpha_bar[t]` for a batched `t` has shape `(batch,)` but `x0` is `(batch, 2)`, so add `.unsqueeze(-1)` (or `[:, None]`) before multiplying.
- Feed the timestep as `t.float() / T` so it lives in [0, 1]; concatenate with the coordinates via `torch.cat([x, t_scaled], dim=1)`.
- Samples form a blob instead of a spiral? Usual suspects, in order: no fresh noise re-added during intermediate sampling steps (the samples "freeze" too early), an index slip between `t` and `t-1` in the update, or data never normalized to ~[-1, 1].
- Sanity-check the sampler before debugging the model: at `t=199` the predicted noise should have roughly unit variance (`pred.std()` near 1).

</details>

**Definition of done:** `diffusion.gif` shows noise collapsing into the spiral, and the notes connect autoencoder → GAN → diffusion in the learner's own words.

## Stretch goals

- **Tiny image diffusion on MNIST** (Colab GPU recommended): swap the MLP for a small CNN with a few residual blocks, T=300, and diffuse actual 28×28 digits. Getting recognizable digits from noise, with code written from scratch, is a rite of passage.
- **Latent arithmetic**: with Project 1's encoder, compute the mean latent of all 1s and all 7s, then decode points along `mean_1 + t*(mean_7 - mean_1)`. Try `z(7) - z(1) + z(4)`: what comes out?
- **Conditional generation**: give Project 3's MLP a one-hot class input (spiral vs moons trained together) and generate whichever shape the class input asks for. This is the toy version of a text prompt guiding Stable Diffusion.
- **FashionMNIST DCGAN**, and interpolate between two noise vectors `z`. Do the *generated garments* morph smoothly like the autoencoder latents did?

## Getting unstuck

- **Read the error bottom-up.** The last line names the problem, and for shape mismatches it names both shapes.
- **Print shapes obsessively.** Every bug in this lesson is a shape bug in disguise: broadcasting `(batch,)` against `(batch, 2)`, an image flattened twice, a grid saved in [-1,1] without `normalize=True` (it renders as grey mush).
- **GAN looks broken?** First decide whether it actually is: oscillating losses are normal, and epoch-3 samples are always ugly. Judge by the fixed-noise grids across epochs, never by the loss curve alone.
- **Bisect the diffusion pipeline.** Forward process wrong? (Plot `x_t` at several t; it should degrade gradually.) Model wrong? (Training loss should fall.) Sampler wrong? (Most likely; recheck each symbol against the boxed algorithm.)
- Study the problem for 20 minutes, then ask an AI assistant for a **HINT, not a solution** (for example, "my DCGAN discriminator loss goes to zero by epoch 2, what direction should I look?"), and **type all code by hand**. Muscle memory is the point of this curriculum.

## Resources

- [PyTorch DCGAN tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html) — the canonical DCGAN recipe (on faces); read the intro before Project 2, and compare code *after* writing `dcgan.py`.
- [Lilian Weng — What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) — the reference diffusion write-up. Heavy math; skim for the diagrams and the two boxed algorithms, which map 1:1 onto Project 3.
- [pytorch.org docs](https://pytorch.org/docs/stable/index.html) — look up `ConvTranspose2d` output-size arithmetic and `torchvision.utils.save_image`.

## Skills unlocked

- [ ] I can explain generative vs discriminative models with an example of each.
- [ ] I can build and train an autoencoder and explain what its latent space is.
- [ ] I can interpolate in latent space and explain why the decoded morph looks smooth.
- [ ] I can explain the GAN training game, and name one instability (mode collapse) with a fix I have used.
- [ ] I can write a diffusion forward-noising process and the shortcut formula to any timestep.
- [ ] I can train a noise-prediction model and sample from it by iterative denoising.
- [ ] I can explain in plain words why diffusion training is more stable than GAN training.
- [ ] I can trace the line from the three toy projects to how Stable Diffusion works (latent diffusion).

## Next up

After models that perceive and imagine, the next lesson turns to models that *act*: [39 · Reinforcement Learning: Agents That Learn from Reward](39-reinforcement-learning.md) trains agents that improve by trial, error and reward.
