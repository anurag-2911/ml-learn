# 27 · Transfer Learning: A Custom Image Classifier + Demo

**Phase 3 — Deep Learning** · Estimated time: 1 week · Prerequisites: [24 · PyTorch Fundamentals](24-pytorch-fundamentals.md), [25 · The Dark Arts of Training Deep Networks](25-training-deep-nets.md), [26 · Convolutional Neural Networks](26-cnns-computer-vision.md)

> **MILESTONE LESSON.** In lesson 26 you trained CNNs on standard datasets that thousands of students have used before you. This week you do something nobody else has done: you train a model on *your own photos* — your pets, your plants, your cooking — and put it on the public internet where anyone with the link can use it. The trick that makes this possible with only ~100 photos per class is **transfer learning**: borrowing a network that someone else already trained on millions of images and gently adapting it to your problem. By the end of the week you will have shipped your first real, working AI product. That is not a metaphor — there will be a URL.

## What you will build

- **Project 1 — Your own dataset:** a folder of 2-5 image classes you personally collected (50-150 photos each), organized into `train/` and `val/`, loading cleanly through PyTorch's `ImageFolder` with augmentation.
- **Project 2 — A fine-tuned ResNet18:** a pretrained network with its final layer swapped for your classes, trained in two stages (head-only, then full fine-tune), hitting 90%+ validation accuracy in minutes — plus a from-scratch baseline that proves why transfer learning wins.
- **Project 3 — A live web demo:** a Gradio app (upload an image → top-3 predictions with confidence bars) deployed free on Hugging Face Spaces, with a shareable public link.

## Concepts you will learn by doing

- What a pretrained ImageNet model already knows (edges → textures → parts → objects) and why that knowledge transfers to your photos.
- **Feature extraction vs fine-tuning**: freezing the pretrained layers vs unfreezing them, and when to do which.
- Small-data tricks: data augmentation, and using a lower learning rate for pretrained layers than for new ones.
- The `torchvision.models` and `torchvision.transforms` toolkits — loading pretrained weights and matching their expected preprocessing.
- Building a web UI in a few lines of Python with Gradio.
- Deploying a demo for free on Hugging Face Spaces.

## Before you start

1. **Prerequisites check.** From lesson 26 you can build and train a CNN in PyTorch, use `DataLoader`, and plot a training curve. From lesson 25 you know what learning rate and overfitting look like. If yes, go.
2. **Install packages** (inside the venv at your repo root):

   ```bash
   cd ~/ml/ml-learn
   source .venv/bin/activate
   pip install torch torchvision gradio
   ```

   (torch and torchvision are already installed from lesson 24 — the line is just so a fresh machine works too.)
3. **No dataset download this time — YOU are the dataset.** You will need your phone and ~1-2 hours of photo taking or photo-library digging. Think now about what 2-5 things you want to classify.
4. **Accounts:** Projects 1 and 2 need no account. Project 3 needs a free Hugging Face account (sign up at [huggingface.co](https://huggingface.co) — it is the standard hub for sharing ML models and demos, and the free tier is enough).
5. **Create your work folder:**

   ```bash
   mkdir -p work/27-transfer-learning
   cd work/27-transfer-learning
   ```

## Project 1 — Your own dataset

**Goal:** Collect and organize a real image dataset of 2-5 classes that you care about, and load it into PyTorch with the right transforms. Data collection is half of real-world ML — this project makes you feel that.

**Milestones**

- [ ] Pick 2-5 classes of things you can photograph or already have photos of. Good picks: your pets (one class per animal), house plants, 3-4 dishes you cook, hand gestures (fist / palm / thumbs-up), types of objects on your desk. Pick things that *look* different — "cat vs dog" is easier than "two similar mugs". Write your class list in a `README.md` in your work folder.
- [ ] Collect **50-150 images per class**. Phone photos are perfect. Vary them deliberately: different angles, lighting, backgrounds, distances. Variety is what teaches the model to generalize instead of memorizing "my cat is always on the blue sofa". Checkpoint: `ls` shows one folder per class, and `ls <class> | wc -l` reports 50+ files each.
- [ ] Transfer photos to WSL2 and organize them like this (this exact layout is what `ImageFolder` expects — folder name = class label):

  ```
  data/
    train/
      class_a/   img001.jpg ...
      class_b/   ...
    val/
      class_a/   ...
      class_b/   ...
  ```

  Write a small Python script that randomly moves ~20% of each class into `val/` (use `random.sample` and `shutil.move` from lesson 04). Never split by hand — you will bias it. Checkpoint: train and val folders exist, val has roughly 20% of each class, and no image appears in both.
- [ ] Write `dataset.py`: build two transform pipelines with `torchvision.transforms`. A **transform** is a preprocessing step applied to each image as it loads. For training: resize/crop to 224×224, random horizontal flip, small random rotation and color jitter (this is **data augmentation** — creating label-preserving variations so the model sees a "bigger" dataset), then `ToTensor()` and `Normalize` with the ImageNet means and stds (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]` — pretrained models expect input scaled exactly the way they were trained). For validation: just resize, center-crop, tensor, normalize — no augmentation, because you want an honest, repeatable measurement.
- [ ] Load both folders with `torchvision.datasets.ImageFolder` and wrap them in `DataLoader`s (batch size 16-32). Checkpoint: printing `dataset.classes` shows your class names, and `len()` of each dataset matches your file counts.
- [ ] Visualize one training batch with matplotlib (undo the normalization before showing, or colors look psychedelic). Checkpoint: a grid of your own photos with correct labels, some visibly flipped/rotated by augmentation.

<details><summary>Hints</summary>

- Getting photos into WSL2: your Windows drives are mounted under `/mnt/c/`, so `cp /mnt/c/Users/<you>/Pictures/... ~/ml/ml-learn/work/27-transfer-learning/data/...` works. Google Photos / phone-to-PC transfer gets them to Windows first.
- HEIC photos (iPhone default) may not load with PIL. Convert them: `sudo apt install libheif-examples` then use `heif-convert`, or set your phone camera to "Most compatible" (JPEG).
- To un-normalize for display: `img * std + mean` (broadcast over the channel dimension — lesson 05 skills), then `plt.imshow(img.permute(1, 2, 0))` because matplotlib wants height × width × channels.
- If `ImageFolder` errors on a corrupt file, find it by looping over the dataset in a try/except and printing the index that fails.

</details>

**Definition of done:** `python dataset.py` prints your classes and dataset sizes and saves a labeled batch grid image, with augmentation visibly working.

## Project 2 — Fine-tune ResNet18

**Goal:** Take ResNet18 pretrained on ImageNet, adapt it to your classes in two stages, and beat 90% validation accuracy in minutes of training. Then train the same architecture from scratch and watch it lose badly — the whole argument for transfer learning in one plot.

**Milestones**

- [ ] Understand what you are borrowing. **ImageNet** is a dataset of ~1.2 million labeled photos in 1000 categories; models pretrained on it have already learned general visual features — early layers detect edges and textures, middle layers detect parts like eyes and wheels, late layers detect whole objects. Those early and middle features are useful for *any* photo problem, including yours. Load the model in `train.py`:

  ```python
  from torchvision import models
  model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
  print(model)
  ```

  Checkpoint: the printout ends with `(fc): Linear(in_features=512, out_features=1000)` — a final layer producing 1000 ImageNet scores you do not want.
- [ ] Swap the head: replace `model.fc` with a fresh `nn.Linear(512, num_classes)` for your classes. This new layer is the only randomly-initialized part of the whole network.
- [ ] **Stage 1 — feature extraction** (train the head only): freeze every pretrained parameter by setting `param.requires_grad = False` for all of them, then un-freeze just the new `model.fc`. **Freezing** means gradients are not computed and weights never update — the pretrained layers become a fixed feature extractor and only your new head learns. Build the optimizer over only the trainable parameters. Checkpoint: counting parameters with `requires_grad=True` gives a few thousand (512 × classes + biases), out of ~11 million total.
- [ ] Reuse your training loop from lesson 26 (train + validate per epoch, track loss and accuracy). Train stage 1 for 3-5 epochs with Adam, lr around `1e-3`. On CPU this is minutes; if lesson 24 gave you a GPU, use it. Checkpoint: validation accuracy is already **80-95%** after the first epoch or two — the pretrained features are doing almost all the work.
- [ ] **Stage 2 — fine-tuning**: unfreeze everything and continue training for 3-5 more epochs at a much lower learning rate (around `1e-4` or lower). The low LR matters: the pretrained weights are already good, and big updates would destroy them ("catastrophic forgetting"). Refinement: pass two parameter groups to the optimizer — pretrained layers at `1e-4`, your head at `1e-3` — which is the standard "lower LR for pretrained layers" trick. Checkpoint: **validation accuracy ≥ 0.90** (most personal datasets land at 92-99%).
- [ ] Save the winner: `torch.save` the model's `state_dict` plus the class-name list to `model.pt` whenever validation accuracy improves. Project 3 needs both.
- [ ] **The control experiment:** train the identical architecture from scratch — `models.resnet18(weights=None)` — same data, same number of total epochs, lr `1e-3`. Plot validation accuracy per epoch for both runs on one chart. Checkpoint: the from-scratch curve crawls (often stuck near 40-70% and unstable) while the pretrained curve jumps above 90% almost immediately. With ~100 images per class, 11 million random weights simply cannot learn what 1.2 million images already taught the pretrained ones.
- [ ] Look at every validation image the model gets wrong (loop over the val set, collect misclassified ones, show them in a grid with predicted vs true labels). Checkpoint: the mistakes make human sense — blurry shots, weird angles, genuinely ambiguous photos — or they reveal a data bug you can now fix.

<details><summary>Hints</summary>

- Freeze pattern: `for p in model.parameters(): p.requires_grad = False`, then replace `model.fc` — a newly-created layer has `requires_grad=True` by default, so the order does the work for you.
- Per-layer learning rates: `optim.Adam([{"params": backbone_params, "lr": 1e-4}, {"params": model.fc.parameters(), "lr": 1e-3}])`. `backbone_params` is every parameter not in `model.fc`.
- Remember `model.train()` before training and `model.eval()` plus `with torch.no_grad():` for validation — ResNet contains batch-norm layers that behave differently in each mode, and forgetting this is the classic "val accuracy is garbage but loss looks fine" bug.
- Stuck below 90%? Usual suspects in order: forgot ImageNet normalization; classes too similar or images too samey (add variety, not epochs); learning rate too high in stage 2 (accuracy *drops* when you unfreeze — lower it 10×).
- The PyTorch transfer learning tutorial (Resources) follows this exact recipe if you want to sanity-check your structure — read it *after* attempting, not before.

</details>

**Definition of done:** `model.pt` saved with ≥90% validation accuracy, plus one plot showing pretrained-vs-scratch validation curves where pretrained wins decisively.

## Project 3 — Ship it

**Goal:** Wrap your model in a web app and deploy it publicly for free. This is the milestone: a working AI product with a URL you can text to a friend.

**Milestones**

- [ ] Write `predict.py` with one function: it takes a PIL image, applies your *validation* transforms (never the augmentation ones — you want the model's honest opinion, not a random rotation of it), runs the model inside `torch.no_grad()`, applies `softmax` to turn raw scores into probabilities, and returns a dict of `{class_name: probability}` for the top 3. Load the model once at module import, not inside the function. Checkpoint: calling it on a photo of class A returns class A above 0.9.
- [ ] Build the UI in `app.py`. **Gradio** is a Python library that turns a function into a web interface — you declare inputs and outputs and it builds the page. The skeleton:

  ```python
  import gradio as gr

  demo = gr.Interface(
      fn=predict,                      # your function from predict.py
      inputs=gr.Image(type="pil"),
      outputs=gr.Label(num_top_classes=3),
      title="My First Shipped Model",
  )
  demo.launch()
  ```

  `gr.Label` renders a probability dict as confidence bars automatically. Run `python app.py` and open the printed `http://127.0.0.1:7860` in your Windows browser (WSL2 forwards it). Checkpoint: you upload a photo and see top-3 predictions with bars.
- [ ] Polish: add a description telling strangers what the model expects ("upload a photo of X, Y or Z"), and add 2-3 bundled sample images via the `examples=` argument so visitors can try it in one click.
- [ ] Create the deployment target. On [huggingface.co](https://huggingface.co) (free account — this is the one signup of the lesson), create a new **Space**: a free container that runs your app on their servers. Choose the **Gradio** SDK and CPU hardware (free). A Space is a git repo — clone it into a *separate* folder outside this repo.
- [ ] Add your files to the Space repo: `app.py`, `predict.py`, `model.pt`, sample images, and a `requirements.txt` listing `torch`, `torchvision`, `gradio`. Two deployment gotchas: the Space runs `app.py` on CPU, so load with `torch.load("model.pt", map_location="cpu")`; and if `model.pt` is over ~10 MB (it will be, ~45 MB), git needs Git LFS for it — `git lfs install` then `git lfs track "*.pt"` before adding (Spaces docs cover this; ask an AI assistant if LFS fights you).
- [ ] `git push`, then watch the build log on the Space page. First build takes a few minutes. Debug until the status turns green and the app loads. Checkpoint: your app works at `https://huggingface.co/spaces/<your-username>/<space-name>` — in a browser, on the public internet, on hardware you have never touched.
- [ ] **Send the link to a friend or family member.** Have them upload their own photo. When their reply comes back — that is the milestone. You have shipped a real, working AI product: you collected the data, trained the model, built the interface, and deployed it. Most people who "know ML" have never done all four.
- [ ] Commit your work folder to your learning repo (the model file and dataset can stay out via `.gitignore` — code and README in).

<details><summary>Hints</summary>

- Test locally in "deployment mode" first: fresh terminal, CPU-only load, `python app.py`. 90% of Space build failures are just a missing package in `requirements.txt` — the build log names it.
- If the Space builds but crashes at runtime, open its log tab: it is the same Python traceback you have read since lesson 02, bottom-up as always.
- Gradio changes its API between major versions. If an argument from a tutorial errors, trust the current docs at the Gradio quickstart (Resources) over any blog post.
- Keep the Space repo and your learning repo separate — nesting git repos causes pain.

</details>

**Definition of done:** A public Hugging Face Spaces URL that serves your model, and at least one other human has used it.

## Stretch goals

- **Confusion-matrix upgrade:** compute a confusion matrix (lesson 14) on your validation set and add it as an image on your Space page — real products document their failure modes.
- **Try bigger backbones:** swap ResNet18 for `resnet50` or a `convnext_tiny` from `torchvision.models`. Measure: accuracy gain vs model size vs prediction speed on CPU. Is bigger worth it for your data?
- **Webcam mode:** change the Gradio input to `gr.Image(sources=["webcam"])` and classify live snapshots — great for the hand-gestures dataset.
- **Grad-CAM:** search for "Grad-CAM PyTorch" and overlay a heatmap of *where* the model looked when deciding. Seeing your model stare at your cat's face (and not the sofa) is deeply satisfying — and sometimes reveals it learned the wrong thing.

## If you get stuck

- **Accuracy suspiciously perfect (100%)?** Check for leakage: the same or near-duplicate photo in both train and val (burst shots are the usual culprit). Fix the split script, not the model.
- **Accuracy stuck low?** Print one batch's tensor min/max/mean — if the mean is nowhere near 0, normalization is wrong or missing. Then check `dataset.classes` ordering matches the labels you think you are predicting.
- **Unfreezing made things worse?** Your stage-2 learning rate is too high. Drop it 10× — pretrained weights bruise easily.
- **Shape errors?** Same drill as always: print `tensor.shape` at every step. Images should be `[batch, 3, 224, 224]` entering the model.
- Standing advice: read the traceback bottom-up — the last lines name the real error; print shapes and values instead of guessing; ask an AI assistant for a **hint**, not a solution ("why might val accuracy drop when I unfreeze layers?"); and type all code yourself — muscle memory is the point.

## Resources

- [PyTorch transfer learning tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) — the official walkthrough of exactly this lesson's recipe; read after your own attempt to compare approaches.
- [Gradio quickstart](https://www.gradio.app/guides/quickstart) — everything for Project 3's UI, and the current API when tutorials disagree.
- [Hugging Face Spaces](https://huggingface.co/spaces) — where your demo lives; browse other people's Spaces for inspiration on what a good demo page looks like.
- fast.ai course, lessons 1-2 — a famous free course that starts with exactly this fine-tune-and-deploy workflow; a great parallel take (search for "fast.ai practical deep learning").

## Skills unlocked

- [ ] I can explain what a pretrained ImageNet model knows and why its features transfer to new image problems.
- [ ] I can collect, split and organize my own image dataset and load it with `ImageFolder` and augmentation transforms.
- [ ] I can swap the head of a pretrained torchvision model and freeze/unfreeze layers deliberately.
- [ ] I can run the two-stage recipe: train the head, then fine-tune everything at a lower learning rate.
- [ ] I can demonstrate with a controlled experiment why transfer learning beats training from scratch on small data.
- [ ] I can wrap a model in a Gradio interface and run it locally.
- [ ] I have deployed a model to Hugging Face Spaces and another person has used it.

## Next up

You have conquered images — next you teach neural networks to read and write, starting with a character-level language model built Karpathy-style: [28 · Language Models 101: makemore](../phase-4-nlp-transformers/28-language-models-makemore.md).
