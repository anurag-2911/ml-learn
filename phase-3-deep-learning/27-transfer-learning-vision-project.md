# 27 · Transfer Learning: A Custom Image Classifier + Demo

**Phase 3 — Deep Learning** · Estimated time: 1 week · Prerequisites: [24 · PyTorch Fundamentals](24-pytorch-fundamentals.md), [25 · The Dark Arts of Training Deep Networks](25-training-deep-nets.md), [26 · Convolutional Neural Networks](26-cnns-computer-vision.md)

> **MILESTONE LESSON.** Lesson 26 trained CNNs on standard datasets that thousands of students have used before. This lesson trains a model on a dataset that nobody else has (personal photos of pets, plants or home cooking) and puts it on the public internet, where anyone with the link can use it. The trick that makes this possible with only ~100 photos per class is **transfer learning**: borrowing a network that someone else already trained on millions of images and gently adapting it to the new problem. The week ends with a first real, working AI product, shipped with its own URL.

## What this lesson builds

- **Project 1 — A personal dataset:** a folder of 2-5 image classes collected by hand (50-150 photos each), organized into `train/` and `val/`, loading cleanly through PyTorch's `ImageFolder` with augmentation.
- **Project 2 — A fine-tuned ResNet18:** a pretrained network with its final layer swapped for the dataset's classes, trained in two stages (head-only, then full fine-tune), hitting 90%+ validation accuracy in minutes, plus a from-scratch baseline that proves why transfer learning wins.
- **Project 3 — A live web demo:** a Gradio app (upload an image → top-3 predictions with confidence bars) behind a free public link that anyone can open, with a permanent home on Hugging Face Spaces as an optional extra.

## Concepts covered

- What a pretrained ImageNet model already knows (edges → textures → parts → objects) and why that knowledge transfers to personal photos.
- **Feature extraction vs fine-tuning**: freezing the pretrained layers vs unfreezing them, and when to do which.
- Small-data tricks: data augmentation, and using a lower learning rate for pretrained layers than for new ones.
- The `torchvision.models` and `torchvision.transforms` toolkits: loading pretrained weights and matching their expected preprocessing.
- Building a web UI in a few lines of Python with Gradio.
- Sharing a demo through a free public link, and what hosting it permanently on Hugging Face Spaces involves.

## Before starting

1. **Prerequisites check.** Lesson 26 covered building and training a CNN in PyTorch, using `DataLoader`, and plotting a training curve. Lesson 25 showed what learning rate and overfitting look like. If those skills are in place, go ahead.
2. **Install packages** (inside the venv at the repo root):

   ```bash
   cd ~/ml/ml-learn
   source .venv/bin/activate
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   pip install gradio
   ```

   (torch and torchvision are already installed from lesson 24. Their line is only there so that a fresh machine works too, and it leaves an existing install as it is, even a GPU build.) As in lesson 24, that line uses PyTorch's own package index, which has no gradio, so gradio gets a plain `pip install` line of its own. On an Intel Mac the torch line fails, because current PyTorch has no Intel Mac version: collect and sort the photos on the Mac, zip the folder (`zip -r data.zip data`), upload `data.zip` to a Google Colab notebook, unpack it there with `!unzip -q data.zip`, and run the code there, as in lesson 26 (in Colab, `demo.launch()` shows the app inside the notebook). Before closing the notebook, download `model.pt` with `from google.colab import files; files.download("model.pt")`, because Colab deletes the session's files when it ends.
3. **No dataset download this time: the dataset comes from personal photos.** Collecting it takes a phone and ~1-2 hours of photo taking or photo-library digging. Think now about which 2-5 things to classify.
4. **Accounts:** none are needed. Only the optional last part of Project 3, which hosts the demo permanently on Hugging Face Spaces, needs a Hugging Face account ([huggingface.co](https://huggingface.co) is the standard hub for sharing ML models and demos), and Project 3 explains what that costs.
5. **Create the work folder:**

   ```bash
   mkdir -p work/27-transfer-learning
   cd work/27-transfer-learning
   ```

   Keep the photos and the model out of git. The fork is public, a phone photo often records where it was taken (GPS data inside the file), and `model.pt` is about 45 MB. In this folder, run `printf 'data/\ndata.zip\n*.pt\n' > .gitignore` (`data.zip` is the photo archive made on the Intel Mac route). Checkpoint: `git check-ignore data/x.jpg model.pt` prints both names.

## Project 1 — A personal dataset

**Goal:** Collect and organize a real image dataset of 2-5 classes of personal interest, and load it into PyTorch with the right transforms. Data collection is half of real-world ML, and this project gives a first-hand feel for it.

**Milestones**

- [ ] Pick 2-5 classes of things that can be photographed now or already appear in existing photos. Good picks: pets (one class per animal), house plants, 3-4 home-cooked dishes, hand gestures (fist / palm / thumbs-up), types of objects on a desk. Pick things that *look* different: "cat vs dog" is easier than "two similar mugs". Write the class list in a `README.md` in the work folder.
- [ ] Collect **50-150 images per class**. Phone photos are perfect. Vary them deliberately: different angles, lighting, backgrounds, distances. Variety is what teaches the model to generalize instead of memorizing "my cat is always on the blue sofa". Checkpoint: `ls` shows one folder per class, and `ls CLASS-NAME | wc -l` reports 50+ files each.
- [ ] Copy the photos into the work folder and organize them like this (this exact layout is what `ImageFolder` expects: folder name = class label):

  ```
  data/
    train/
      class_a/   img001.jpg ...
      class_b/   ...
    val/
      class_a/   ...
      class_b/   ...
  ```

  Write a small Python script that randomly moves ~20% of each class into `val/` (use `random.sample` to pick the files and `shutil.move` to move them; both come with Python). Never split by hand, because hand-picking biases the split. Checkpoint: train and val folders exist, val has roughly 20% of each class, and no image appears in both.
- [ ] Write `dataset.py`: build two transform pipelines with `torchvision.transforms`. A **transform** is a preprocessing step applied to each image as it loads. For training: resize/crop to 224×224, random horizontal flip, small random rotation and color jitter (this is **data augmentation**: creating label-preserving variations so the model sees a "bigger" dataset), then `ToTensor()` and `Normalize` with the ImageNet means and stds (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`; pretrained models expect input scaled exactly the way they were trained). For validation: just resize, center-crop, tensor, normalize. There is no augmentation, because validation needs an honest, repeatable measurement.
- [ ] Load both folders with `torchvision.datasets.ImageFolder` and wrap them in `DataLoader`s (batch size 16-32). Checkpoint: printing `dataset.classes` shows the class names, and `len()` of each dataset matches the file counts.
- [ ] Visualize one training batch with matplotlib (undo the normalization before showing, or colors look psychedelic). Checkpoint: a grid of the collected photos with correct labels, some visibly flipped/rotated by augmentation.

<details><summary>Hints</summary>

- Getting photos into the work folder: Google Photos or a phone-to-computer transfer gets them onto the computer first. Make each class folder with `mkdir -p data/train/CLASS-NAME`. To drag photos in, open the work folder in the file manager: `open .` on macOS, `explorer.exe .` on Windows (WSL2), `xdg-open .` on Linux. Or copy them in the terminal: on macOS and Linux, `cp ~/Pictures/YOUR-FOLDER/* data/train/CLASS-NAME/`; on Windows (WSL2), Windows files are under `/mnt/c/Users/`, so `cp /mnt/c/Users/YOUR-WINDOWS-NAME/Pictures/YOUR-FOLDER/* data/train/CLASS-NAME/`. When OneDrive backs up the Pictures folder, the photos are in `/mnt/c/Users/YOUR-WINDOWS-NAME/OneDrive/Pictures/` instead. On macOS, Finder can leave a hidden `.DS_Store` file in each folder opened in it, so make the split script skip names that start with a dot. On Windows (WSL2), dragging downloaded photos in can add a file ending in `:Zone.Identifier` next to each one (the clutter from lesson 01); delete them with `rm data/train/*/*:Zone.Identifier` so that the file counts stay right.
- HEIC photos (the iPhone default) do not load with PIL, and `ImageFolder` skips them. Convert them to JPEG with the pillow-heif package, which works the same on every system: `pip install pillow-heif`, then call `register_heif_opener()` (from `pillow_heif`) once at the top of a small script. After that, `Image.open` can read `.heic` files, and `.save()` writes them out as `.jpg`. Or set the iPhone to shoot JPEG from now on: Settings → Camera → Formats → **Most Compatible**.
- To un-normalize for display: first `img = img.permute(1, 2, 0)`, because matplotlib wants height × width × channels, then `img = img * torch.tensor(std) + torch.tensor(mean)` (with the channels now last, the 3 values broadcast over the channel dimension, a lesson 05 skill), then `plt.imshow(img.clamp(0, 1))`.
- If `ImageFolder` errors on a corrupt file, find it by looping over the dataset in a try/except and printing the index that fails.

</details>

**Definition of done:** `python dataset.py` prints the classes and dataset sizes and saves a labeled batch grid image, with augmentation visibly working.

## Project 2 — Fine-tune ResNet18

**Goal:** Take ResNet18 pretrained on ImageNet, adapt it to the dataset's classes in two stages, and beat 90% validation accuracy in minutes of training. Then train the same architecture from scratch and watch it lose badly. The result is the whole argument for transfer learning in one plot.

**Milestones**

- [ ] Understand what is being borrowed. **ImageNet** is a dataset of ~1.2 million labeled photos in 1000 categories. Models pretrained on it have already learned general visual features: early layers detect edges and textures, middle layers detect parts like eyes and wheels, and late layers detect whole objects. Those early and middle features are useful for *any* photo problem, including this one. Load the model in `train.py`:

  ```python
  from torchvision import models
  model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
  print(model)
  ```

  Checkpoint: the last layer in the printout is `(fc): Linear(in_features=512, out_features=1000, bias=True)`, a final layer producing 1000 ImageNet scores that this project does not need.
- [ ] Swap the head: replace `model.fc` with a fresh `nn.Linear(512, num_classes)` for the dataset's classes. This new layer is the only randomly-initialized part of the whole network.
- [ ] **Stage 1: feature extraction.** Train the head only: freeze every pretrained parameter by setting `param.requires_grad = False` for all of them, then un-freeze just the new `model.fc`. **Freezing** means gradients are not computed and weights never update: the pretrained layers become a fixed feature extractor, and only the new head learns. Build the optimizer over only the trainable parameters. Checkpoint: counting parameters with `requires_grad=True` gives 513 × the number of classes (512 weights plus 1 bias per class, so 1,026 for 2 classes and 2,565 for 5), out of ~11 million total.
- [ ] Reuse the training loop from lesson 26 (train + validate per epoch, track loss and accuracy). Train stage 1 for 3-5 epochs with Adam, lr around `1e-3`. On CPU this takes minutes; if lesson 24's check showed a GPU (`cuda` on an NVIDIA card, `mps` on an Apple Silicon Mac), use it. Checkpoint: validation accuracy is already **80-95%** after the first epoch or two; the pretrained features are doing almost all the work.
- [ ] **Stage 2: fine-tuning.** Unfreeze everything and continue training for 3-5 more epochs at a much lower learning rate (around `1e-4` or lower). The low LR matters: the pretrained weights are already good, and big updates would destroy them ("catastrophic forgetting"). Refinement: pass two parameter groups to the optimizer (pretrained layers at `1e-4`, the new head at `1e-3`), which is the standard "lower LR for pretrained layers" trick. Checkpoint: **validation accuracy ≥ 0.90** (most personal datasets land at 92-99%).
- [ ] Save the winner: `torch.save` the model's `state_dict` plus the class-name list to `model.pt` whenever validation accuracy improves. Project 3 needs both.
- [ ] **The control experiment:** train the identical architecture from scratch (`models.resnet18(weights=None)`) with the same data, the same number of total epochs and lr `1e-3`. Plot validation accuracy per epoch for both runs on one chart. Checkpoint: the from-scratch curve crawls (often stuck near 40-70% and unstable) while the pretrained curve jumps above 90% almost immediately. With ~100 images per class, 11 million random weights simply cannot learn what 1.2 million images already taught the pretrained ones.
- [ ] Look at every validation image the model gets wrong (loop over the val set, collect misclassified ones, show them in a grid with predicted vs true labels). Checkpoint: the mistakes make human sense (blurry shots, weird angles, genuinely ambiguous photos), or they reveal a data bug that can now be fixed.

<details><summary>Hints</summary>

- Freeze pattern: `for p in model.parameters(): p.requires_grad = False`, then replace `model.fc`. A newly-created layer has `requires_grad=True` by default, so the order does the work.
- Per-layer learning rates: `optim.Adam([{"params": backbone_params, "lr": 1e-4}, {"params": model.fc.parameters(), "lr": 1e-3}])`. `backbone_params` is every parameter not in `model.fc`.
- Remember `model.train()` before training and `model.eval()` plus `with torch.no_grad():` for validation. ResNet contains batch-norm layers that behave differently in each mode, and forgetting this is the classic "val accuracy is garbage but loss looks fine" bug.
- Stuck below 90%? Usual suspects in order: forgot ImageNet normalization; classes too similar or images too samey (add variety, not epochs); learning rate too high in stage 2 (accuracy *drops* after unfreezing, so lower it 10×).
- The PyTorch transfer learning tutorial (Resources) covers both halves of this recipe as two separate runs (training only the head, and fine-tuning everything, both with SGD), so it can be used to sanity-check the code's structure. Read it *after* attempting, not before.

</details>

**Definition of done:** `model.pt` saved with ≥90% validation accuracy, plus one plot showing pretrained-vs-scratch validation curves where pretrained wins decisively.

## Project 3 — Ship it

**Goal:** Wrap the model in a web app and put it on the public internet for free. This is the milestone: a working AI product with a URL that can be texted to a friend.

**Milestones**

- [ ] Write `predict.py` with one function, `predict(image)`: it takes a PIL image, applies the *validation* transforms (never the augmentation ones, because the goal is the model's honest opinion, not a random rotation of it), runs the model inside `torch.no_grad()`, applies `softmax` to turn raw scores into probabilities, and returns a dict of `{class_name: probability}` for the top 3 (or for every class when there are fewer than 3; with `torch.topk`, use `k=min(3, len(classes))`). Load the model once at module import, not inside the function. Checkpoint: calling it on a photo of class A returns class A above 0.9.
- [ ] Build the UI in `app.py`. **Gradio** is a Python library that turns a function into a web interface: the code declares inputs and outputs, and Gradio builds the page. The skeleton:

  ```python
  import gradio as gr
  from predict import predict

  demo = gr.Interface(
      fn=predict,                      # the function from predict.py
      inputs=gr.Image(type="pil"),
      outputs=gr.Label(num_top_classes=3),
      title="My First Shipped Model",
  )
  demo.launch()
  ```

  `gr.Label` renders a probability dict as confidence bars automatically. Run `python app.py` and open the printed `http://127.0.0.1:7860` in a browser (on Windows, WSL2 forwards it automatically). Checkpoint: uploading a photo shows top-3 predictions with bars.
- [ ] Polish: add a description telling strangers what the model expects ("upload a photo of X, Y or Z"), and add 2-3 bundled sample images via the `examples=` argument so visitors can try it in one click. Make a `samples` folder (`mkdir samples`), then make each sample with `ImageOps.exif_transpose(Image.open(PATH)).convert("RGB").save("samples/NAME.jpg")` (after `from PIL import Image, ImageOps`); saving through PIL this way drops the EXIF data, including any GPS location, and `exif_transpose` first turns the photo upright, because the rotation a phone records is part of that EXIF data.
- [ ] **Go public with a share link.** In `app.py`, change the last line to `demo.launch(share=True)` and run `python app.py` again. Besides the local URL, Gradio now prints a public one, such as `https://823f0b90fe8b52f9c1.gradio.live`. The app still runs on this computer; Gradio only passes each visitor's request through to it. So the link works while `python app.py` is running and the computer is awake and online, it stops when the script stops, and it expires after at most 1 week (run the script again for a new link). No account is needed. Anyone who has the link can use the app, so send it only to people who are meant to try it. (In Colab the public link is printed too, and it works while the notebook cell is running.) Checkpoint: the public link opens on a phone that is on mobile data, not on the home Wi-Fi, and a photo uploaded there gets a prediction.
- [ ] **Send the link to a friend or family member** while `app.py` is still running. Have them upload their own photo. When their reply comes back, that is the milestone: a real, working AI product has been shipped. Getting there took four steps: collecting the data, training the model, building the interface and putting it online. Most people who "know ML" have never done all four.
- [ ] Commit the work folder to the learning repo. The `.gitignore` from Before starting keeps `data/`, `data.zip` and `model.pt` out; code, README and the sample images go in.

*Optional — a permanent address on Hugging Face Spaces*

A share link stops working when the script stops. A **Space** on [huggingface.co](https://huggingface.co) is a container that runs the app on Hugging Face's servers, at an address that stays up. It is no longer free for a new account: creating a Gradio Space needs the paid PRO plan ($9 per month when this was written). The one free route is **ZeroGPU** hardware, open to accounts that are at least 30 days old and have a verified email address (up to 2 Spaces). It needs the changes that the [ZeroGPU docs](https://huggingface.co/docs/hub/spaces-zerogpu) list: Python 3.12 instead of 3.13, a torch version from their supported list, and `@spaces.GPU` on the prediction function. (The message Gradio prints next to a share link still calls Spaces free; it is older than this change.) The three milestones below are written for a PRO account. Without one, skip them: the lesson is complete with the share link.

- [ ] Create the deployment target. On [huggingface.co](https://huggingface.co), sign up and subscribe to PRO, then create a new **Space**. Choose the **Gradio** SDK and the CPU Basic hardware, which adds no hourly cost. A Space is a git repo. Clone it into a *separate* folder outside this repo. (`share=True` can stay in `app.py`; a Space ignores it.)
- [ ] Add the files to the Space repo: `app.py`, `predict.py`, `model.pt`, sample images, and a `requirements.txt` listing `torch`, `torchvision`, `gradio`. Three deployment gotchas: the Space runs `app.py` on CPU, so load with `torch.load("model.pt", map_location="cpu")`; a Space runs Python 3.10 unless its `README.md` says otherwise, so add the line `python_version: "3.13"` between the two `---` lines at the top of that file, to match the venv; and Hugging Face rejects a push that contains binary files such as `model.pt` (~45 MB) or full-size phone photos unless Git LFS stores them, so run `git lfs install` and then `git lfs track "*.pt" "*.jpg" "*.jpeg" "*.png"` before adding, and add the updated `.gitattributes` together with the other files (the Spaces docs cover this; ask an AI assistant if LFS causes trouble). Git LFS is a separate program that does not come with git, so install it before those two commands:

  - **macOS:** `brew install git-lfs` if Homebrew is installed. On a Mac without it (an Intel Mac, or macOS 14 or older), download the Mac version that matches the chip (Apple Silicon or Intel) from [git-lfs.com](https://git-lfs.com), double-click the downloaded `.zip` if the browser has not unpacked it already, and run `sudo ~/Downloads/git-lfs-*/install.sh` (it asks for the Mac password).
  - **Windows (WSL2) and Linux:** `sudo apt install -y git-lfs` (other distributions have a `git-lfs` package too).
- [ ] Hugging Face does not accept the account password for git. Create a token at huggingface.co/settings/tokens (**Create new token**, type **Write**), and when `git push` asks for a password, paste the token instead (the username is the Hugging Face username). Never put the token in the remote URL or in any file in the repo; anyone who sees it can push to the account. Then `git push`, and watch the build log on the Space page. The first build takes a few minutes. Debug until the status turns green and the app loads. Checkpoint: the app works at `https://huggingface.co/spaces/YOUR-USERNAME/YOUR-SPACE-NAME` in a browser, on the public internet, running on remote hardware.

<details><summary>Hints</summary>

- No public URL after `share=True`, only a "Could not create share link" message? The first time, Gradio downloads a small helper program, and an antivirus or firewall can block that download. The message names the file, where to download it by hand, and the folder to put it in.
- Before pushing to a Space, test locally in "deployment mode" first: fresh terminal, CPU-only load, `python app.py`. 90% of Space build failures are just a missing package in `requirements.txt`, and the build log names it.
- If the Space builds but crashes at runtime, open its log tab: it shows the same kind of Python traceback seen since lesson 02, so read it bottom-up as always.
- Gradio changes its API between major versions. If an argument from a tutorial errors, trust the current docs at the Gradio quickstart (Resources) over any blog post.
- Keep the Space repo and the learning repo separate; nesting git repos causes trouble.

</details>

**Definition of done:** A public link (a Gradio share link or a Space URL) that serves the model, and at least one other human has used it.

## Stretch goals

- **Confusion-matrix upgrade:** compute a confusion matrix (as in lesson 26) on the validation set and add it as an image to the work folder's `README.md` (and to the Space page, if there is one). Real products document their failure modes.
- **Try bigger backbones:** swap ResNet18 for `resnet50` or a `convnext_tiny` from `torchvision.models`. Measure: accuracy gain vs model size vs prediction speed on CPU. Is bigger worth it for this data?
- **Webcam mode:** change the Gradio input to `gr.Image(sources=["webcam"])` and classify live snapshots. This is a good fit for the hand-gestures dataset.
- **Grad-CAM:** search for "Grad-CAM PyTorch" and overlay a heatmap of *where* the model looked when deciding. The heatmap can show the model looking at the cat's face (and not the sofa), and sometimes it reveals that the model learned the wrong thing.

## Getting unstuck

- **Accuracy suspiciously perfect (100%)?** Check for leakage: the same or near-duplicate photo in both train and val (burst shots are the usual culprit). Fix the split script, not the model.
- **Accuracy stuck low?** Print one batch's tensor min/max/mean. If the mean is nowhere near 0, normalization is wrong or missing. Then check that the `dataset.classes` ordering matches the labels the code assumes.
- **Unfreezing made things worse?** The stage-2 learning rate is too high. Drop it 10×; pretrained weights are easily damaged.
- **Shape errors?** Same drill as always: print `tensor.shape` at every step. Images should be `[batch, 3, 224, 224]` entering the model.
- Standing advice: read the traceback bottom-up, because the last lines name the real error; print shapes and values instead of guessing; ask an AI assistant for a **hint**, not a solution ("why might val accuracy drop when I unfreeze layers?"); and type all code by hand, because muscle memory is the point.

## Resources

- [PyTorch transfer learning tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) — the official walkthrough of fine-tuning and feature extraction with ResNet18, as two separate runs; read it after attempting the projects, to compare approaches.
- [Gradio quickstart](https://www.gradio.app/guides/quickstart) — everything for Project 3's UI, and the current API when tutorials disagree.
- [Gradio's guide to sharing an app](https://www.gradio.app/guides/sharing-your-app) — how share links work and what their limits are.
- [Hugging Face Spaces](https://huggingface.co/spaces) — where demos get a permanent home; browse other people's Spaces for inspiration on what a good demo page looks like.
- fast.ai course, lessons 1-2 — a famous free course that starts with exactly this fine-tune-and-deploy workflow; a good parallel take (search for "fast.ai practical deep learning").

## Skills unlocked

- [ ] I can explain what a pretrained ImageNet model knows and why its features transfer to new image problems.
- [ ] I can collect, split and organize an image dataset and load it with `ImageFolder` and augmentation transforms.
- [ ] I can swap the head of a pretrained torchvision model and freeze/unfreeze layers deliberately.
- [ ] I can run the two-stage recipe: train the head, then fine-tune everything at a lower learning rate.
- [ ] I can demonstrate with a controlled experiment why transfer learning beats training from scratch on small data.
- [ ] I can wrap a model in a Gradio interface and run it locally.
- [ ] I have put a model behind a public link, and another person has used it.

## Next up

With images covered, the next lesson teaches neural networks to read and write, starting with a character-level language model built Karpathy-style: [28 · Language Models 101: makemore](../phase-4-nlp-transformers/28-language-models-makemore.md).
