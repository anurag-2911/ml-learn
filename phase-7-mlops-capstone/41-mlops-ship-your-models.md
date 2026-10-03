# 41 · MLOps: Track, Serve, Containerize, Deploy

**Phase 7 — MLOps and Capstone** · Estimated time: 2 weeks · Prerequisites: [20 · Capstone: End-to-End ML Project](../phase-2-classical-ml/20-end-to-end-ml-project.md), [26 · Convolutional Neural Networks](../phase-3-deep-learning/26-cnns-computer-vision.md), [27 · Transfer Learning: A Custom Image Classifier + Demo](../phase-3-deep-learning/27-transfer-learning-vision-project.md)

> You have trained dozens of models by now — and every single one of them dies when you close the notebook. A model that only you can run, on your machine, with settings you half-remember, is not a product; it is a science experiment. **MLOps** (machine learning operations) is the craft of turning "a notebook that worked once" into "a service anyone can use": tracked experiments, a real API, a container that runs anywhere, and a public demo. This lesson builds the minimum viable MLOps stack, and it is exactly what will make your lesson-42 capstone shippable instead of just impressive-on-your-laptop.

## What you will build

- **Project 1 — Instrumented training runs:** your lesson-26 CIFAR CNN (or lesson-20 tabular model) retrained 4+ times with every parameter, metric and model file logged to MLflow or W&B, plus a screenshot of the comparison dashboard.
- **Project 2 — Model as a service:** a FastAPI app with a validated `/predict` endpoint and a `/health` endpoint, a pytest test suite, and a Docker image you can `curl` from outside the container.
- **Project 3 — Public demo + drift drill:** a Gradio demo of your model live on Hugging Face Spaces, and a `drift_check.py` script that flags shifted feature distributions with one plot.

## Concepts you will learn by doing

- Why notebooks are not products: reproducibility, and what it takes to get it
- Experiment tracking: logging params, metrics and artifacts so no result is ever lost
- Model serialization done right (what to save, what never to pickle-and-pray)
- Serving: a FastAPI endpoint with pydantic input validation
- Docker: packaging your API into a container that runs anywhere
- Testing an ML service with pytest: schema checks and sanity predictions
- Monitoring, in miniature: detecting data drift in one plot
- The deploy menu: free demo hosting (HF Spaces) vs renting a real VM

## Before you start

Check you have:

- A trained model from earlier lessons: the CIFAR CNN from lesson 26 **or** the tabular model from lesson 20 (you need the tabular one for the drift drill in Project 3 either way).
- Your Hugging Face account from lesson 27 (Spaces needs it; it is free).
- Git working in your repo — Project 3's stretch goal uses GitHub.

Install the tools inside your repo-root venv:

```bash
cd ~/ml/ml-learn
source .venv/bin/activate
pip install mlflow fastapi "uvicorn[standard]" pydantic pytest httpx gradio joblib
```

Install Docker:

- **macOS:** download **Docker Desktop for Mac** from https://docs.docker.com/desktop/setup/install/mac-install/ (pick the Apple silicon or Intel chip version to match your Mac), drag Docker into Applications, open it and accept the terms, then open a new Terminal window. Docker Desktop needs macOS 14 or newer; on an older Mac, use the Codespace route below.
- **Windows (WSL2):** install **Docker Desktop for Windows** from https://docs.docker.com/desktop/setup/install/windows-install/ on the Windows side, not inside Ubuntu; if the installer shows **Use WSL 2 instead of Hyper-V**, make sure it is ticked. Start Docker Desktop from the Start menu and accept the terms. Then, in Docker Desktop → Settings → Resources → WSL integration, make sure integration is on for your Ubuntu distro (it is on by default for your default distro), click **Apply**, and close and reopen your Ubuntu terminal. The WSL2 backend guide, https://docs.docker.com/desktop/features/wsl/, has the details.
- **Linux:** install **Docker Engine** (not Docker Desktop) by following "Install using the apt repository" at https://docs.docker.com/engine/install/ubuntu/ (other distributions: https://docs.docker.com/engine/install/). Then add yourself to the `docker` group so you can run `docker` without `sudo` (membership gives root-level control through Docker, which is fine on your own computer), and log out and back in, or restart, for it to take effect:

  ```bash
  sudo usermod -aG docker $USER
  ```

No computer that can run Docker, such as a Mac on macOS 13 or older? Do the Docker milestones in a GitHub Codespace, a Linux machine in the cloud with Docker preinstalled and a free monthly allowance: commit and push your work, then on your fork's GitHub page click **Code**, open the **Codespaces** tab, create a codespace on `main`, and run the Docker commands in its terminal.

Verify in your terminal:

```bash
docker run hello-world
```

If that prints a welcome message, Docker works. Then create your work folder:

```bash
mkdir -p work/41-mlops-ship-your-models
cd work/41-mlops-ship-your-models
```

## Project 1 — Instrument a training run

**Goal:** Retrain a model you already built, but this time log every parameter, metric and output file to an experiment tracker — then run 4+ variations and compare them on a dashboard. Never again wonder "which settings got 82%?"

**Milestones**

- [ ] Copy your lesson-26 CIFAR training script (or lesson-20 training code) into `work/41-mlops-ship-your-models/train.py` as a plain Python script — no notebook. Make the key hyperparameters (learning rate, batch size, epochs) command-line arguments using `argparse` (Python's built-in library for reading command-line options). Checkpoint: `python train.py --lr 0.001 --epochs 2` runs a short training and prints a final accuracy.
- [ ] Write down, honestly, everything someone would need to reproduce your best lesson-26 result: exact code version, hyperparameters, random seed, library versions, data version. Notice how much of it lives only in your memory. That gap is why **reproducibility** — the ability to rerun an experiment and get the same result — is the first problem MLOps solves.
- [ ] Pick your tracker: **MLflow** (runs locally, nothing leaves your machine — https://mlflow.org) or **Weights & Biases** (hosted dashboard, free tier, needs an account — https://wandb.ai). The rest of the milestones assume MLflow; the W&B API is nearly one-to-one.
- [ ] Add tracking to `train.py`: a **param** is an input you chose (learning rate), a **metric** is a measured number (loss, accuracy), an **artifact** is an output file (the model, a plot). Skeleton:

  ```python
  import mlflow

  mlflow.set_experiment("cifar-cnn")
  with mlflow.start_run():
      mlflow.log_param("lr", args.lr)
      # ... inside your epoch loop:
      mlflow.log_metric("val_accuracy", val_acc, step=epoch)
  ```

  Checkpoint: after one run, a `mlruns/` folder exists next to your script.
- [ ] Serialize the model correctly and log it as an artifact. **Serialization** means saving a live Python object to a file. For PyTorch save `model.state_dict()` (just the weights — rebuild the architecture in code when loading), not the whole model object; for scikit-learn use `joblib.dump(pipeline, "model.joblib")` and save the *whole pipeline* including preprocessing, not just the estimator. Log the file with `mlflow.log_artifact(...)`. Checkpoint: the saved file loads in a fresh Python session and predicts on one example.
- [ ] Launch the dashboard with `mlflow ui --port 5001` and open http://localhost:5001 in your browser (on Windows, WSL2 forwards localhost to Windows automatically). MLflow's default port, 5000, is used by AirPlay on Macs, so this lesson uses 5001 everywhere. Checkpoint: your run appears with its params and a metric curve.
- [ ] Run at least 4 experiments varying something meaningful — e.g. lr ∈ {0.01, 0.001}, batch size ∈ {32, 128}. Select all runs in the UI and hit Compare. Checkpoint: a chart of val_accuracy across runs, and you can point at the winner and state its exact settings.
- [ ] Screenshot the comparison view and save it as `experiments.png` in your work folder. Write 3 sentences in a `NOTES.md`: which config won, by how much, and one hypothesis why.

<details><summary>Hints</summary>

- Training too slow for 4 runs? Shrink the job: 2–3 epochs and/or a 10% subset of CIFAR is fine. You are practicing tracking, not chasing accuracy.
- `mlflow ui` must be run from the directory that contains `mlruns/`, otherwise the dashboard looks empty.
- Log the metric *inside* the epoch loop with `step=epoch` to get curves instead of single points.
- W&B instead? `wandb.init(config=...)`, `wandb.log({...})`, and the dashboard is on their website after `wandb login`.

</details>

**Definition of done:** 4+ tracked runs visible on one comparison dashboard, a serialized model artifact that loads and predicts in a fresh session, and a screenshot proving it.

## Project 2 — Model as a service

**Goal:** Wrap your model in a web API — a program other programs can call over the network — then prove it works with automated tests, and package the whole thing into a Docker container.

**Which model to serve:** use the lesson-20 tabular model here, even if Project 1 tracked the CNN — the CNN was for tracking practice, and every milestone below (the pydantic schema, the JSON curl call) assumes tabular input: a few named numbers per request. Serving the CNN instead is a stretch option; images travel as file uploads, not JSON — see the `UploadFile` hint below.

**Milestones**

*Stage 1 — freeze the environment*

- [ ] Create a fresh folder `work/41-mlops-ship-your-models/service/` with the serialized model you are serving inside (retrain and serialize the lesson-20 model here if Project 1 tracked the CNN). Write `requirements.txt` listing only what serving needs (fastapi, uvicorn, pydantic, plus torch or scikit-learn/joblib) — then pin versions using the ones `pip freeze` reports. Pinning means writing `fastapi==...` so the exact same versions install anywhere. Checkpoint: `pip install -r requirements.txt` succeeds in a brand-new throwaway venv.

*Stage 2 — the API works*

- [ ] Build `app.py` with FastAPI (https://fastapi.tiangolo.com). Define the input schema with **pydantic** — a library that declares what valid input looks like and rejects everything else automatically. Starter shape (fill in your own fields and prediction logic):

  ```python
  from fastapi import FastAPI
  from pydantic import BaseModel, Field

  class PredictRequest(BaseModel):
      # one field per model input, e.g.:
      age: float = Field(ge=0, le=120)

  app = FastAPI()
  # load your model ONCE here, at startup — not inside the endpoint

  @app.get("/health")
  def health():
      return {"status": "ok"}

  @app.post("/predict")
  def predict(req: PredictRequest):
      ...  # run the model, return {"prediction": ..., "score": ...}
  ```

- [ ] Run it with `uvicorn app:app --reload` and open http://localhost:8000/docs — FastAPI auto-generates an interactive page for your API. Checkpoint: `/health` returns `{"status":"ok"}` and a `/predict` call from the docs page returns a JSON prediction.
- [ ] Hit it from the terminal:

  ```bash
  curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"age": 34, ...}'
  ```

  Then send garbage (a string where a number belongs, a missing field). Checkpoint: valid input returns a prediction; garbage returns HTTP 422 with an error message you did not have to write — that is pydantic validation working.

*Stage 3 — tests pass*

- [ ] Write `test_app.py` using pytest and FastAPI's `TestClient` (which calls your app in-process, no server needed). Test at least: `/health` returns 200; `/predict` on one known-good example returns 200 with the fields you expect; an invalid payload returns 422; and one **sanity prediction** — an input where you know roughly what the answer must be (e.g. a clearly-survived passenger, an obvious digit) actually gets it. Checkpoint: `pytest -v` shows 4+ tests passing.

*Stage 4 — the container works*

- [ ] A **container** is a lightweight box holding your app plus its entire environment — Python version, libraries, files — so it runs identically on any machine with Docker. Write a `Dockerfile` (the recipe for building the box). Skeleton:

  ```dockerfile
  FROM python:3.13-slim
  WORKDIR /app
  COPY requirements.txt .
  RUN pip install --no-cache-dir -r requirements.txt
  COPY . .
  CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
  ```

- [ ] Build and run it, mapping the container's port to your machine:

  ```bash
  docker build -t ml-service .
  docker run -p 8000:8000 ml-service
  ```

  Checkpoint: `docker build` finishes without errors and `docker ps` (in a second terminal) shows the container running.
- [ ] From a second terminal — outside the container — repeat your curl calls against http://localhost:8000. Checkpoint: same predictions as Stage 2, but now served from inside the box. Your model officially runs somewhere other than "your setup".

<details><summary>Hints</summary>

- Loading the model at import time (module top level) is correct here: it happens once per container start, not once per request.
- Torch makes images huge. For a tabular model the slim image stays small; for the CNN, expect a multi-GB image and don't worry about it yet — or serve the tabular model instead. If you do serve the CNN, put the line `--extra-index-url https://download.pytorch.org/whl/cpu` at the top of `requirements.txt` so the container gets the CPU-only PyTorch build: on Linux and WSL, `pip freeze` writes that build's version with `+cpu` at the end, which pip can only find on that index.
- Determined to serve the CNN? Swap the pydantic schema for FastAPI's file-upload pattern: `from fastapi import File, UploadFile`, then `@app.post("/predict")` with `async def predict(file: UploadFile = File(...))`, read the bytes with `await file.read()`, open them with PIL (`Image.open(io.BytesIO(data))`), apply your test-time transforms and predict. Test it with `curl -F "file=@some_image.png" http://localhost:8000/predict` — note `-F` (a form upload), not a JSON body.
- `--host 0.0.0.0` in the CMD matters: it makes the server listen on all network interfaces so traffic from outside the container can reach it. The default `127.0.0.1` only listens inside.
- Container builds but curl hangs? Check the `-p 8000:8000` port mapping and that the CMD actually starts uvicorn (`docker logs CONTAINER-ID`, with the ID from `docker ps -a`).

</details>

**Definition of done:** `pytest -v` green, and a curl from outside the running container returns a valid JSON prediction.

## Project 3 — Public demo + drift drill

**Goal:** Put a demo of your model on the public internet, then practice the monitoring skill every production ML system needs: noticing when incoming data stops looking like training data.

**Milestones**

*Part A — public demo*

- [ ] Build `demo.py`: a Gradio interface (the same library you used in lesson 27) that takes your model's inputs via simple widgets and shows the prediction. Checkpoint: `python demo.py` opens a working local demo in your browser.
- [ ] Deploy it to Hugging Face Spaces (free, needs your HF account): create a new Space at https://huggingface.co, choose the Gradio SDK, and push `demo.py`, your model file and a `requirements.txt` to it with git, exactly as you did in lesson 27. Before you push, edit the settings block between the `---` lines at the top of the Space's `README.md`: set `app_file: demo.py`, because the Space runs the file named there, and add the line `python_version: "3.13"`, because Spaces use Python 3.10 unless told otherwise, which is too old for the package versions in your venv. Checkpoint: the public URL works — send it to a friend and have them get a prediction on their phone.

*Part B — drift drill*

- [ ] **Data drift** is when the data a deployed model receives shifts away from the data it was trained on — the silent killer of production models, because the service keeps returning confident answers that slowly go wrong. Take your lesson-20 tabular training set and create a drifted copy in pandas: add an offset to one numeric feature (e.g. everyone is suddenly 15 years older), scale another (incomes ×3), and leave the rest untouched.
- [ ] Write `drift_check.py`: for each numeric feature, compare new-data mean vs training mean, in units of training standard deviation:

  ```python
  drift_score = abs(new_mean - train_mean) / train_std
  ```

  Flag any feature with a score above a threshold (start at 0.5). Checkpoint: run it on an *undrifted* sample of training data — zero flags; run it on your drifted copy — exactly the features you tampered with get flagged.
- [ ] Make it one plot: a horizontal bar chart of drift score per feature, threshold drawn as a vertical line, flagged bars in a different color. Save as `drift.png`. Checkpoint: anyone can look at the plot for 5 seconds and name the drifted features.
- [ ] In `NOTES.md`, write 3 sentences: what a real system would do when this alarm fires (retrain? investigate the data source? roll back?), and why checking predictions alone would not have caught it.

<details><summary>Hints</summary>

- Spaces builds fail most often on `requirements.txt`: it must list every import your `demo.py` makes, including joblib or torch.
- Keep the drift math honest: compute train mean/std once from the training set and save them (e.g. to JSON) — the check must not peek at the new data to define "normal".
- Mean-shift is the simplest drift signal and misses some shifts (e.g. variance changes with equal means). Noticing that limitation is part of the lesson — put it in your notes.

</details>

**Definition of done:** a public HF Spaces URL anyone can use, plus a drift script and plot that flag your simulated shift and stay quiet on clean data.

## Stretch goals

- **CI on push:** add a GitHub Action — a small YAML workflow in `.github/workflows/` that GitHub runs automatically on every push — that installs your service's requirements and runs `pytest`. This is **continuous integration**: your tests become a gate, not a chore. Search GitHub's docs for "Python application workflow" for the template, and set its `python-version` to "3.13" to match your venv (the template's older Python cannot install the versions you pinned).
- **Batch endpoint:** add `/predict_batch` accepting a list of inputs, with pydantic validating each item, and a test proving it.
- **Latency logging:** time each `/predict` call and log it; report p50 and p95 latency (the median and the 95th-percentile — the "slow tail" users actually feel) over 100 curl requests.
- **Registry push:** push your Docker image to Docker Hub (free account) and pull-and-run it on any other machine — the "runs anywhere" promise, cashed in. Images are built for your computer's processor type, so one built on an Apple Silicon Mac may not run on an Intel or AMD machine: add `--platform linux/amd64,linux/arm64` to `docker build` to build for both.

## If you get stuck

- **Docker permission or "cannot connect to the Docker daemon" errors:** on **macOS**, Docker Desktop must be running: open it from Applications and wait until it has started. On **Windows (WSL2)**, Docker Desktop must be running on Windows, and WSL integration enabled for your distro (Settings → Resources → WSL integration). Restart the Ubuntu terminal after changing it. On **Linux**, "permission denied" means you are not in the `docker` group yet, or have not logged out and back in since joining it; "cannot connect" means the Docker service is stopped, and `sudo systemctl start docker` starts it.
- **Port confusion:** "connection refused" usually means nothing is listening (server not started, wrong port); a hang usually means a mapping problem (`-p host:container`). `docker logs CONTAINER-ID` (the ID is the first column of `docker ps -a`) shows what the app inside actually did.
- **422 errors you didn't expect:** read the response body — pydantic tells you exactly which field failed and why. That error message is the feature you built.
- Standing advice: read tracebacks bottom-up — the last line names the error; print shapes and values before the crash line; ask an AI assistant for a HINT, not a solution; and type every line yourself — muscle memory is the curriculum.

## Resources

- FastAPI docs — https://fastapi.tiangolo.com — tutorial-style docs for the API and pydantic validation; the "First Steps" section covers everything Project 2 needs.
- MLflow — https://mlflow.org — the local experiment tracker for Project 1; see its Tracking quickstart.
- Weights & Biases — https://wandb.ai — the hosted alternative tracker (free tier, account required).
- Get Docker — https://docs.docker.com/get-started/get-docker/ — Docker's install guides for macOS, Windows and Linux (on Linux, follow its link to Docker Engine); on Windows, also the WSL2 backend guide at https://docs.docker.com/desktop/features/wsl/.
- Hugging Face — https://huggingface.co — where your Space lives; their Spaces docs cover the Gradio SDK setup.
- Gradio docs — search for "Gradio documentation" — reference for demo widgets beyond what lesson 27 used.

## Skills unlocked

- [ ] I can explain why a notebook alone is not reproducible, and list what a full experiment record contains.
- [ ] I can instrument a training script with an experiment tracker and compare runs on a dashboard instead of in my memory.
- [ ] I can serialize a model correctly (weights/pipeline, versions pinned) so it loads and predicts on another machine.
- [ ] I can wrap a model in a FastAPI service with validated inputs and a health check.
- [ ] I can write pytest tests for an ML service, including a sanity-prediction test.
- [ ] I can containerize a service with Docker and call it from outside the container.
- [ ] I can deploy a free public demo on Hugging Face Spaces.
- [ ] I can detect feature-mean drift against a saved training baseline and explain what to do when the alarm fires.

## Next up

You now own the full pipeline — data to model to tracked experiment to tested, containerized, deployed service — so it is time to put it all together on a project that is entirely yours: [42 · Capstone: Your Hero Project (and What Comes Next)](42-capstone-and-beyond.md).
