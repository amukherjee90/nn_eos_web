# nn_eos_web — Project Log
**From Flask to Docker: Complete Development Record**
*Started: July 14, 2026*

---

## Project Overview

Extension of the `nn_eos` neural network surrogate project into a deployable web application. The original `nn_eos` project trained neural networks to replace thermodynamic equation of state (EOS) lookups in a Fortran DNS solver (FluTAS). This project packages that pipeline into a Flask web app with MLflow experiment tracking and Docker containerisation.

**GitHub:** https://github.com/amukherjee90/nn_eos_web

---

## Folder Structure

```
nn_eos_web/
├── app.py                      ← Flask application — main entry point
├── Dockerfile                  ← Docker container definition
├── requirements.txt            ← Python dependencies (minimal)
├── config_scale.yaml           ← Scaling config (written dynamically by Flask)
├── config_train.yaml           ← Training config (written dynamically by Flask)
├── .gitignore                  ← Excludes mlruns/, mlflow.db, models/, data/scaled/
├── templates/
│   ├── index.html              ← Landing page
│   ├── train.html              ← Training form
│   └── results.html            ← Results page (MAPE, loss curve, download)
├── static/
│   └── plots/                  ← Loss curve and parity plot images
├── uploads/                    ← Temporary uploaded CSV files
├── logs/
│   ├── scaling.log             ← Scaling subprocess output
│   └── training.log            ← Training subprocess output
├── data/
│   ├── raw/                    ← Input CSV files (liquid_ph2_qcc.csv, vapor_ph2_qcc.csv)
│   └── scaled/                 ← Scaled CSVs and scaling params (gitignored)
├── models/                     ← Trained TF SavedModels + training logs (gitignored)
├── mlruns/                     ← MLflow run data (gitignored)
├── mlflow.db                   ← MLflow SQLite database (gitignored)
└── src/
    ├── architecture.py         ← build_model(), SReLU, SReLU3
    ├── scaling.py              ← Scale raw CSV → scaled CSV + scaling params
    ├── helpers.py              ← load_scaled_csv, load_scaling_params, save_model etc.
    ├── train.py                ← Training script (plain MAE + GradientTape modes)
    └── visual.py               ← Plotting utilities
```

---

## Step 0 — Project Setup

### Create new repo from existing nn_eos

```bash
cd /home/aritram/Desktop/eos_work/github_upload
mkdir nn_eos_web
cp -r nn_eos/src nn_eos_web/
cp -r nn_eos/scripts nn_eos_web/
cp nn_eos/config_scale.yaml nn_eos_web/
cp nn_eos/config_train.yaml nn_eos_web/
mkdir -p nn_eos_web/data/raw
cp nn_eos/data/raw/liquid_ph2_qcc.csv nn_eos_web/data/raw/
cp nn_eos/data/raw/vapor_ph2_qcc.csv nn_eos_web/data/raw/
```

### Verify scaling and training work

```bash
cd nn_eos_web
python src/scaling.py --eos ph2_qcc --phase liquid
python src/scaling.py --eos ph2_qcc --phase vapor
python src/train.py   # reads config_train.yaml
```

### Push to GitHub

```bash
git init
git add .
git commit -m "Initial setup — scaling and training pipeline"
git remote add origin git@github.com:amukherjee90/nn_eos_web.git
git branch -M main
git push -u origin main
```

---

## Step 1 — Flask Web Application

### Install Flask

```bash
pip install flask
```

### What was built

Three pages:
1. **`/` — Landing page** — describes the app, link to training form
2. **`/train` GET — Form page** — upload CSV, configure training parameters
3. **`/train` POST — Results page** — shows MAPE, loss curve, download button

### How Flask triggers the pipeline

Flask does NOT import `src/train.py` directly. Instead it:
1. Receives the uploaded CSV and form parameters
2. Writes `config_scale.yaml` dynamically from form inputs using `yaml.dump`
3. Writes `config_train.yaml` dynamically from form inputs using `yaml.dump`
4. Calls `src/scaling.py` via `subprocess.run`
5. Calls `src/train.py` via `subprocess.run`
6. Reads `training_log.txt` for final MAPE and val_loss
7. Reads `history.csv` to plot loss curve
8. Renders results page

### Key app.py logic

```python
# Write config dynamically
with open('config_train.yaml', 'w') as f:
    yaml.dump(train_cfg, f, default_flow_style=False)

# Run scaling via subprocess
result = subprocess.run(
    ['python', 'src/scaling.py', '--eos', 'ph2_qcc', '--phase', phase],
    stdout=log, stderr=log
)

# Run training via subprocess
result = subprocess.run(
    train_cmd,    # includes --resume flag if checked
    stdout=log, stderr=log
)
```

### Why subprocess.run?

`src/train.py` was written as a standalone script with `if __name__ == '__main__':`. Importing it directly would require restructuring. `subprocess.run` calls it exactly as if typed in the terminal — no changes to existing code needed.

### Results page shows

- Final val_loss and val_MAPE (parsed from `training_log.txt`)
- Loss curve plot (val_loss and val_MAPE vs epoch, from `history.csv`)
- Download button for trained TF SavedModel (zipped)

### Plot — last run only

`history.csv` appends across runs. To plot only the last run:

```python
if not resume:
    df = df.tail(epochs).reset_index(drop=True)
df['epoch'] = df.index + 1
```

When resuming: plot all epochs (shows full training history).
When fresh start: plot only the current run's epochs.

### history.csv — fresh vs append

In `src/train.py`:

```python
write_mode = 'a' if args.resume else 'w'
write_header = not hist_path.exists() or not args.resume
with open(hist_path, write_mode, newline='') as hf:
```

- Fresh start → overwrites history.csv
- Resume → appends to history.csv

### Download

```python
@app.route('/download/<prop>/<phase>')
def download(prop, phase):
    model_dir = f"models/{prop}_{phase}/saved_model"
    zip_path  = f"models/{prop}_{phase}/{prop}_{phase}_model"
    shutil.make_archive(zip_path, 'zip', model_dir)
    return send_file(f"{zip_path}.zip", as_attachment=True)
```

### Resume training

Checkbox in `train.html`. In `app.py`:

```python
resume = request.form.get('resume', 'false') == 'true'
train_cmd = ['python', 'src/train.py']
if resume:
    train_cmd.append('--resume')
```

`src/train.py` already supports `--resume` — loads existing SavedModel and continues training.

### Logs

Subprocess output redirected to files — not terminal:

```python
with open('logs/scaling.log', 'w') as log:
    result = subprocess.run([...], stdout=log, stderr=log)
with open('logs/training.log', 'w') as log:
    result = subprocess.run([...], stdout=log, stderr=log)
```

---

## Step 2 — MLflow Experiment Tracking

### Install

```bash
pip install mlflow
# Fix TF compatibility conflict:
pip install "numpy<2.0.0" "protobuf<5.0.0"
```

### What MLflow tracks

For every training run:
- **Parameters:** property, phase, n_hidden, activation, lr, epochs, batch_size, use_deriv_loss
- **Metrics:** val_loss, val_MAPE
- **Artifacts:** history.csv, trained TF model

### Changes to src/train.py

**Import at top (after sklearn import):**
```python
import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
```

**After config loading and print statements (inside `if __name__ == '__main__':`):**
```python
mlflow.set_experiment("nn_eos")
mlflow.start_run()
run_id = mlflow.active_run().info.run_id
print(f"\nMLflow run ID: {run_id}")
mlflow.log_param("property", property_name)
mlflow.log_param("phase", phase)
mlflow.log_param("n_hidden", str(n_hidden))
mlflow.log_param("activation", activation)
mlflow.log_param("lr", lr)
mlflow.log_param("epochs", epochs)
mlflow.log_param("batch_size", batch_size)
mlflow.log_param("use_deriv_loss", use_deriv_loss)
```

**At the very end:**
```python
mlflow.log_metric("val_loss", final_val_loss)
mlflow.log_metric("val_mape", final_val_mape)
mlflow.log_artifact(str(hist_path))
mlflow.tensorflow.log_model(model, name="model")
mlflow.end_run()
```

**run_id also written to training_log.txt** — connects the human-readable log to the MLflow run.

### Opening MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open browser at `http://127.0.0.1:5000` → Model training tab → nn_eos experiment.

### Important: MLflow 3.x uses SQLite not file store

MLflow 3.14 dropped file store support. Must use:
```python
mlflow.set_tracking_uri("sqlite:///mlflow.db")
```

**NOT** `file:./mlruns` — that causes errors in 3.x.

### Deleting an experiment

If you delete an experiment from the UI, MLflow soft-deletes it. You cannot create a new one with the same name. Fix:

```bash
rm mlflow.db   # delete the whole database and start fresh
```

### Model storage in MLflow 3.x

Models stored under `mlruns/1/models/m-{ID}/artifacts/data/model/` — separate from run folders. Each run's model stored independently. Load with:

```python
mlflow.tensorflow.load_model("models:/model/latest")
```

### .gitignore additions

```
mlruns/
mlflow.db
```

To remove already-tracked files from GitHub:
```bash
git rm -r --cached mlruns/
git rm -r --cached mlflow.db
git add .gitignore
git commit -m "Remove mlruns and mlflow.db from tracking"
git push
```

---

## Step 3 — Docker Containerisation

### Install Docker (Ubuntu)

```bash
sudo apt-get update
sudo apt-get install -y docker.io
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER
# Log out and log back in for group change to take effect
```

### requirements.txt (minimal)

```
flask==3.1.3
tensorflow==2.15.0
pandas==1.5.3
numpy==1.24.0
matplotlib==3.10.8
pyyaml==6.0.3
scikit-learn==1.7.1
mlflow==3.14.0
```

**Why minimal?** `pip freeze` captures the entire conda environment (hundreds of packages). Docker image would be enormous. Only list what the app actually imports.

### Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p uploads logs static/plots

EXPOSE 5000

CMD ["python", "app.py"]
```

**Key decisions:**
- `python:3.10-slim` — minimal Ubuntu base, smaller image
- `COPY requirements.txt` before `COPY . .` — Docker caches the pip install layer. If only code changes (not requirements), rebuild skips reinstalling packages — much faster.
- `--no-cache-dir` — don't store pip download cache inside image, keeps it smaller
- `EXPOSE 5000` — documents the port, doesn't open it. `-p 5000:5000` at runtime does the mapping.
- `host='0.0.0.0'` in `app.run()` — required for Flask to be accessible from outside the container

### app.py — host binding fix

```python
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
```

Without `host='0.0.0.0'`, Flask only listens on `127.0.0.1` inside the container — unreachable from host machine.

### Build and run

```bash
docker build -t nn-eos-web .
docker run -p 5000:5000 nn-eos-web
```

### Stop and remove container

```bash
docker ps                          # get container ID
docker stop CONTAINER_ID
docker rm CONTAINER_ID
```

### Development workflow

- **During development:** `python app.py` — no rebuild needed
- **Testing Docker:** `docker build` + `docker run`
- **Every code change requires Docker rebuild** — but fast due to layer caching if requirements unchanged

### Running without any Python environment

Once Docker image is built, anyone can run:
```bash
docker run -p 5000:5000 nn-eos-web
```
No conda, no pip install, no Python setup needed.

---

## What's on CV

**Project entry:**
*"Developed and deployed a Flask web app for neural network surrogate training with MLflow experiment tracking and Docker containerisation."*

**Skills added:**
- Flask
- MLflow — experiment tracking, hyperparameter logging, model artifact storage
- Docker — containerisation, single-command deployment
- End-to-end ML pipeline development (data upload → scaling → training → results → model download)
- subprocess management in Python

---

## Future Plans

### Parity plot (next addition)
Add actual vs predicted scatter plot to results page. Data available from `X_val`, `y_val`, `model.predict(X_val)` after training. Save as image in `src/train.py` alongside history.csv.

### SQL model library (Step 5 — later)
SQLite database logging every training run — property, phase, architecture, MAPE, run_id, timestamp. Model library page showing all past runs, re-downloadable.

### CoolProp auto data generation (Step 6 — later)
Replace CSV upload with fluid/property dropdown. App queries CoolProp to generate training data automatically. No CSV needed from user.

### AWS EC2 deployment (Step 7 — 2027)
Deploy Docker container to AWS EC2 free tier (12 months). Public URL for portfolio. Plan: create AWS account in early 2027, deploy for full year of job search.

### Hyperparameter tuning with KerasTuner (optional)
"Find best architecture" button — auto-searches architectures and learning rates, returns best model.

---

## Known Issues / Notes

- MLflow model registry ("Register" button in UI) — not needed for this use case, skip
- MLflow warning about missing signature on `log_model` — harmless, ignore
- `mlflow.db` gets created in whichever directory the script runs from — make sure to run from project root
- history.csv path is absolute in train.py — MLflow artifact logging uses `str(hist_path)` which must be accessible at log time
- Docker image is large (~2-3GB) due to TensorFlow — normal for ML apps
- Training inside Docker is CPU-only (no GPU passthrough configured) — sufficient for small [16,8,4] networks

---

## Commands Reference

```bash
# Run locally
python app.py

# Run scaling manually
python src/scaling.py --eos ph2_qcc --phase liquid

# Run training manually
python src/train.py
python src/train.py --resume

# MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db

# Docker build
docker build -t nn-eos-web .

# Docker run
docker run -p 5000:5000 nn-eos-web

# Docker stop
docker stop $(docker ps -q)

# Kill training subprocess if stuck
pkill -f train.py

# Git push (SSH)
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
git push

# Remove MLflow from Git tracking
git rm -r --cached mlruns/
git rm -r --cached mlflow.db
```
