# nn_eos_web — Project Log
**From Flask to Docker: Complete Development Record**


---

## Project Overview

Extension of the `nn_eos` neural network surrogate project into a deployable web application. The original `nn_eos` project trained neural networks to replace thermodynamic equation of state(EOS) lookups in a compressible CFD solver (FluTAS). This project packages that pipeline into a Flask web app with MLflow experiment tracking and Docker containerisation.

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
└── scripts/(part of original nn_eos, redundant now, will be removed or edited)
    ├── validate.py             ← used for validation of the neural network model
    |── export_weights.py       ← can be used to export saved model weights to a text file
                                
    
```

---

## Step 0 — Project Setup

### Create new repo from existing nn_eos


### Verify scaling and training work


## Step 1 — Creating the Web Application using Flask

### Install Flask

```bash
pip install flask
```

### Details of app.py

Three pages:
1. **`/` — Landing page** — describes the app, link to training form
2. **`/train` GET — Form page** — upload CSV, configure training parameters
3. **`/train` POST — Results page** — shows MAPE, loss curve, download button

### Flask pipeline

1. Receives the uploaded CSV and form parameters
2. Writes `config_scale.yaml` dynamically from form inputs using `yaml.dump`
3. Writes `config_train.yaml` dynamically from form inputs using `yaml.dump`
4. Calls `src/scaling.py` via `subprocess.run`
5. Calls `src/train.py` via `subprocess.run`
6. Reads `training_log.txt` for final MAPE and val_loss
7. Reads `history.csv` to plot loss curve
8. Renders results page


### Results page shows

- Final val_loss and val_MAPE (parsed from `training_log.txt`)
- Loss curve plot (val_loss and val_MAPE vs epoch, from `history.csv`)
- Download button for trained TF SavedModel (zipped)

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

Open browser at `http://127.0.0.0:5000` → Model training tab → nn_eos experiment.

### Important: MLflow 3.x uses SQLite not file store

MLflow 3.14 dropped file store support. Must use:
```python
mlflow.set_tracking_uri("sqlite:///mlflow.db")
```

**NOT** `file:./mlruns` — that causes errors in 3.x.



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
```

### requirements.txt

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



## Future Plans

### Data Analysis 
Add data analysis routines from, plotted from the uploaded csv.

### Parity plot 
Add actual vs predicted scatter plot to results page. 

### SQL model library 
SQLite database logging every training run — property, phase, architecture, MAPE, run_id, timestamp. Model library page showing all past runs, re-downloadable.

### CoolProp auto data generation
Replace CSV upload with fluid/property dropdown. App queries CoolProp to generate training data automatically. No CSV needed from user.

### AWS EC2 deployment 
Deploy Docker container to AWS EC2 free tier (12 months). 

### Hyperparameter tuning with KerasTuner
"Find best architecture" button — auto-searches architectures and learning rates, returns best model.



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
