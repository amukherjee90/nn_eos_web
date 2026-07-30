# nn_eos_web

A Flask web application for training neural network surrogates that replace expensive equation-of-state (EOS) lookups in CFD solvers, with MLflow experiment tracking and Docker containerisation.

## Background

Traditional EOS libraries are accurate but slow — they evaluate complex analytical functions to compute thermodynamic properties (density, speed of sound, enthalpy, etc.), and this evaluation cost slows down time-critical simulations. A neural network trained on EOS data can reproduce this accuracy at a fraction of the computational cost, making it suitable for integration into compressible CFD solvers such as FluTAS.

`nn_eos_web` packages this training pipeline — originally developed as the standalone `nn_eos` project — into a browser-based tool: upload a dataset, configure the network and training run, and download a ready-to-use TensorFlow SavedModel.

## Features

- **Upload & train** — upload a CSV of thermodynamic data (`P`, `T`, and target properties), pick a property/phase, configure the network architecture and hyperparameters, and train directly from the browser.
- **Configurable scaling** — per-column scaling methods (`minmax`, `standard`, `maxval`) written dynamically to `config_scale.yaml` from the form inputs.
- **Configurable architecture** — hidden layer sizes, activation (`relu`, `tanh`, `srelu`, `srelu3`), learning rate, epochs, and batch size, written dynamically to `config_train.yaml`.
- **Resume training** — continue training from the last saved model instead of starting from scratch.
- **Results dashboard** — final validation loss/MAPE, a val-loss and val-MAPE-vs-epoch plot, and a one-click download of the trained model as a zipped TensorFlow SavedModel.
- **MLflow tracking** — every run logs its parameters, metrics, and artifacts to a local MLflow instance (SQLite backend).
- **Dockerised** — runs anywhere with a single `docker run`, no local Python environment needed.

## Folder structure

```
nn_eos_web/
├── app.py                      # Flask application — main entry point
├── Dockerfile                  # Docker container definition
├── requirements.txt            # Python dependencies
├── config_scale.yaml           # Scaling config (written dynamically by Flask)
├── config_train.yaml           # Training config (written dynamically by Flask)
├── templates/
│   ├── index.html              # Landing page
│   ├── train.html              # Training form
│   ├── analysis.html           # Data analysis page
│   └── results.html            # Results page (MAPE, loss curve, download)
├── static/plots/                # Generated loss-curve / parity plots
├── uploads/                      # Temporary uploaded CSVs
├── logs/                        # Scaling / training subprocess logs
├── data/
│   ├── raw/                     # Input CSVs (e.g. liquid_ph2_qcc.csv, vapor_ph2_qcc.csv)
│   └── scaled/                  # Scaled CSVs + scaling params (generated, gitignored)
├── models/                      # Trained TF SavedModels + training logs (generated, gitignored)
├── mlruns/ , mlflow.db           # MLflow tracking store (generated, gitignored)
├── src/
│   ├── architecture.py          # build_model(), SReLU / SReLU3 activations
│   ├── scaling.py                # Raw CSV → scaled CSV + scaling params
│   ├── helpers.py                # Data / model loading and saving utilities
│   ├── train.py                  # Training script (plain MAE or derivative-loss modes)
│   └── visual.py                 # Plotting utilities
└── scripts/
    ├── validate.py                # Model validation
    ├── export_weights.py          # Export saved model weights to text
    └── archive/                   # Retired scripts from the original nn_eos project
```

## Requirements

- Python 3.10+
- See [`requirements.txt`](requirements.txt) — Flask, TensorFlow, pandas, NumPy, Matplotlib, PyYAML, scikit-learn, MLflow.

## Getting started

### Option 1 — Run locally

```bash
git clone <this-repo-url>
cd nn_eos_web

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

python app.py
```

The app is served at `http://localhost:5000`.

### Option 2 — Run with Docker

```bash
docker build -t nn-eos-web .
docker run -p 5000:5000 nn-eos-web
```

No local Python setup needed — everything runs inside the container.

## Usage

1. Open `http://localhost:5000` in a browser.
2. Go to **Train** and:
   - Upload a CSV with `P`, `T`, and target property columns.
   - Select the property (`rho`, `cs2`, `h`, `cp`) and phase (`liquid`, `vapor`).
   - Set the hidden layer sizes, activation, learning rate, epochs, batch size, and input scaling method.
   - Optionally check **Resume from last saved model**.
   - Submit to scale the data and train the model.
3. View the results page: final validation loss / MAPE, a loss-curve plot, and a **Download** button for the trained model (zipped TensorFlow SavedModel).
4. Go to **Analysis** page, upload the csv, and then check out line plot for pressure and temperature vs. property.

### Under the hood

Submitting the training form:

1. Saves the uploaded CSV to `data/raw/`.
2. Writes `config_scale.yaml` and `config_train.yaml` from the form inputs.
3. Runs `src/scaling.py` as a subprocess (logged to `logs/scaling.log`).
4. Runs `src/train.py` as a subprocess (logged to `logs/training.log`).
5. Parses `models/{property}_{phase}/training_log.txt` for the final MAPE/loss.
6. Plots `models/{property}_{phase}/history.csv` to `static/plots/`.
7. Renders the results page.

### Running the pipeline manually

```bash
# Scale a dataset
python src/scaling.py --eos ph2_qcc --phase liquid

# Train (settings read from config_train.yaml)
python src/train.py
python src/train.py --resume
```

### MLflow UI

Every training run logs its parameters, metrics, and artifacts to a local MLflow store:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open `http://127.0.0.1:5000` (or the port MLflow reports) and select the `nn_eos` experiment.

## Configuration

- **`config_scale.yaml`** — scaling method (`minmax`, `standard`, `maxval`) per data column.
- **`config_train.yaml`** — property, phase, EOS name, input columns, network architecture (`n_hidden`), activation, learning rate, epochs, batch size, and loss weights.

Both files are regenerated automatically from the web form on each training run, but can also be edited directly for manual/CLI runs.

## Ongoing work

- **Data analysis page** — exploring the uploaded CSV (distributions, ranges, correlations) before committing to a training run.

## Roadmap

- Parity (actual vs. predicted) scatter plot on the results page.
- SQLite-backed model library — a browsable, re-downloadable history of all past training runs.
- Automatic dataset generation via CoolProp (fluid/property dropdown instead of manual CSV upload).
- Polynomial regression as an alternative training method.
- Hyperparameter tuning with KerasTuner ("find best architecture" button).
- Deployment to AWS EC2.

## Acknowledgements

Built on top of the original `nn_eos` surrogate-modelling project, developed for use with the FluTAS compressible CFD solver.
