import os
import yaml
import subprocess
import shutil
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, send_file
import mlflow

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'data/raw'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['GET', 'POST'])
def train():
    if request.method == 'GET':
        return render_template('train.html')

    # ── get form inputs ───────────────────────────────────────────
    csv_file   = request.files['csv_file']
    prop       = request.form['property']
    phase      = request.form['phase']
    n_hidden   = [int(x) for x in request.form['n_hidden'].split(',')]
    lr         = float(request.form['lr'])
    epochs     = int(request.form['epochs'])
    batch_size = int(request.form['batch_size'])
    activation = request.form['activation']
    scaling    = request.form['scaling']

    # ── save uploaded CSV ─────────────────────────────────────────
    csv_filename = f"{phase}_ph2_qcc.csv"
    csv_path = os.path.join(app.config['UPLOAD_FOLDER'], csv_filename)
    csv_file.save(csv_path)

    # ── write config_scale.yaml ───────────────────────────────────
    scale_cfg = {
        'ph2_qcc': {
            'T_K':  {'method': scaling},
            'P_Pa': {'method': scaling},
            'rho':  {'method': 'maxval'},
            'drdT': {'method': 'standard'},
            'drdP': {'method': 'standard'},
            'cs2':  {'method': 'maxval'},
            'h':    {'method': 'minmax'},
            'cp':   {'method': 'minmax'},
            'u':    {'method': 'minmax'},
            'gamma':{'method': 'standard'},
            's':    {'method': 'minmax'},
            'g':    {'method': 'minmax'},
        }
    }
    with open('config_scale.yaml', 'w') as f:
        yaml.dump(scale_cfg, f, default_flow_style=False)

    # ── write config_train.yaml ───────────────────────────────────
    train_cfg = {
        'property': prop,
        'phase': phase,
        'eos': 'ph2_qcc',
        'inputs': ['T_K', 'P_Pa'],
        'n_hidden': n_hidden,
        'activation': activation,
        'srelu_eps': 1.0,
        'lr': lr,
        'epochs': epochs,
        'batch_size': batch_size,
        'use_deriv_loss': False,
        'loss_weights': {'rho': 1.0, 'drdT': 0.01, 'drdP': 0.01}
    }
    with open('config_train.yaml', 'w') as f:
        yaml.dump(train_cfg, f, default_flow_style=False)

    # ── run scaling ───────────────────────────────────────────────
    with open('logs/scaling.log', 'w') as log:
        result = subprocess.run(
            ['python', 'src/scaling.py', '--eos', 'ph2_qcc', '--phase', phase],
            stdout=log, stderr=log
        )
    if result.returncode != 0:
        return "Scaling failed. Check logs/scaling.log"

    # ── run training ──────────────────────────────────────────────
    with open('logs/training.log', 'w') as log:
        result = subprocess.run(
            ['python', 'src/train.py'],
            stdout=log, stderr=log
        )
    if result.returncode != 0:
        return "Training failed. Check logs/training.log"

    # ── read final MAPE from training_log.txt ─────────────────────
    model_dir  = f"models/{prop}_{phase}"
    log_path   = f"{model_dir}/training_log.txt"
    final_mape = None
    final_loss = None
    with open(log_path) as f:
        for line in f:
            if 'final val_MAPE' in line:
                final_mape = line.split(':')[1].strip()
            if 'final val_loss' in line:
                final_loss = line.split(':')[1].strip()

    # ── plot loss curve — last run only ───────────────────────────
    history_path = f"{model_dir}/history.csv"
    df = pd.read_csv(history_path)
    df = df.tail(epochs).reset_index(drop=True)
    df['epoch'] = df.index + 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(df['epoch'], df['val_loss'])
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Val Loss')
    ax1.set_title('Validation Loss')

    ax2.plot(df['epoch'], df['val_mape'], color='orange')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Val MAPE (%)')
    ax2.set_title('Validation MAPE')

    plt.tight_layout()
    plot_path = f"static/plots/{prop}_{phase}_loss.png"
    plt.savefig(plot_path)
    plt.close()

    return render_template('results.html',
        prop=prop,
        phase=phase,
        final_mape=final_mape,
        final_loss=final_loss,
        plot_path=plot_path,
        model_dir=model_dir
    )

@app.route('/download/<prop>/<phase>')
def download(prop, phase):
    model_dir  = f"models/{prop}_{phase}/saved_model"
    zip_path   = f"models/{prop}_{phase}/{prop}_{phase}_model"
    shutil.make_archive(zip_path, 'zip', model_dir)
    return send_file(f"{zip_path}.zip", as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
