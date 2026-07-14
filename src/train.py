"""
scripts/train.py
Train a single-output NN model for nn_eos.

Usage:
    python scripts/train.py
    python scripts/train.py --resume

All settings (property, phase, architecture, hyperparameters) are read from
config_train.yaml. Edit the yaml to switch between models.

Output:
    models/{property}_{phase}/saved_model/
    models/{property}_{phase}/training_log.txt   (appended each run)
"""

import sys
import argparse
import numpy as np
import yaml
import tensorflow as tf
import mlflow
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split


sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.architecture import build_model, SReLU, SReLU3
from src.helpers import (load_scaled_csv, load_scaling_params,
                         compute_deriv_scale_factors,
                         save_model, load_model, get_project_root)


def plain_train(model, X_train, y_train, X_val, y_val, epochs, batch_size,
                callbacks=None):
    """Plain model.fit() training with MAE loss."""
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    return history


def deriv_train(model, X_train, y_train, X_val, y_val,
                deriv_train_dict, deriv_val_dict,
                loss_weights, deriv_factors,
                lr, epochs, batch_size,
                model_name, project_root):
    """
    GradientTape training loop with derivative loss.
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    mae = tf.keras.losses.MeanAbsoluteError()

    X_train_tf = tf.constant(X_train, dtype=tf.float32)
    y_train_tf = tf.constant(y_train.reshape(-1, 1), dtype=tf.float32)
    X_val_tf   = tf.constant(X_val,   dtype=tf.float32)
    y_val_tf   = tf.constant(y_val.reshape(-1, 1),   dtype=tf.float32)

    # identify derivative columns
    prop_keys = {'rho', 'cs2', 'h', 'u', 's', 'g', 'gamma'}
    deriv_cols = [k for k in loss_weights if k not in prop_keys]

    deriv_weights = [float(loss_weights[col]) for col in deriv_cols]
    prop_col = next(k for k in loss_weights if k in prop_keys)
    w_out = float(loss_weights[prop_col])

    print(f"\n  GradientTape training: {epochs} epochs, lr={lr}")
    print(f"  Loss weights: {loss_weights}")
    
    w_deriv_tf      = [tf.constant(w, dtype=tf.float32) for w in deriv_weights]
    factor_deriv_tf = [tf.constant(f, dtype=tf.float32)
                       for f in deriv_factors[:len(deriv_cols)]]
    w_out_tf        = tf.constant(w_out, dtype=tf.float32)

    # Include derivative targets in dataset to keep alignment correct
    deriv_arrays = [deriv_train_dict[col].reshape(-1, 1).astype(np.float32)
                    for col in deriv_cols]

    dataset_tensors = (X_train_tf, y_train_tf) + tuple(
        tf.constant(a) for a in deriv_arrays)
    dataset = tf.data.Dataset.from_tensor_slices(dataset_tensors) \
                .batch(batch_size).prefetch(tf.data.AUTOTUNE)

    @tf.function
    def train_step(X_batch, y_batch, deriv_batches):
        X_tensor = tf.identity(tf.cast(X_batch, tf.float32))
        with tf.GradientTape() as outer:
            with tf.GradientTape(watch_accessed_variables=False) as inner:
                inner.watch(X_tensor)
                y_pred = model(X_tensor, training=True)
            dy_dX = inner.gradient(y_pred, X_tensor)
            loss = w_out_tf * mae(y_batch, y_pred)
            for i in range(len(deriv_cols)):
                pred_deriv = tf.expand_dims(dy_dX[:, i], 1)
                loss += w_deriv_tf[i] * mae(deriv_batches[i], pred_deriv)
        grads = outer.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

    def diagnose_step(X_batch, y_batch, deriv_batches):
        """Read-only diagnostic — no weight update."""
        X_tensor = tf.identity(tf.cast(X_batch, tf.float32))
        with tf.GradientTape(watch_accessed_variables=False) as inner:
            inner.watch(X_tensor)
            y_pred = model(X_tensor, training=False)
        dy_dX = inner.gradient(y_pred, X_tensor)
        val_loss_only = w_out_tf * mae(y_batch, y_pred)
        loss = val_loss_only
        for i in range(len(deriv_cols)):
            pred_deriv = tf.expand_dims(dy_dX[:, i], 1)
            deriv_loss_i = w_deriv_tf[i] * mae(deriv_batches[i], pred_deriv)
            print(f"    deriv_col={deriv_cols[i]}"
                  f"  pred_deriv mean={tf.reduce_mean(tf.abs(pred_deriv)).numpy():.6f}"
                  f"  target mean={tf.reduce_mean(tf.abs(deriv_batches[i])).numpy():.6f}"
                  f"  deriv_loss={deriv_loss_i.numpy():.6f}")
            loss += deriv_loss_i
        print(f"    val_loss_only={val_loss_only.numpy():.6f}"
              f"  total_loss={loss.numpy():.6f}")

    first_batch = next(iter(dataset))
    print("\n  === Gradient diagnostic (first batch, read-only) ===")
    diagnose_step(first_batch[0], first_batch[1], first_batch[2:])
    print("  ===================================================\n")

    history_epochs    = []
    history_val_loss  = []
    history_val_mape  = []
    best_val_loss     = float('inf')

    for epoch in range(epochs):
        for batch in dataset:
            X_batch   = batch[0]
            y_batch   = batch[1]
            d_batches = batch[2:]
            train_step(X_batch, y_batch, d_batches)

        y_val_pred = model(X_val_tf, training=False)
        val_loss = mae(y_val_tf, y_val_pred).numpy()
        val_mape = tf.keras.metrics.mean_absolute_percentage_error(
            y_val_tf, y_val_pred).numpy().mean()
        history_epochs.append(epoch + 1)
        history_val_loss.append(val_loss)
        history_val_mape.append(val_mape)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:4d}/{epochs}  val_loss={val_loss:.6f}"
                  f"  val_mape={val_mape:.4f}%  best={best_val_loss:.6f}")

    # Save once at end — avoids repeated I/O during training
    save_model(model, model_name, project_root)
    print(f"  Model saved (final): models/{model_name}/")

    final_val_loss = history_val_loss[-1]
    final_val_mape = history_val_mape[-1]
    return final_val_loss, final_val_mape, history_epochs, history_val_loss, history_val_mape


def write_training_log(log_path, cfg, model_name, resumed,
                       final_val_loss, final_val_mape,run_id):
    """Append a training summary block to models/{model}/training_log.txt."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    sep  = '=' * 72
    dash = '-' * 72
    with open(log_path, 'a') as f:
        f.write(f"\n{sep}\n")
        f.write(f"{timestamp}\n")
        f.write(f"{dash}\n")
        f.write(f"property       : {cfg['property']}\n")
        f.write(f"phase          : {cfg['phase']}\n")
        f.write(f"eos            : {cfg['eos']}\n")
        f.write(f"architecture   : {cfg['n_hidden']}  {cfg['activation']}\n")
        f.write(f"lr             : {cfg['lr']}\n")
        f.write(f"batch_size     : {cfg['batch_size']}\n")
        f.write(f"epochs         : {cfg['epochs']}\n")
        f.write(f"resumed        : {'yes' if resumed else 'no'}\n")
        f.write(f"use_deriv_loss : {cfg.get('use_deriv_loss', False)}\n")
        f.write(f"loss_weights   : {cfg.get('loss_weights', {})}\n")
        f.write(f"{dash}\n")
        f.write(f"final val_loss : {final_val_loss:.6f}\n")
        f.write(f"final val_MAPE : {final_val_mape:.4f} %\n")
        f.write(f"mlflow_run_id  : {run_id}\n")
        f.write(f"{sep}\n")
    print(f"\nTraining log appended: {log_path}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from saved model')
    args = parser.parse_args()

    project_root = get_project_root()

    # ── load config ───────────────────────────────────────────────────
    with open(project_root / 'config_train.yaml') as f:
        cfg = yaml.safe_load(f)

    property_name  = cfg['property']
    phase          = cfg['phase']
    eos            = cfg['eos']
    input_cols     = cfg['inputs']
    use_deriv_loss = cfg.get('use_deriv_loss', False)
    loss_weights   = cfg.get('loss_weights', {})
    n_hidden       = cfg['n_hidden']
    activation     = cfg['activation']
    srelu_eps      = float(cfg.get('srelu_eps', 0.5))
    lr             = float(cfg['lr'])
    epochs         = int(cfg['epochs'])
    batch_size     = int(cfg['batch_size'])

    model_name = f"{property_name}_{phase}"
    output_col = property_name   # always the same as property name

    print(f"\nTraining model: {model_name}")
    print(f"  Resume: {args.resume}")
    print(f"\nConfig:")
    print(f"  eos           : {eos}")
    print(f"  output        : {output_col}")
    print(f"  n_hidden      : {n_hidden}")
    print(f"  activation    : {activation}")
    print(f"  lr            : {lr}")
    print(f"  epochs        : {epochs}")
    print(f"  batch_size    : {batch_size}")
    print(f"  use_deriv_loss: {use_deriv_loss}")
    print(f"  loss_weights  : {loss_weights}")
    
    # ── MLflow — start run and log params ─────────────────────────
   # mlflow.set_tracking_uri(":./mlruns")
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
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


    # ── load data ─────────────────────────────────────────────────────
    df = load_scaled_csv(eos, phase, project_root)
    print(f"\nLoaded scaled data: {df.shape}")

    X = df[input_cols].values
    y = df[output_col].values

    # ── train/val split — split indices to keep derivative targets aligned ──
    idx = np.arange(len(X))
    idx_train, idx_val = train_test_split(idx, test_size=0.2, random_state=42)

    X_train, X_val = X[idx_train], X[idx_val]
    y_train, y_val = y[idx_train], y[idx_val]
    print(f"  Train: {X_train.shape}  Val: {X_val.shape}")

    # ── build or load model ───────────────────────────────────────────
    tf.random.set_seed(42)
    if args.resume:
        print("\nResuming from saved model...")
        model = load_model(model_name, project_root,
                           custom_objects={'SReLU': SReLU, 'SReLU3': SReLU3})
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss=tf.keras.losses.MeanAbsoluteError(),
            metrics=['mape']
        )
    else:
        print("\nBuilding model...")
        model = build_model(
            input_dim=len(input_cols),
            n_hidden=n_hidden,
            activation=activation,
            lr=lr,
            output_dim=1,
            srelu_eps=srelu_eps
        )

    model.summary()

    # ── train ─────────────────────────────────────────────────────────
    if use_deriv_loss:
        # GradientTape path — derivative targets aligned via index split
        scaling_params = load_scaling_params(eos, phase, project_root)
        deriv_factors  = compute_deriv_scale_factors(
            scaling_params, input_cols, output_col)

        prop_keys = {'rho', 'cs2', 'h', 'u', 's', 'g', 'gamma'}
        deriv_cols_needed = [k for k in loss_weights if k not in prop_keys]

        for col in deriv_cols_needed:
            if col not in df.columns:
                raise ValueError(
                    f"Derivative column '{col}' not found in scaled CSV.\n"
                    f"Available columns: {list(df.columns)}\n"
                    f"Make sure the raw CSV has this column and re-run scaling."
                )

        # Derivative targets must be in same space as dy_dX[:,i] from GradientTape
        # dy_dX[:,i] = d(output_scaled)/d(input_i_scaled)
        # For cp (= dh/dT): target = cp_physical * scale_T / unscale_h
        #                          = cp_physical / factor_for_h_wrt_T
        # factor_for_h_wrt_T = deriv_factors[0] = unscale_h * scale_T
        from src.helpers import unscale_value
        deriv_train_dict = {}
        deriv_val_dict   = {}
        for col_idx, col in enumerate(deriv_cols_needed):
            col_params = scaling_params[col]
            # unscale cp to physical space
            cp_train_phys = unscale_value(
                df[col].values[idx_train],
                col_params['method'], col_params['p1'], col_params['p2'])
            cp_val_phys = unscale_value(
                df[col].values[idx_val],
                col_params['method'], col_params['p1'], col_params['p2'])
            # convert to d(h_scaled)/d(T_scaled) space by dividing by factor
            factor = deriv_factors[col_idx]  # = unscale_h * scale_T
            deriv_train_dict[col] = cp_train_phys / factor
            deriv_val_dict[col]   = cp_val_phys   / factor

        final_val_loss, final_val_mape, hist_epochs, hist_loss, hist_mape = deriv_train(
            model, X_train, y_train, X_val, y_val,
            deriv_train_dict, deriv_val_dict,
            loss_weights, deriv_factors,
            lr, epochs, batch_size,
            model_name, project_root
        )
        # save epoch history CSV
        hist_path = project_root / 'models' / model_name / 'history.csv'
        hist_path.parent.mkdir(parents=True, exist_ok=True)
        import csv
        write_header = not hist_path.exists()
        with open(hist_path, 'a', newline='') as hf:
            writer = csv.writer(hf)
            if write_header:
                writer.writerow(['epoch', 'val_loss', 'val_mape'])
            for e, l, m in zip(hist_epochs, hist_loss, hist_mape):
                writer.writerow([e, f'{l:.8f}', f'{m:.6f}'])
        print(f"  History saved: {hist_path}")

    else:
        # plain model.fit() path — use ModelCheckpoint to save best only
        print("\nTraining (plain MAE)...")
        best_ckpt_dir = str(project_root / 'models' / model_name / 'saved_model')
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            filepath=best_ckpt_dir,
            save_best_only=True,
            monitor='val_loss',
            verbose=0
        )
        history = plain_train(model, X_train, y_train,
                               X_val, y_val, epochs, batch_size,
                               callbacks=[checkpoint_cb])
        final_val_loss = min(history.history['val_loss'])
        final_val_mape = history.history['val_mape'][-1] \
                         if 'val_mape' in history.history \
                         else float('nan')
        # save epoch history CSV
        hist_path = project_root / 'models' / model_name / 'history.csv'
        hist_path.parent.mkdir(parents=True, exist_ok=True)
        import csv
        write_mode = 'a' if args.resume else 'w'
        write_header = not hist_path.exists() or not args.resume
        with open(hist_path, write_mode, newline='') as hf:
            writer = csv.writer(hf)
            if write_header:
                writer.writerow(['epoch', 'val_loss', 'val_mape'])
            for e, (l, m) in enumerate(zip(history.history['val_loss'],
                                           history.history.get('val_mape',
                                           [float('nan')]*len(history.history['val_loss']))), 1):
                writer.writerow([e, f'{l:.8f}', f'{m:.6f}'])
        print(f"  History saved: {hist_path}")

    print(f"\nTraining complete.")
    print(f"  Final val_loss : {final_val_loss:.6f}")
    print(f"  Final val_MAPE : {final_val_mape:.4f} %")

    # GradientTape path saves best model during training loop.
    # Plain path saves via ModelCheckpoint callback above.
    # No unconditional save here — never overwrite best with worse.

    # ── write training log ────────────────────────────────────────────
    log_path = project_root / 'models' / model_name / 'training_log.txt'
    write_training_log(log_path, cfg, model_name, args.resume,
                       final_val_loss, final_val_mape,run_id)
    # ── MLflow — log metrics and end run ──────────────────────────
    mlflow.log_metric("val_loss", final_val_loss)
    mlflow.log_metric("val_mape", final_val_mape)
    mlflow.log_artifact(str(hist_path))
    mlflow.tensorflow.log_model(model, name="model")
    mlflow.end_run()
