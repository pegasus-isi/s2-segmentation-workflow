#!/usr/bin/env python3

"""Train a U-Net model for Sentinel-2 sea ice segmentation.

Supports three training modes:
  - single-gpu: Standard single-GPU training
  - mirrored: Multi-GPU via tf.distribute.MirroredStrategy
  - horovod: Multi-node via Horovod (launched with horovodrun)

The model definition is imported from model.py (staged by Pegasus).
"""

import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import tensorflow as tf
from keras import backend as K

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Import shared model from working directory (staged by Pegasus)
sys.path.insert(0, os.getcwd())
from model import multi_unet_model


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + K.epsilon())


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


class EpochTimer(tf.keras.callbacks.Callback):
    """Record per-epoch wall time + samples/sec for paper Fig 12-style plots.

    Populates two lists on ``self.history`` after ``model.fit`` returns:

    - ``epoch_time_seconds`` — wall-clock seconds for each epoch's training
      step (excludes Keras setup / teardown overhead).
    - ``samples_per_second`` — ``samples_per_epoch / epoch_time_seconds``.

    ``samples_per_epoch`` defaults to ``len(X_train)``; pass it explicitly
    when ``model.fit`` uses ``steps_per_epoch`` (e.g. the Horovod path)
    so the throughput number stays correct.
    """

    def __init__(self, samples_per_epoch):
        super().__init__()
        self.samples_per_epoch = samples_per_epoch
        self.epoch_times = []
        self.samples_per_sec = []

    def on_epoch_begin(self, epoch, logs=None):
        self._t0 = time.time()

    def on_epoch_end(self, epoch, logs=None):
        dt = time.time() - self._t0
        self.epoch_times.append(dt)
        sps = (self.samples_per_epoch / dt) if dt > 0 else 0.0
        self.samples_per_sec.append(sps)


def train_single_gpu(X_train, y_train_cat, args):
    """Single-GPU training."""
    model = multi_unet_model(
        n_classes=args.n_classes,
        IMG_HEIGHT=X_train.shape[1],
        IMG_WIDTH=X_train.shape[2],
        IMG_CHANNELS=X_train.shape[3],
    )
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy", f1_m, precision_m, recall_m],
    )
    logger.info(model.summary())

    epoch_timer = EpochTimer(samples_per_epoch=len(X_train))
    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir="./logs"),
        epoch_timer,
    ]

    history = model.fit(
        X_train, y_train_cat,
        batch_size=args.batch_size,
        verbose=1,
        epochs=args.epochs,
        callbacks=callbacks,
        shuffle=False,
    )
    return model, history, epoch_timer


def train_mirrored(X_train, y_train_cat, args):
    """Multi-GPU training using MirroredStrategy."""
    strategy = tf.distribute.MirroredStrategy()
    logger.info(f"Number of devices: {strategy.num_replicas_in_sync}")

    with strategy.scope():
        model = multi_unet_model(
            n_classes=args.n_classes,
            IMG_HEIGHT=X_train.shape[1],
            IMG_WIDTH=X_train.shape[2],
            IMG_CHANNELS=X_train.shape[3],
        )
        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy", f1_m, precision_m, recall_m],
        )
        logger.info(model.summary())

        epoch_timer = EpochTimer(samples_per_epoch=len(X_train))
        callbacks = [
            tf.keras.callbacks.TensorBoard(log_dir="./logs"),
            epoch_timer,
        ]

        batch_size = args.batch_size * strategy.num_replicas_in_sync
        history = model.fit(
            X_train, y_train_cat,
            batch_size=batch_size,
            verbose=1,
            epochs=args.epochs,
            callbacks=callbacks,
            shuffle=False,
        )
    return model, history, epoch_timer


def train_horovod(X_train, y_train_cat, args):
    """Multi-node training using Horovod."""
    import horovod.tensorflow.keras as hvd

    hvd.init()

    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], "GPU")

    model = multi_unet_model(
        n_classes=args.n_classes,
        IMG_HEIGHT=X_train.shape[1],
        IMG_WIDTH=X_train.shape[2],
        IMG_CHANNELS=X_train.shape[3],
    )

    opt = tf.keras.optimizers.Adam()
    opt = hvd.DistributedOptimizer(
        opt, backward_passes_per_step=1, average_aggregated_gradients=True
    )

    model.compile(
        optimizer=opt,
        loss="categorical_crossentropy",
        metrics=["accuracy", f1_m, precision_m, recall_m],
        experimental_run_tf_function=False,
    )

    if hvd.rank() == 0:
        logger.info(model.summary())

    # Convert to tf.data pipeline for Horovod
    X_tensor = tf.convert_to_tensor(X_train, dtype=tf.float32)
    y_tensor = tf.convert_to_tensor(y_train_cat, dtype=tf.int32)
    dataset = tf.data.Dataset.from_tensor_slices((X_tensor, y_tensor))
    dataset = dataset.repeat().shuffle(10000).batch(args.batch_size)

    # Effective samples processed per epoch across all ranks.
    steps_per_epoch = len(y_train_cat) // (args.batch_size * hvd.size())
    samples_per_epoch = steps_per_epoch * args.batch_size * hvd.size()
    epoch_timer = EpochTimer(samples_per_epoch=samples_per_epoch)
    callbacks = [
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),
        hvd.callbacks.MetricAverageCallback(),
        epoch_timer,
    ]

    verbose = 1 if hvd.rank() == 0 else 0

    history = model.fit(
        dataset,
        steps_per_epoch=steps_per_epoch,
        verbose=verbose,
        epochs=args.epochs,
        callbacks=callbacks,
    )
    return model, history, epoch_timer


def main():
    parser = argparse.ArgumentParser(description="Train U-Net model")
    parser.add_argument("--train-data", required=True, help="X_train.npy")
    parser.add_argument("--train-labels", required=True, help="y_train_cat.npy")
    parser.add_argument("--output-model", required=True, help="Output model.hdf5")
    parser.add_argument("--output-history", required=True, help="Output training_history.json")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--n-classes", type=int, default=0,
                        help="Number of classes (0 = infer from labels or metadata)")
    parser.add_argument("--metadata", default=None,
                        help="Preprocessing metadata JSON (overrides --n-classes)")
    parser.add_argument("--mode", choices=["single-gpu", "mirrored", "horovod"],
                        default="single-gpu", help="Training mode")
    args = parser.parse_args()

    logger.info(f"Training mode: {args.mode}")
    logger.info(f"Epochs: {args.epochs}, Batch size: {args.batch_size}")

    X_train = np.load(args.train_data)
    y_train_cat = np.load(args.train_labels)
    logger.info(f"Training data: {X_train.shape}, Labels: {y_train_cat.shape}")

    # Determine n_classes: metadata > explicit arg > infer from labels shape
    if args.metadata and os.path.exists(args.metadata):
        with open(args.metadata) as f:
            meta = json.load(f)
        args.n_classes = meta["n_classes"]
        logger.info(f"n_classes={args.n_classes} (from metadata)")
    elif args.n_classes <= 0:
        args.n_classes = y_train_cat.shape[-1]
        logger.info(f"n_classes={args.n_classes} (inferred from label shape)")
    else:
        logger.info(f"n_classes={args.n_classes} (from --n-classes)")

    t0 = time.time()

    if args.mode == "single-gpu":
        model, history, epoch_timer = train_single_gpu(X_train, y_train_cat, args)
    elif args.mode == "mirrored":
        model, history, epoch_timer = train_mirrored(X_train, y_train_cat, args)
    elif args.mode == "horovod":
        model, history, epoch_timer = train_horovod(X_train, y_train_cat, args)

    t1 = time.time()
    logger.info(f"Training time: {t1 - t0:.2f} seconds")

    # Save model (only rank 0 for Horovod)
    save = True
    if args.mode == "horovod":
        import horovod.tensorflow.keras as hvd
        save = hvd.rank() == 0

    if save:
        model.save(args.output_model)
        logger.info(f"Model saved: {args.output_model}")

        # Determine the effective replica count (number of devices doing
        # data-parallel training) so downstream throughput plots can
        # group runs by GPU count.
        if args.mode == "mirrored":
            replicas = len(
                tf.config.experimental.list_physical_devices("GPU")) or 1
        elif args.mode == "horovod":
            import horovod.tensorflow.keras as hvd
            replicas = hvd.size()
        else:
            replicas = 1

        # Save training history
        hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        hist["training_time_seconds"] = t1 - t0
        hist["epoch_time_seconds"] = epoch_timer.epoch_times
        hist["samples_per_second"] = epoch_timer.samples_per_sec
        hist["training_meta"] = {
            "mode": args.mode,
            "replicas": replicas,
            "batch_size": args.batch_size,
            "samples_per_epoch": epoch_timer.samples_per_epoch,
            "epochs": args.epochs,
        }
        with open(args.output_history, "w") as f:
            json.dump(hist, f, indent=2)
        logger.info(f"History saved: {args.output_history}")


if __name__ == "__main__":
    main()
