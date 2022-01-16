import argparse

from litmodules import LitT5
from datamodules import CoQADatamodule

import wandb

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger


def main(args):
    dict_args = vars(args)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=3,
        mode="min",
        auto_insert_metric_name=True,
        save_last=True,
    )

    wandb.login()

    model = LitT5(**dict_args)
    coqa = CoQADatamodule(**dict_args)

    wandb_logger = WandbLogger(
        name=args.name,
        project=args.project_name,
    )
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=3,
        strict=False,
        verbose=False,
        mode="min",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
    )
    if args.auto_lr_find:
        trainer.tune(model, coqa)
    else:
        trainer.fit(model, coqa)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="batch size, learning_rate doesn't change with batch size",
    )
    parser.add_argument(
        "--use_adafactor",
        action="store_true",
        default=False,
        help="flag for Adafactor",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-5,
        help="learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="weight decay coeeficient",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="num_workers arg in DataLoader",
    )
    parser.add_argument(
        "--project_name",
        type=str,
        default="CoQA",
        help="wandb project name",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="default",
        help="wandb logger name",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=15,
        help="number of epochs",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=1000,
        help="number of warmup steps",
    )
    parser.add_argument(
        "--max_input_length",
        type=int,
        default=512,
        help="max input length",
    )
    parser.add_argument(
        "--max_output_length",
        type=int,
        default=64,
        help="max input length",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="t5-base",
        help="huggingface model name or path",
    )

    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    main(args)