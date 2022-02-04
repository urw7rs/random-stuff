import pytorch_lightning as pl

from transformers import (
    Adafactor,
    AdamW,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    T5ForConditionalGeneration,
)

from utils import get_max_length


class LitT5(pl.LightningModule):
    def __init__(
        self,
        lr=3e-5,
        batch_size=1,
        val_batch_size=8,
        metric_batch_size=16,
        num_train_epochs=15,
        warmup_steps=1000,
        max_input_length=512,
        max_output_length=64,
        gradient_checkpointing=False,
        model_name_or_path="t5-base",
        num_workers=0,
        freeze_layers=None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)
        for layer in freeze_layers:
            getattr(self.model, layer).requires_grad = False

        self.gradient_checkpointing = gradient_checkpointing
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            self.model.config.use_cache = False

    def setup(self, stage=None) -> None:
        if stage != "fit":
            return

        # workaround
        train_loader = (
            self.trainer._data_connector._train_dataloader_source.dataloader()
        )
        tb_size = self.hparams.batch_size * max(1, self.trainer.gpus)
        ab_size = self.trainer.accumulate_grad_batches
        self.total_steps = (
            len(train_loader.dataset) // tb_size // ab_size * self.trainer.max_epochs
        )

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs

    def step(self, batch, batch_idx):
        outputs = self(**batch)
        return outputs.loss, outputs.logits

    def configure_optimizers(self):
        if self.hparams.use_adafactor:
            optimizer = Adafactor(
                self.model.parameters(),
                lr=self.hparams.lr,
                eps=(1e-30, 1e-3),
                clip_threshold=1.0,
                decay_rate=-0.8,
                beta1=None,
                weight_decay=0.0,
                relative_step=False,
                scale_parameter=True,
                warmup_init=False,
            )
            lr_scheduler = {
                "scheduler": get_constant_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=self.hparams.warmup_steps,
                ),
                "interval": "step",
                "name": "learning_rate",
            }
        else:
            optimizer = AdamW(
                self.model.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
            )
            num_training_steps = self.total_steps
            lr_scheduler = {
                "scheduler": get_linear_schedule_with_warmup(
                    optimizer,
                    num_training_steps=num_training_steps,
                    num_warmup_steps=self.hparams.warmup_steps,
                ),
                "interval": "step",
                "name": "learning_rate",
            }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        max_length = get_max_length(input_ids, 0)

        batch["input_ids"] = input_ids[:, :max_length]
        batch["attention_mask"] = attention_mask[:, :max_length]

        max_length = get_max_length(labels, 100)

        batch["labels"] = labels[:, :max_length]

        outputs = self.model(**batch)
        loss = outputs.loss

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        max_length = get_max_length(input_ids, 0)

        batch["input_ids"] = input_ids[:, :max_length]
        batch["attention_mask"] = attention_mask[:, :max_length]

        max_length = get_max_length(labels, 100)

        batch["labels"] = labels[:, :max_length]

        outputs = self.model(**batch)
        loss = outputs.loss

        self.log("val_loss", loss)

        return loss
