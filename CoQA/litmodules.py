import re
import string
from collections import Counter

import torch

import pytorch_lightning as pl

from transformers import (
    Adafactor,
    AdamW,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    T5ForConditionalGeneration,
)


class LitT5(pl.LightningModule):
    def __init__(
        self,
        lr=3e-5,
        batch_size=1,
        num_train_epochs=15,
        warmup_steps=1000,
        max_input_length=512,
        max_output_length=64,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = T5ForConditionalGeneration.from_pretrained("t5-base")

    def setup(self, stage=None) -> None:
        if stage != "fit":
            return
        train_loader = (
            self.trainer._data_connector._train_dataloader_source.dataloader()
        )
        tb_size = self.hparams.batch_size * max(1, self.trainer.gpus)
        ab_size = self.trainer.accumulate_grad_batches
        self.total_steps = (
            (len(train_loader.dataset) // tb_size)
            // ab_size
            * float(self.trainer.max_epochs)
        )

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        return outputs

    def step(self, batch, batch_idx):
        outputs = self(**batch)
        return outputs.loss, outputs.logits

    def normalize_answer(self, s):
        """Lower text and remove punctuation, storys and extra whitespace."""

        def remove_articles(text):
            regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
            return re.sub(regex, " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def get_tokens(self, s):
        if not s:
            return []
        return self.normalize_answer(s).split()

    def compute_f1(self, a_gold, a_pred):
        gold_toks = self.get_tokens(a_gold)
        pred_toks = self.get_tokens(a_pred)
        common = Counter(gold_toks) & Counter(pred_toks)
        num_same = sum(common.values())
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
            return int(gold_toks == pred_toks)
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def compute_metrics(self, batch):
        labels = torch.flatten(batch["labels"])
        labels = torch.where(labels == -100, 0, labels)

        a_gold_list = torch.flatten(labels).tolist()
        a_gold_list = self.tokenizer.decode(
            a_gold_list,
            skip_special_tokens=True,
        )

        outputs = self.model.generate(input_ids=batch["input_ids"])
        a_pred_list = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
        )

        f1_sum = 0.0
        for a_gold, a_pred in zip(a_gold_list, a_pred_list):
            f1_sum += self.compute_f1(a_gold, a_pred)

        return f1_sum / len(a_gold_list)

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
                "scheduler": get_cosine_schedule_with_warmup(
                    optimizer,
                    num_training_steps=num_training_steps,
                    num_warmup_steps=self.hparams.warmup_steps,
                ),
                "interval": "step",
                "name": "learning_rate",
            }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        logits = outputs.logits

        preds = torch.argmax(logits, axis=1)
        labels = batch["labels"]

        self.log("val_loss", loss)

        return {"loss": loss, "preds": preds, "labels": labels}
