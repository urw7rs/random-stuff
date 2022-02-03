import pytorch_lightning as pl

from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset


class CoQADatamodule(pl.LightningDataModule):
    def __init__(
        self,
        model_name_or_path: str,
        max_input_length: int = 512,
        max_output_length: int = 64,
        batch_size: int = 4,
        val_batch_size: int = 8,
        num_workers=0,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    def prepare_data(self):
        dataset = load_dataset("coqa")  # noqa

    def setup(self, stage: str):
        self.dataset = load_dataset("coqa")

        self.dataset = self.dataset.map(
            self.preprocess_examples,
            batch_size=8,
            batched=True,
            remove_columns=self.dataset["train"].column_names,
        )

        self.dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"],
        )

    def preprocess_examples(self, examples):
        stories = ["c: " + story for story in examples["story"]]

        questions = []
        for question_list in examples["questions"]:
            questions.append(["q: " + question for question in question_list])

        answers = []
        for answer_list in examples["answers"]:
            answers.append(["a: " + ans for ans in answer_list["input_text"]])

        inputs = []
        for q_list, a_list, c in zip(questions, answers, stories):
            inputs.append(f"{q_list[0]} {c}")

            prev_qa = ""
            for i, (q, a) in enumerate(zip(q_list[:-1], a_list[:-1])):
                prev_qa += f"{q} {a}"
                inputs.append(f"{prev_qa} {q_list[i+1]} {c}")

        model_inputs = self.tokenizer(
            inputs,
            max_length=self.hparams.max_input_length,
            padding="max_length",
            truncation=True,
        )

        answers = []
        for answer_list in examples["answers"]:
            answers.extend([answer for answer in answer_list["input_text"]])

        # encode the summaries
        labels = self.tokenizer(
            answers,
            max_length=self.hparams.max_output_length,
            padding="max_length",
            truncation=True,
        ).input_ids

        # important: we need to replace the index of the padding tokens by -100
        # such that they are not taken into account by the CrossEntropyLoss
        labels_with_ignore_index = []
        for labels_example in labels:
            labels_example = [label if label != 0 else -100 for label in labels_example]
            labels_with_ignore_index.append(labels_example)

        model_inputs["labels"] = labels_with_ignore_index

        return model_inputs

    def train_dataloader(self):
        return DataLoader(
            self.dataset["train"],
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            shuffle=True,
            batch_size=self.hparams.batch_size,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset["validation"],
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            batch_size=self.hparams.val_batch_size,
        )
