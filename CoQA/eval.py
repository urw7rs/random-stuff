import argparse
import json

import torch

from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
from transformers import AutoTokenizer
from litmodules import LitT5

from utils import get_max_length

from evaluate import CoQAEvaluator

from datasets.utils.download_manager import DownloadManager


_DEV_DATA_URL = "https://nlp.stanford.edu/data/coqa/coqa-dev-v1.0.json"


class MetricData(Dataset):
    def __init__(
        self,
        gold_file,
        tokenizer,
        max_input_length,
        max_output_length,
        **kwargs,
    ):
        with open(gold_file) as f:
            raw = json.load(f)

        dataset = raw["data"]

        ids = []
        input_texts = []
        for data in dataset:
            context_id = data["id"]

            questions = data["questions"]
            answers = data["answers"]

            context = data["story"]
            context = f"c: {context}"

            turn_id = questions[0]["turn_id"]
            input_text = f"{questions[0]['input_text']} {context}"
            input_texts.append(input_text)

            ids.append(
                {
                    "id": context_id,
                    "turn_id": turn_id,
                }
            )

            prev_qa = ""
            for i, (q, a) in enumerate(zip(questions[:-1], answers[:-1])):
                turn_id = questions[i + 1]["turn_id"]
                q = q["input_text"]
                a = a["input_text"]
                prev_qa += f"q: {q} a: {a}"
                input_text = f"{prev_qa} q: {questions[i+1]['input_text']} c: {context}"
                input_texts.append(input_text)

                ids.append(
                    {
                        "id": context_id,
                        "turn_id": turn_id,
                    }
                )

        self.ids = ids
        self.input_ids = tokenizer(
            input_texts,
            padding=True,
            truncation=True,
            max_length=max_input_length,
            return_tensors="pt",
        ).input_ids

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "id": self.ids[idx]["id"],
            "turn_id": self.ids[idx]["turn_id"],
        }

    def __len__(self):
        return len(self.ids)


def main(args):
    dl_manager = DownloadManager()

    gold_file = dl_manager.download(_DEV_DATA_URL)
    evaluator = CoQAEvaluator(gold_file)

    pl_model = LitT5.load_from_checkpoint(args.ckpt_path)

    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    pl_model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(pl_model.hparams.model_name_or_path)
    model = pl_model.model

    dataset = MetricData(
        gold_file,
        tokenizer,
        max_input_length=args.max_input_length,
        max_output_length=args.max_output_length,
    )

    dataloader = DataLoader(
        dataset,
        num_workers=args.num_workers,
        pin_memory=True,
        batch_size=args.batch_size,
    )

    preds = []
    for batch in tqdm(dataloader):
        input_ids = batch["input_ids"]
        max_length = get_max_length(input_ids, 0)
        input_ids = input_ids[:max_length]

        outputs = model.generate(
            input_ids.to(pl_model.device),
            max_length=args.max_output_length,
        )
        answers = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for a, c_id, t_id in zip(answers, batch["id"], batch["turn_id"]):
            preds.append({"id": c_id, "turn_id": t_id.item(), "answer": a})

    pred_dict = {}
    for pred in preds:
        pred_dict[(pred["id"], pred["turn_id"])] = pred["answer"]

    metrics = evaluator.model_performance(pred_dict)
    print(json.dumps(metrics, indent=2))

    with open(args.output_path, "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="batch size",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="num_workers arg in DataLoader",
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
        "--ckpt_path",
        type=str,
        default=None,
        help="checkpointfile path",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="metrics.json",
        help="output file path",
    )

    args = parser.parse_args()

    main(args)
