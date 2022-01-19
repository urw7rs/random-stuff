import argparse
import json

import torch

from tqdm import tqdm
from transformers import AutoTokenizer
from litmodules import LitT5

from evaluate import CoQAEvaluator

from datasets.utils.download_manager import DownloadManager


_DEV_DATA_URL = "https://nlp.stanford.edu/data/coqa/coqa-dev-v1.0.json"


def generate(input_text, tokenizer, model, max_length, device):
    input_ids = tokenizer(input_text, truncation=True, return_tensors="pt").input_ids
    outputs = model.generate(
        input_ids.to(device),
        max_length=max_length,
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return answer


def main(args):
    dl_manager = DownloadManager()

    gold_file = dl_manager.download(_DEV_DATA_URL)
    evaluator = CoQAEvaluator(gold_file)

    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"

    pl_model = LitT5.load_from_checkpoint(args.ckpt_path)
    tokenizer = AutoTokenizer.from_pretrained(pl_model.hparams.model_name_or_path)
    model = pl_model.model.to(device)

    with open(gold_file) as f:
        raw = json.load(f)

    dataset = raw["data"]

    preds = []
    for data in tqdm(dataset):
        context_id = data["id"]

        questions = data["questions"]
        answers = data["answers"]

        context = data["story"]
        context = f"c: {context}"

        turn_id = questions[0]["turn_id"]
        input_text = f"{questions[0]} {context}"
        answer = generate(
            input_text,
            tokenizer,
            model,
            pl_model.hparams.max_output_length,
            device,
        )

        preds.append(
            {
                "id": context_id,
                "turn_id": turn_id,
                "answer": answer,
            }
        )

        prev_qa = ""
        for i, (q, a) in enumerate(zip(questions[:-1], answers[:-1])):
            turn_id = questions[i + 1]["turn_id"]
            q = q["input_text"]
            a = a["input_text"]
            prev_qa += f"q: {q} a: {a}"
            input_text = f"{prev_qa} {questions[i+1]} {context}"

            answer = generate(
                input_text,
                tokenizer,
                model,
                pl_model.hparams.max_output_length,
                device,
            )

            preds.append(
                {
                    "id": context_id,
                    "turn_id": turn_id,
                    "answer": answer,
                }
            )

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
        "--pred_file",
        type=str,
        default="preds.json",
        help="prediction file path",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="metrics.json",
        help="output file path",
    )

    args = parser.parse_args()

    main(args)
