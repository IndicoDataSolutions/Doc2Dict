import ast
import os
import json
import shutil
import subprocess
import random

import fire

from finetune.base_models.huggingface.models import HFT5
from finetune.target_models.seq2seq import HFS2S


def get_text_from_row(x, is_nda):
    # This selects the pdf text for nda and textract for charity
    text_idx = -4 if is_nda else -2
    return x.split("\t")[1] + " | " + x.split("\t", 6)[text_idx].strip()


def get_target(y):
    y_sample = []
    for entity in y.strip().split(" "):
        name, value = entity.split("=", 1)
        value = value.replace("_", " ")
        y_sample.append((name, value))
    return set(y_sample)


def get_tsv_row_from_pred(pred):
    try:
        pred = ast.literal_eval(pred)
        out = []
        for k, v, *_ in pred:
            v = v.replace(" ", "_")
            out.append(f"{k}={v}")
        return " ".join(out) + "\n"
    except:
        return "\n"


def get_dataset(dataset_dir, split, is_nda):
    x_out = []
    y_out = []
    with open(os.path.join(dataset_dir, split, "in.tsv"), "rt") as xs, open(
        os.path.join(dataset_dir, split, "expected.tsv"), "rt"
    ) as ys:
        for x, y in zip(xs, ys):
            row_text = get_text_from_row(x, is_nda=is_nda)
            x_out.append(row_text)
            y_sample = get_target(y)
            y_out.append(str(y_sample))
    return x_out, y_out


def get_test_dataset(dataset_dir, split, is_nda):
    x_out = []
    with open(os.path.join(dataset_dir, split, "in.tsv"), "rt") as xs:
        for x in xs:
            x_out.append(get_text_from_row(x, is_nda=is_nda))
    return x_out


def make_shuffled_epochs(x, y, n_epochs):
    x_out = []
    y_out = []
    for _ in range(n_epochs):
        for xi, yi in zip(x, y):
            yi = list(ast.literal_eval(yi))
            # Note that the sort is only by the keys - values will remain in the shuffled order.
            random.shuffle(yi)
            y_out.append("{" + str(sorted(yi, key=lambda v: v[0]))[1:-1] + "}")
            x_out.append(str(xi))
    return x_out, y_out


def make_pred_tsv(
    model,
    dataset_dir,
    split,
    output_dir,
    is_nda,
    beam_size=1,
    lower=False,
    commas=False,
):
    os.mkdir(os.path.join(output_dir, split))
    x_out = get_test_dataset(dataset_dir, split, is_nda=is_nda)

    if lower:
        x_out = [x.lower() for x in x_out]
    if commas:
        x_out = [strip_commas_numeric(x) for x in x_out]

    model.config.beam_size = beam_size
    raw_preds = model.predict(x_out)
    expected_path = os.path.join(dataset_dir, split, "expected.tsv")
    if os.path.exists(expected_path):
        shutil.copy(expected_path, os.path.join(output_dir, split, "expected.tsv"))
    with open(os.path.join(output_dir, split, "out.tsv"), "wt") as out:
        for pred in raw_preds:
            out.write(get_tsv_row_from_pred(pred))


def strip_commas_numeric(string):
    out = ""
    for c in string:
        if out and c == "," and out[-1].isnumeric():
            continue
        out += c
    return out


def run_eval(directory):
    output = subprocess.getoutput("./geval -t {}".format(directory))
    uncased_f1 = output.split()[4]
    try:
        f1 = float(uncased_f1.split("±")[0])
    except:
        f1 = None
    return output, f1


def train_and_eval(
    dataset_dir=None,
    output_dir=None,
    is_nda=None,
    shuffled_epochs=False,
    n_epochs=64,
    lowercase=False,
    strip_commas=False,
    learning_rate=None,
    remove_trunc=False,
    chunk_pos=None,
    encoder_shards=None,
    batch_size=1,
    decoder_max_length=512,
    beam_size=1,
):
    try:
        os.mkdir(output_dir)
        shutil.copy(os.path.join(dataset_dir, "in-header.tsv"), os.path.join(dataset_dir, "in-header.tsv"))
        shutil.copy(os.path.join(dataset_dir, "config.txt"), os.path.join(dataset_dir, "config.txt"))
    except:
        pass
    if encoder_shards is None:
        encoder_shards = 32 if is_nda else 64

    finetune_model = HFS2S(
        base_model=HFT5,
        n_epochs=1 if shuffled_epochs else n_epochs,
        batch_size=batch_size,
        max_length=512 * encoder_shards,
        chunk_long_sequences=False,
        low_memory_mode=True,
        s2s_decoder_max_length=decoder_max_length,
        beam_size=beam_size,
        predict_batch_size=1,
        num_fusion_shards=encoder_shards,
        lr=learning_rate or 6.25e-5,
        chunk_pos_embed=chunk_pos,
    )

    x_in, y_in = get_dataset(
        dataset_dir,
        "train",
        is_nda=is_nda
    )
    if shuffled_epochs:
        x_in, y_in = make_shuffled_epochs(x_in, y_in, n_epochs=n_epochs)
    if lowercase:
        x_in = [x.lower() for x in x_in]
        y_in = [y.lower() for y in y_in]
    if strip_commas:
        x_in = [strip_commas_numeric(x) for x in x_in]
    finetune_model.fit(x_in, y_in)
    finetune_model.save(os.path.join(output_dir, "model.jl"))
    make_pred_tsv(
        model=finetune_model,
        dataset_dir=dataset_dir,
        split="dev-0",
        output_dir=output_dir,
        is_nda=is_nda,
    )
    make_pred_tsv(
        model=finetune_model,
        dataset_dir=dataset_dir,
        split="test-A",
        output_dir=output_dir,
        is_nda=is_nda,
    )
    dev_output, dev_f1 = run_eval(os.path.join(output_dir, "dev-0"))
    print("Dev Metrics, overall f1 = {}".format(dev_f1))
    print(dev_output)
    test_output, test_f1 = run_eval(os.path.join(output_dir, "test-A"))
    print("Test Metrics, overall f1 = {}".format(test_f1))
    print(test_output)


if __name__ == "__main__":
    fire.Fire(train_and_eval)
