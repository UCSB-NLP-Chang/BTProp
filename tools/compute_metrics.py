"""
python tools/compute_metrics.py --dataset felm --tree_type orig --backbone llama
"""

import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import json  # noqa: E402

import numpy as np  # noqa: E402
from absl import app, flags  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from src.data_utils import (  # noqa: E402
    load_factcheckgpt_processed_human,
    load_felm_science_processed_human,
    load_wikibio_processed_human,
)

FLAGS = flags.FLAGS


def set_eval_args():
    flags.DEFINE_enum(
        "dataset",
        None,
        ["wikibio", "felm", "factcheckgpt",],
        help="Name of the evaluation Dataset.",
    )
    flags.DEFINE_enum(
        "backbone",
        None,
        ["chatgpt", "llama",],
        help="The name of the model that used to generate belief trees.",
    )


def load_eval_data():
    if FLAGS.dataset == "felm":
        data_list = load_felm_science_processed_human()
    elif FLAGS.dataset == "wikibio":
        data_list = load_wikibio_processed_human()[120:]
    elif FLAGS.dataset == "fact":
        data_list = load_factcheckgpt_processed_human()
    else:
        raise NotImplementedError

    with open("logs/conf_estimation/hmm_posterior.json", "r", encoding="utf-8") as f:
        confidence_list = json.load(f)
    statement_to_conf = {}
    for conf_dict in confidence_list:
        statement = conf_dict["statement"]
        conf = conf_dict["probability"]
        statement_to_conf[statement] = conf

    return data_list, statement_to_conf


def main(argv):
    data_list, statement_to_conf = load_eval_data()
    with open(
        f"logs/conf_estimation/{FLAGS.dataset}_{FLAGS.backbone}.json", "r", encoding="utf-8"
    ) as f:
        base_conf_list = json.load(f)
        base_s2c = {}
        for conf_dict in base_conf_list:
            statement = conf_dict["statement"]
            conf = conf_dict["probability"]
            base_s2c[statement] = conf

    target_ys = []
    pred_probs = []
    base_probs = []

    for log_dict in data_list:
        segments = log_dict["segmented_response"]
        labels = log_dict["labels"]

        for idx, sent in enumerate(segments):
            if sent not in base_s2c:
                continue
            if FLAGS.dataset == "wikibio":
                if "inaccurate" in labels[idx]:
                    tgt_y = 0
                elif "accurate" in labels[idx]:
                    tgt_y = 1
                else:
                    raise NotImplementedError
            elif FLAGS.dataset == "factcheckgpt":
                if isinstance(labels[idx], str):
                    continue
                if labels[idx]:
                    tgt_y = 1
                else:
                    tgt_y = 0
            else:
                if labels[idx]:
                    tgt_y = 1
                else:
                    tgt_y = 0
            pred_probs.append(statement_to_conf[sent])
            base_probs.append(base_s2c[sent])

            target_ys.append(tgt_y)

    target_ys = np.array(target_ys)
    pred_probs = np.array(pred_probs)
    base_probs = np.array(base_probs)

    print("total example: ", len(pred_probs))

    true_statement_conf = pred_probs[target_ys == 1]
    false_statement_conf = pred_probs[target_ys == 0]
    print(
        f"posterior conf distribution on true statements: mean = {np.mean(true_statement_conf)}, std = {np.std(true_statement_conf)}"
    )
    print(
        f"posterior conf distribution on false statements: mean = {np.mean(false_statement_conf)}, std = {np.std(false_statement_conf)}"
    )

    true_statement_base_conf = base_probs[target_ys == 1]
    false_statement_base_conf = base_probs[target_ys == 0]
    print(
        f"base conf distribution on true statements: mean = {np.mean(true_statement_base_conf)}, std = {np.std(true_statement_base_conf)}"
    )
    print(
        f"base conf distribution on false statements: mean = {np.mean(false_statement_base_conf)}, std = {np.std(false_statement_base_conf)}"
    )

    target_ys = 1 - target_ys
    pred_probs = 1 - pred_probs
    base_probs = 1 - base_probs

    auroc = roc_auc_score(target_ys, pred_probs)
    print("auroc: ", auroc)

    prauc = average_precision_score(target_ys, pred_probs)
    print("prauc: ", prauc)

    auroc = roc_auc_score(target_ys, base_probs)
    print("base auroc: ", auroc)

    prauc = average_precision_score(target_ys, base_probs)
    print("base prauc: ", prauc)


if __name__ == "__main__":
    app.run(main)
