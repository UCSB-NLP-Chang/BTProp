"""
python hmm_forward.py --dataset=wikibio --backbone=chatgpt
"""
import json
from typing import Dict, List

from absl import app, flags

from src.data_utils import (
    load_factcheckgpt_processed_human,
    load_felm_science_processed_human,
    load_wikibio_processed_human,
)
from src.hmm import HMMTreeNode

MAX_LAYER = 2
CONFIG_PATH = "configs/hmm.yaml"
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


def main():
    if FLAGS.dataset == "felm":
        data_list = load_felm_science_processed_human()
    elif FLAGS.dataset == "wikibio":
        data_list = load_wikibio_processed_human()[120:]
    elif FLAGS.dataset == "fact":
        data_list = load_factcheckgpt_processed_human()
    else:
        raise NotImplementedError

    to_eval_sents = []
    label_dict = {}
    for log_dict in data_list:
        segments = log_dict["segmented_response"]
        to_eval_sents += segments
        labels = log_dict["labels"]
        for idx in range(len(segments)):
            if FLAGS.dataset == "wikibio":
                if "inaccurate" in labels[idx]:
                    label_dict[segments[idx]] = 0
                elif "accurate" in labels[idx]:
                    label_dict[segments[idx]] = 1
                else:
                    raise NotImplementedError
            else:
                if labels[idx]:
                    label_dict[segments[idx]] = 1
                else:
                    label_dict[segments[idx]] = 0

    statement_to_conf = {}
    conf_path = f"logs/conf_estimation/{FLAGS.dataset}_{FLAGS.tree_type}_{FLAGS.backbone}.json"
    with open(conf_path, "r", encoding="utf-8") as f:
        conf_logs = json.load(f)
    for conf_dict in conf_logs:
        statement = conf_dict["statement"]
        conf = conf_dict["probability"]
        statement_to_conf[statement] = conf

    with open(
        f"logs/belief_trees/{FLAGS.dataset}_{FLAGS.tree_type}_{FLAGS.backbone}_w_label.json",
        "r",
        encoding="utf-8",
    ) as f:
        all_prems: List[Dict] = json.load(f)
    prem_dict: Dict[str, List[Dict]] = {}
    for log_dict in all_prems:
        prem_dict[log_dict["statement"]] = log_dict

    propagation_logs = []
    count = 0
    for idx in range(len(to_eval_sents)):
        curr_sent = to_eval_sents[idx]
        tgt_statement = curr_sent

        if curr_sent not in statement_to_conf:
            raise NotImplementedError
        else:
            orig_conf = statement_to_conf[curr_sent]
        if curr_sent not in prem_dict:
            log_dict = {
                "statement": tgt_statement,
                "probability": 1.0,
            }
            propagation_logs.append(log_dict)
            continue

        rootnode = HMMTreeNode(CONFIG_PATH, curr_sent, orig_conf)
        visited_node = []
        nodes_to_visit = [tgt_statement]
        while True:
            if len(nodes_to_visit) == 0:
                break
            curr_node = nodes_to_visit.pop()
            if curr_node not in prem_dict:
                continue
            if curr_node in visited_node:
                continue
            visited_node.append(curr_node)

            curr_child_nodes_dicts = prem_dict[curr_node]["child_nodes"]
            for child_node_dict in curr_child_nodes_dicts:
                curr_child_nodes = child_node_dict["child_nodes"]
                prem_gen_method = child_node_dict["prem_gen_method"]
                rechecked_relations = child_node_dict["mnli_relations"]

                deduplicated_relations = rechecked_relations
                deduplicated_sentences = curr_child_nodes
                if prem_gen_method == "decompose":
                    rechecked_relations = deduplicated_relations[:]
                    curr_child_nodes = deduplicated_sentences[:]
                elif prem_gen_method == "logical relation":
                    rechecked_relations = deduplicated_relations[:]
                    curr_child_nodes = deduplicated_sentences[:]
                else:
                    max_nodes = 5
                    rechecked_relations = deduplicated_relations[:max_nodes]
                    curr_child_nodes = deduplicated_sentences[:max_nodes]

                for child_idx, prem in enumerate(curr_child_nodes):
                    if prem == curr_node:
                        continue
                    if prem_gen_method == "perturbation":
                        prem_conf = 1.0
                    else:
                        prem_conf = statement_to_conf[prem]

                    nli_rel = rechecked_relations[child_idx]
                    if nli_rel == "neutral-neutral":
                        continue

                    insert_layer = rootnode.add_node_via_statement(
                        parent_statement=curr_node,
                        child_statement=prem,
                        conf=prem_conf,
                        nli_rel=nli_rel,
                        general_rel=prem_gen_method,
                    )
                    if insert_layer < MAX_LAYER and prem_gen_method != "perturbation":
                        nodes_to_visit.append(prem)

        rootnode.print()
        if len(rootnode.child_nodes) >= 1:
            rootnode.print()
            new_conf = rootnode.get_posterior()
            rootnode.print(printall=False, posteval=True)
            print(curr_sent)
            print("orig_conf: ", orig_conf)
            print("new_conf: ", new_conf)
            print("gt: ", label_dict[tgt_statement])
            if label_dict[tgt_statement] == 0:
                if new_conf > 0.5:
                    print("wrong!")
            elif label_dict[tgt_statement] == 1:
                if new_conf < 0.5:
                    print("wrong!")
            print("--------------------------------------\n")
        else:
            new_conf = orig_conf
        rootnode.print(printall=False, posteval=True)

        print(curr_sent)
        print("orig_conf: ", orig_conf)
        print("new_conf: ", new_conf)
        print("gt: ", label_dict[tgt_statement])
        print("--------------------------------------\n")

        log_dict = {
            "statement": tgt_statement,
            "probability": new_conf,
        }
        propagation_logs.append(log_dict)
        count += 1

    with open("logs/conf_estimation/hmm_posterior.json", "w", encoding="utf-8") as f:
        json.dump(propagation_logs, f, indent=4)


if __name__ == "__main__":
    set_eval_args()
    app.run(main=main)
