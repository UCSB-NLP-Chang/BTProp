import copy
import inspect
import json
import os
import sys

import tqdm
from absl import app, flags

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from src.model_utils import CrossEncoderNLI  # noqa: E402

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


def main(argv):
    nli_model = CrossEncoderNLI(device = 'cuda')

    data_path = f"logs/belief_trees/{FLAGS.dataset}_{FLAGS.backbone}.json"
    with open(data_path,'r',encoding='utf-8') as f:
        prem_logs = json.load(f)

    new_prem_logs = []
    for log_dict in tqdm.tqdm(prem_logs):
        new_log_dict = copy.deepcopy(log_dict)
        target_statement = log_dict['statement']
        childnodes = log_dict['child_nodes']
        cp_child_nodes = []
        for childnode in childnodes:
            premises = childnode['child_nodes']
            logits = nli_model.judge_relation(parent_node = target_statement, child_nodes = premises)
            bert_pred_nlis = nli_model.decode_relation(logits)
            processed_bert_nlis = []
            for idx, bidirection_rel in enumerate(bert_pred_nlis):
                forward_rel, backward_rel = bidirection_rel.split("-", 1)
                bidirection_rel = f"{forward_rel}-{backward_rel}"
                processed_bert_nlis.append(bidirection_rel)

            cp_child = copy.deepcopy(childnode)
            cp_child['mnli_relations'] = processed_bert_nlis
            cp_child_nodes.append(cp_child)
        new_log_dict['child_nodes'] = cp_child_nodes
        new_prem_logs.append(new_log_dict)

    with open(data_path,'w',encoding='utf-8') as f:
        json.dump(new_prem_logs, f, indent= 4)

if __name__ == '__main__':
    set_eval_args()
    app.run(main=main)
