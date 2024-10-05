from typing import List

import numpy as np
import torch
from sentence_transformers import CrossEncoder


class CrossEncoderNLI:
    """
    The NLI model to predict the NLI relationships between two nodes.
    """

    def __init__(self, device="cuda") -> None:
        self.model = CrossEncoder(
            "cross-encoder/nli-deberta-v3-large",
            device=torch.device(device)
        )

    @torch.no_grad()
    def _batch_pred(self, input_list):
        scores = self.model.predict(input_list)
        return scores

    def judge_relation(self, parent_node: str, child_nodes: List[str], visualize=False):
        input_list = []
        for c in child_nodes:
            input_example = (parent_node, c)
            input_list.append(input_example)
        for c in child_nodes:
            input_example = (c, parent_node)
            input_list.append(input_example)

        return self._batch_pred(input_list)

    def decode_relation(self, logits: np.ndarray):
        CONTRADICT, NEUTRAL, AGREE = 0, 2, 1
        label2rel = {CONTRADICT: "contradict", NEUTRAL: "neutral", AGREE: "entail"}

        pred_labels = logits.argmax(axis=1)
        relations = []
        str_relations = []
        pred_labels = np.stack(np.split(pred_labels, 2), axis=0)
        for child_id in range(pred_labels.shape[1]):
            pred_rel = pred_labels[:, child_id].tolist()
            relations.append(pred_rel)
            str_rel = label2rel[pred_rel[0]] + "-" + label2rel[pred_rel[1]]
            str_relations.append(str_rel)
        return str_relations
