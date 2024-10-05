import json
import os

from absl import flags
from tqdm.auto import tqdm as auto_tqdm

from src.data_utils import (
    load_factcheckgpt_processed_human,
    load_felm_science_processed_human,
    load_wikibio_processed_human,
)
from src.tree_gen import TreeGenerator

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
    flags.DEFINE_integer("max_layers", 2, help="The maximum number of layers of the belief tree.")


if __name__ == "__main__":
    prem_layer_dict = {}
    if FLAGS.dataset == "felm":
        data_list = load_felm_science_processed_human()
    elif FLAGS.dataset == "wikibio":
        data_list = load_wikibio_processed_human()
    elif FLAGS.dataset == "fact":
        data_list = load_factcheckgpt_processed_human()
    else:
        raise NotImplementedError

    prem_generator = TreeGenerator(backbone=FLAGS.backbone,)

    to_generate = []
    total_num = 0
    for idx, data_example in enumerate(data_list):
        segments = data_example["segmented_response"]
        total_num += len(segments)
        to_generate += segments

    prog_bar = auto_tqdm(range(total_num))

    premise_gen_logs = []
    already_generated = []
    for segment in to_generate:
        if segment not in already_generated:
            already_generated.append(segment)
        else:
            continue

        # the sentences in FactCheckGPT has already been processed into atomic statements
        # So, we do not decompose it.
        if FLAGS.dataset == "factcheckgpt":
            root_can_be_decompose = False
        else:
            root_can_be_decompose = True
        generated_logs = prem_generator.chatgpt_generate_pipeline(
            root_statement=segment,
            root_can_be_decompose=root_can_be_decompose,
            max_layers=FLAGS.max_layers,
        )

        premise_gen_logs += generated_logs
        prog_bar.update(1)

    savename = f"logs/belief_trees/{FLAGS.dataset}_{FLAGS.backbone}.json"

    if not os.path.exists(os.path.dirname(savename)):
        os.makedirs(os.path.dirname(savename))

    with open(savename, "w", encoding="utf-8") as f:
        json.dump(premise_gen_logs, f, indent=4)
