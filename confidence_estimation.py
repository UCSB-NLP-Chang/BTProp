import json
import os

import numpy as np
from absl import app, flags
from tqdm.auto import tqdm as auto_tqdm

from src import prompts
from src.common import chatgpt_query, llama_query
from src.data_utils import (
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
    # flags.DEFINE_boolean(
    #     "use_cache",
    #     True,
    #     help="Whether cache the LLM queries and responses."
    # )


def load_all_statment(statement_to_prem):
    all_statements = []
    for log_dict in statement_to_prem:
        statement = log_dict["statement"]
        if statement not in all_statements:
            all_statements.append(statement)
        childnodes = log_dict["child_nodes"]
        for childnode in childnodes:
            gen_method = childnode["prem_gen_method"]
            if gen_method == "perturbation":
                continue
            prems = childnode["child_nodes"]
            for prem_id, prem in enumerate(prems):
                if prem not in all_statements:
                    all_statements.append(prem)

    if FLAGS.dataset == "felm":
        data_list = load_felm_science_processed_human()
    elif FLAGS.dataset == "wikibio":
        data_list = load_wikibio_processed_human()
    elif FLAGS.dataset == "fact":
        data_list = load_factcheckgpt_processed_human()
    else:
        raise NotImplementedError
    for log_dict in data_list:
        segments = log_dict["segmented_response"]
        for segment in segments:
            if segment not in all_statements:
                all_statements.append(segment)


def main(argv):
    datapath = f"logs/premise_gen/recycle/{FLAGS.dataset}_{FLAGS.backbone}.json"
    if FLAGS.backbone == "chatgpt":
        query_fn = chatgpt_query
        confidence_gen_prompt = prompts.chatgpt_confidence_prob
    elif FLAGS.backbone == "llama":
        query_fn = llama_query
        confidence_gen_prompt = prompts.llama_confidence_prob
    else:
        raise NotImplementedError

    with open(datapath, "r", encoding="utf-8") as f:
        statement_to_prem = json.load(f)

    all_statements = load_all_statment(statement_to_prem)
    conf_estimation_logs = []

    prog_bar = auto_tqdm(range(len(all_statements)))

    for data_example in all_statements:
        if FLAGS.backbone == "chatgpt":
            user_prompt = confidence_gen_prompt + " " + data_example.strip()
            response = query_fn(
                messages=[{"role": "user", "content": user_prompt},],
                temperature=0.0,
                max_tokens=5,
                n=1,
                logprobs=True,
                top_logprobs=5,
            )
        elif FLAGS.backbone == "llama":
            instruction_str = confidence_gen_prompt[:]
            messages = [
                {"role": "user", "content": instruction_str},
                {
                    "role": "assistant",
                    "content": "Yes, I understand the task. You can provide the statement"
                    " you'd like me to guess the truthfulness.",
                },
            ]
            messages.append({"role": "user", "content": "Statement: " + data_example.strip()})
            response = query_fn(
                messages=messages, temperature=0.0, max_tokens=5, n=1, logprobs=True, top_logprobs=5
            )

        if FLAGS.backbone == "chatgpt":
            if response is None or response.choices[0].message.content is None:
                prob = 0.5
                raw_outputs = ""
            else:
                ans_model = response.choices[0].message.content
                raw_outputs = ans_model
                first_logprobs = response.choices[0].logprobs.content[0].top_logprobs
                true_logprob = None
                false_logprob = None
                for logprob in first_logprobs:
                    token = logprob.token
                    logit = logprob.logprob
                    if "true" in token.lower():
                        if true_logprob is None:
                            true_logprob = logit
                    elif "false" in token.lower():
                        if false_logprob is None:
                            false_logprob = logit
                    else:
                        continue
                if true_logprob is None or false_logprob is None:
                    prob = 0.5
                else:
                    cls_array = np.array([true_logprob, false_logprob])
                    cls_array = np.exp(cls_array) / np.sum(np.exp(cls_array))
                    prob = cls_array.tolist()[0]
        else:
            ans_model = response.choices[0].message.content
            raw_outputs = ans_model
            logprob_dict = response.choices[0].logprobs.top_logprobs[0]

            true_prob = np.exp(logprob_dict.get(" True", -20)) + np.exp(
                logprob_dict.get("True", -20)
            )
            false_prob = np.exp(logprob_dict.get(" False", -20)) + np.exp(
                logprob_dict.get("False", -20)
            )
            cls_array = np.array([true_prob, false_prob])
            cls_array = cls_array / np.sum(cls_array)
            prob = cls_array.tolist()[0]

        log_dict = {"statement": data_example, "probability": prob, "raw_output": raw_outputs}

        conf_estimation_logs.append(log_dict)
        prog_bar.update(1)

    savename = f"logs/conf_estimation/{FLAGS.dataset}_{FLAGS.backbone}.json"
    if not os.path.exists(os.path.dirname(savename)):
        os.makedirs(os.path.dirname(savename))

    with open(savename, "w", encoding="utf-8") as f:
        json.dump(conf_estimation_logs, f, indent=4)


if __name__ == "__main__":
    app.run(main)
