# pylint: disable=C0114
import os

import openai
from openai import OpenAI
from tenacity import retry, wait_chain, wait_fixed

openai.api_key = os.environ["OPENAI_API_KEY"]


@retry(
    wait=wait_chain(
        *[wait_fixed(1) for i in range(3)] + [wait_fixed(2) for i in range(2)] + [wait_fixed(3)]
    )
)
def chatgpt_query(**kwargs):
    client = OpenAI(api_key=openai.api_key)
    return client.chat.completions.create(**kwargs)


def llama_query(**kwargs):
    with openai.Client(api_key="EMPTY", base_url="http://localhost:10000/v1/") as client:
        res = client.chat.completions.create(
            model="/path/to/vllm_llama", **kwargs,
        )
    return res


def tree_output_extraction(model_ans: str):
    all_lines = model_ans.strip().split("\n")
    premise_list = []
    method_name = ""
    for line in all_lines:
        if line.startswith("Premise"):
            curr_premise = line[len("Premise 1: ") :].strip()
            premise_list.append(curr_premise)
        elif line.startswith("Method"):
            method_name = line[len("Method: ") :].strip()
        else:
            continue
    return premise_list, method_name

