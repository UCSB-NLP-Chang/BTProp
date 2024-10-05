from typing import Dict, List

from src.common import chatgpt_query, llama_query

QUERY_FN_MAP = {"chatgpt": chatgpt_query, "llama": llama_query}


class InferenceAPI:
    """
    A simple wrapper for prompting the LLM APIs.
    We include many different prompts to prompt the LLMs to generate child nodes,
        determine their truthfulness, and judge their relationships. These prompts may
        be instruction-only or with few-shot examples. Therefore, we add this wrapper
        layer to format the prompts.
    """

    def __init__(
        self, model_name: str,
    ):
        """
        Args:
            model_name: the name of the backbone model for child node generation.
                Choices: "chatgpt" and "llama".
        """
        self.model_name = model_name
        self.query_fn = QUERY_FN_MAP[self.model_name]

    def query(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        n: int,
        logprobs=False,
        top_logprobs=5,
    ):
        """
        Query the LLMs with only an instruction.

        Args:
            prompt: the prompt for the task (e.g., 'generate premises for a given statement.').
            temperature: the temperature for LLM decoding.
            max_tokens: maximum number of tokens the LLM can generate.
            n: number of examples to sample from the LLM.
            logprobs: whether return the log-probabilities of the generated tokens.
                This is mainly used when determine the model's confidence on a statement.
            top_logprobs: the number of most likely tokens to return at each token position.
        """
        response = self.query_fn(
            messages=[{"role": "user", "content": prompt},],
            temperature=temperature,
            max_tokens=max_tokens,
            n=n,
            logprobs=logprobs,
            top_logprobs=top_logprobs if logprobs else None,
        )

        return response

    def query_with_msg(
        self,
        msg: List[Dict],
        temperature: float,
        max_tokens: int,
        n: int,
        logprobs=False,
        top_logprobs=5,
    ):
        """
        Query the LLMs with only an instruction.

        Args:
            msg: the message list (user/assistant).
            temperature: the temperature for LLM decoding.
            max_tokens: maximum number of tokens the LLM can generate.
            n: number of examples to sample from the LLM.
            logprobs: whether return the log-probabilities of the generated tokens.
                This is mainly used when determine the model's confidence on a statement.
            top_logprobs: the number of most likely tokens to return at each token position.
        """
        response = self.query_fn(
            messages=msg,
            temperature=temperature,
            max_tokens=max_tokens,
            n=n,
            logprobs=logprobs,
            top_logprobs=top_logprobs if logprobs else None,
        )

        return response

    def query_with_split_examples(
        self,
        instruction: str,
        initial_resp: str,
        test_example: str,
        split_tag: str = "**Examples:**",
        example_split_tag: str = "\n\n",
        example_start_tag: str = "Target statement: ",
        temperature=0.0,
        max_tokens=30,
        n=1,
    ):
        """
        Query the LLMs with few-shot examples. Initially, we found some LLMs could not leverage
            few-shot examples effectively if we organized them into a single message. Instead,
            we had to put it into different messages like:
            [
                {"role": "human", "content": <task instruction>},
                {"role": "assitant", "content": "Sure. Please give me the question."},
                {"role": "human", "content": <1st few-shot input>},
                {"role": "assistant", "content": <1st few-shot output>},
                ...
                {"role": "human", "content": <test question>},
            ]
        Therefore, we incoporate this function to handle such a case. It will interleave the
            few-shot examples in the above way and prompt the LLM to perform the task.

        Args:
            instruction: the full prompt for the task defined in `src/prompt.py`.
            initial_resp: the model's initial response (e.g., Sure. Please give me the question.)
            test_example: the testing question.
            split_tag: the split tag for instructions and few-shot examples in the full prompt
            example_split_tag: the split tag for few-shot examples in the prompt.
            example_start_tag: the prefix of model's answer, such as "Answer: ".
            temperature: the temperature for LLM decoding.
            max_tokens: maximum number of tokens the LLM can generate.
            n: number of examples to sample from the LLM.
        """
        instruction_str, example_str = instruction.split(split_tag, 1)
        instruction_str = instruction_str.strip()
        messages = [
            {"role": "user", "content": instruction_str},
            {"role": "assistant", "content": initial_resp},
        ]
        example_list = example_str.strip().split(example_split_tag)
        for example in example_list:
            lines = example.strip().split("\n")
            user_input = ""
            tgt_output = ""
            for line in lines:
                if line.startswith(example_start_tag):
                    user_input = line
                else:
                    tgt_output += line + "\n"
            tgt_output = tgt_output.strip()
            user_message = {"role": "user", "content": user_input}
            model_message = {"role": "assistant", "content": tgt_output}
            messages.append(user_message)
            messages.append(model_message)
        messages.append({"role": "user", "content": example_start_tag + test_example})
        response = self.query_fn(
            messages=messages, temperature=temperature, max_tokens=max_tokens, n=n,
        )
        return response
