from typing import List, Optional

from src import prompts
from src.inference_module import InferenceAPI
from src.model_utils import CrossEncoderNLI


def decompose_extraction(model_ans: str):
    all_lines = model_ans.strip().split("\n")
    premise_list = []
    for line in all_lines:
        if line.startswith("Claim"):
            if line.startswith("Claim: "):
                curr_premise = line[len("Claim: ") :].strip()
            else:
                curr_premise = line[len("Claim 1: ") :].strip()
            premise_list.append(curr_premise)
        else:
            continue
    return premise_list


def extract_reasoning(model_ans: str):
    lines = model_ans.strip().split("\n")
    all_exts = []
    conclusion = None
    for line in lines:
        line = line.strip()
        if line.startswith("Step"):
            ext = line[len("Step 1: ") :]
            all_exts.append(ext)
        elif line.startswith("Conclusion: "):
            conclusion = line[:]
        else:
            continue
    return all_exts, conclusion


class ClaimNode:
    def __init__(
        self,
        statement: str = None,
        can_be_expanded: bool = True,
        can_be_decomposed: bool = True,
        child_nodes: List[str] = [],
        prem_gen_method: str = None,
        initial_relations: List[str] = None,
        layer: int = 1,
    ) -> None:
        self.statement = statement
        self.can_be_expanded = can_be_expanded
        self.can_be_decomposed = can_be_decomposed
        self.child_nodes = child_nodes
        self.prem_gen_method = prem_gen_method
        self.initial_relations = initial_relations
        self.layer = layer

    def dumps(self):
        save_dict = {
            "statement": self.statement,
            "can_be_expanded": self.can_be_expanded,
            "can_be_decomposed": self.can_be_decomposed,
            "child_nodes": self.child_nodes,
            "prem_gen_method": self.prem_gen_method,
            "layer": self.layer,
        }
        if self.initial_relations is not None:
            save_dict["initial_relations"] = self.initial_relations
        return save_dict

    def get_expandable_childnodes(self):
        if self.can_be_expanded:
            return self.child_nodes
        else:
            return []


class GroupedNode:
    def __init__(self, tgt_statement: str, nodes: List[ClaimNode]) -> None:
        self.all_nodes = nodes
        self.root_statement = nodes[0].statement if len(nodes) >= 1 else tgt_statement
        pass

    def dumps(self):
        save_dicts = [x.dumps() for x in self.all_nodes]
        save_dict = {"statement": self.root_statement, "child_nodes": save_dicts}
        return save_dict

    def get_expandable_childnodes(self):
        expanable_nodes = []
        decompose_property = []
        for node in self.all_nodes:
            if node.can_be_expanded:
                expanable_nodes += node.child_nodes
                if node.can_be_decomposed:
                    decompose_property += [True for _ in range(len(node.child_nodes))]
                else:
                    decompose_property += [False for _ in range(len(node.child_nodes))]
            else:
                continue
        expanable_nodes = list(set(expanable_nodes))
        return expanable_nodes, decompose_property


class TreeGenerator:
    def __init__(self, backbone="chatgpt",) -> None:
        self.backbone = backbone
        self.inference_api = InferenceAPI(backbone)

        self.chatgpt_2nd_cls_prompt = prompts.chatgpt_select_strategy
        self.llama_2nd_cls_prompt = prompts.llama_select_strategy

        self.chatgpt_decompose_prompt = prompts.chatgpt_decompose
        self.llama_decompose = prompts.llama_decompose

        self.premise_prompt = prompts.premise_generation
        self.implication = prompts.implication_generation
        self.contradict_prompt = prompts.contradiction_generation

        self.support_via_explanation_prompt = prompts.support_via_explanation
        self.oppose_via_explanation_prompt = prompts.oppose_via_explanation

        self.prior_true_prompt = prompts.prior_true_prompt
        self.prior_false_prompt = prompts.prior_false_prompt

        self.chatgpt_perturb_prompt = prompts.chatgpt_perturb
        self.chatgpt_perturb_answer_prompt = prompts.chatgpt_perturb_answer
        self.chatgpt_perturb_fillin_prompt = prompts.chatgpt_perturb_fillin
        self.llama_perturb = prompts.llama_perturb
        self.llama_perturb_fillin_prompt = prompts.llama_perturb_fillin

        self.unk_word_pool = [
            "I don't",
            "Not specified",
            "cannot be determined",
            "No Answer",
            "No final answer",
            "I do not",
            "N/A",
            "No information",
            "I cannot",
            "I can't ",
            "unable to",
            "don't know",
            "No answer",
            "Nobody",
            "enough information",
            "specific information",
            "No specific",
            "not provided",
            "None",
            "No character",
            "No output",
            "Cannot answer",
            "Unavailable",
            "TBD",
            "To Be Determined",
            "I am asking about",
            "current year",
            "unclear",
            "confusion",
            "not aware of",
            "invalid",
            "additional context",
            "further context",
            "additional information",
            "Unable to determine",
            "additional information",
            "further information",
        ]
        self.unk_word_pool = [x.lower() for x in self.unk_word_pool]

        self.nli_model = CrossEncoderNLI()

    def strategy_selection(
        self, tgt_statement: str, verbose=False,
    ):
        """Select the child node generation method for a given statement."""
        if self.backbone == "chatgpt":
            second_cls_query = (
                self.chatgpt_2nd_cls_prompt + "\n\nTarget statement: " + tgt_statement
            )
            second_cls_response = self.inference_api.query(
                prompt=second_cls_query, temperature=0.0, max_tokens=30, n=1,
            )
        elif self.backbone == "llama":
            initial_resp = "Yes, I understand the task."
            " Please provide me with the target statement."
            second_cls_response = self.inference_api.query_with_split_examples(
                instruction=self.llama_2nd_cls_prompt[:],
                initial_resp=initial_resp,
                test_example=tgt_statement,
                split_tag="**Examples:**",
                example_split_tag="\n\n",
                example_start_tag="Target statement: ",
                temperature=0.0,
                max_tokens=30,
                n=1,
            )

        if second_cls_response is None or second_cls_response.choices[0].message.content is None:
            return None
        ans_model = second_cls_response.choices[0].message.content.lower()
        if verbose:
            print(ans_model)
        if "perturb" in ans_model:
            return ["statement perturbation"]
        elif "logic" in ans_model:
            return ["logical relation"]
        elif "both" in ans_model:
            return ["logical relation", "statement perturbation"]
        else:
            print("invalid 2nd selection: ", ans_model)
            return ["logical relation", "statement perturbation"]

    def premise_generate_chatgpt(self, tgt_statement: str, layer=1, can_be_decompose=True):
        """
        Recursively generate the child node for a given statement.

        Args:
            tgt_statement: the statement on the given child node.
            layer: the layer index for the given node. 1 means the root node.
            can_be_decompose: whether this node can be decomposed. If it is decomposed
                from another node, then it should not be further decomposed.

        Returns:
            A GroupedNode object containing the generated child nodes.
        """
        if can_be_decompose:
            expand_node = self.decompose_statement(tgt_statement, layer=layer)
            if expand_node is not None and len(expand_node.child_nodes) > 1:
                return GroupedNode(tgt_statement, [expand_node])
            elif expand_node is not None and len(expand_node.child_nodes) == 1:
                if "No check-worthy claims" in expand_node.child_nodes[0]:
                    return None

        selected_strategies = self.strategy_selection(tgt_statement)
        all_nodes = []
        for strategy in selected_strategies:
            if strategy == "logical relation":
                expand_node = self.expand_via_logic_relation(tgt_statement, layer=layer)
            elif strategy == "statement perturbation":
                expand_node = self.statement_correction(tgt_statement, layer=layer)
            else:
                expand_node = None
            if expand_node is not None:
                verify_result = self.node_verification(expand_node)
                if verify_result:
                    all_nodes.append(expand_node)

        grouped_node = GroupedNode(tgt_statement, all_nodes)
        return grouped_node

    def node_verification(self, node: Optional[ClaimNode]):
        """
        Verify if the generated child nodes are valid:
            - node is not None
            - node contains at least 1 child node
        """
        if node is None or node.child_nodes is None:
            return False
        if len(node.child_nodes) == 0:
            return False
        return True

    def filter_claims(self, prem_list: List[str]):
        """Use rule-based method to filter the invalid child nodes."""
        prems_after_filer = []
        for prem in prem_list:
            if "[" in prem and "]" in prem:
                continue
            normalized_prem = prem.lower()
            flag = True
            for unk_w in self.unk_word_pool:
                if unk_w in normalized_prem:
                    flag = False
                    break
            if flag:
                prems_after_filer.append(prem)
        return prems_after_filer

    def remove_unqualified_claims(
        self, parent_node: str, prem_list: List[str], verbose=False,
    ):
        """
        Given a parent node and the generated child nodes, remove the irrelevant child nodes
            according to the NLI relationship.

        Args:
            parent_node: the statement on the parent node.
            prem_list: a list of statement from the generated child nodes.
            verbose: whether output the intermediate information during filtering

        Returns:
            a list of str, each element is a relevant child node.
        """
        if len(prem_list) == 0:
            return prem_list
        logits = self.nli_model.judge_relation(parent_node=parent_node, child_nodes=prem_list)
        pred_nlis = self.nli_model.decode_relation(logits)
        prems_after_filer = []
        assert len(pred_nlis) == len(prem_list)
        for prem_idx, prem in enumerate(prem_list):
            remove = False
            if pred_nlis[prem_idx] == "neutral-neutral":
                remove = True

            if verbose and remove:
                print(f"prem {prem_idx}: {prem}")
                print(f"relation: {pred_nlis[prem_idx]}")
            if remove:
                continue
            prems_after_filer.append(prem)

        return prems_after_filer

    def expand_via_logic_relation(self, tgt_statement: str, layer=1, verbose=False):
        """Generate child nodes using logical relationships."""
        all_contradicts = []
        all_supports = []

        support_query = self.support_via_explanation_prompt[:] + "\n\nStatement: " + tgt_statement
        support_response = self.inference_api.query(
            support_query, temperature=0.7, max_tokens=512, n=2,
        )
        if support_response is None or support_response.choices[0].message.content is None:
            pass
        else:
            for sample_id in range(len(support_response.choices)):
                model_ans = support_response.choices[sample_id].message.content
                if model_ans is None:
                    continue
                lines = model_ans.split("\n")
                for line in lines:
                    line = line.strip()
                    if line.startswith("Premise"):
                        ext = line[len("Premise 1: ") :].strip()
                        if "premises applicable" in ext:
                            continue
                        all_supports.append(ext)
        # If the model generates no supportive premises, we prompt it again for contradictions
        if len(all_supports) == 0:
            contradict_query = (
                self.oppose_via_explanation_prompt[:] + "\n\nStatement: " + tgt_statement
            )
            contradict_response = self.inference_api.query(
                prompt=contradict_query, temperature=0.7, max_tokens=512, n=2
            )
            if (
                contradict_response is None
                or contradict_response.choices[0].message.content is None
            ):
                pass
            else:
                for sample_id in range(len(contradict_response.choices)):
                    model_ans = contradict_response.choices[sample_id].message.content
                    if model_ans is None:
                        continue
                    lines = model_ans.split("\n")
                    for line in lines:
                        line = line.strip()
                        if line.startswith("Premise"):
                            ext = line[len("Premise 1: ") :].strip()
                            if "premises applicable" in ext:
                                continue
                            all_contradicts.append(ext)
                else:
                    all_contradicts = []

        all_logic_premises = all_contradicts + all_supports
        all_logic_premises = self.remove_unqualified_claims(
            tgt_statement, all_logic_premises, verbose=verbose
        )
        all_logic_premises = all_logic_premises[:3]

        print("layer ", layer, "premise num: ", len(all_logic_premises))
        all_relations = ["premise" for _ in range(len(all_logic_premises))]

        claimnode = ClaimNode(
            statement=tgt_statement,
            can_be_expanded=True,
            can_be_decomposed=False,
            child_nodes=all_logic_premises,
            prem_gen_method="logical relation",
            initial_relations=all_relations,
            layer=layer,
        )
        if verbose:
            for prem_id, prem in enumerate(all_logic_premises):
                print(f"premise {prem_id}", ": ", prem)

        return claimnode

    def decompose_statement(self, tgt_statement: str, layer=1, verbose=False):
        """Decompose a given statement into sub-statements."""
        if self.backbone == "chatgpt":
            prem_gen_prompt = self.chatgpt_decompose_prompt
            prem_gen_query = prem_gen_prompt + "\n\nStatement: " + tgt_statement
            prem_gen_response = self.inference_api.query(
                prompt=prem_gen_query, temperature=0.0, max_tokens=300, n=1
            )
        elif self.backbone == "llama":
            initial_resp = "Yes, I understand the task. You can provide the statement"
            " you'd like me to analyze for extracting verifiable factual claims."
            prem_gen_response = self.inference_api.query_with_split_examples(
                instruction=self.llama_decompose[:],
                initial_resp=initial_resp,
                test_example=tgt_statement,
                split_tag="**Examples:**",
                example_split_tag="\n\n",
                example_start_tag="Statement: ",
                temperature=0.0,
                max_tokens=300,
                n=1,
            )
        else:
            raise NotImplementedError
        if self.backbone == "chatgpt":
            if prem_gen_response is None or prem_gen_response.choices[0].message.content is None:
                return None
        ans_model = prem_gen_response.choices[0].message.content
        if verbose:
            print(tgt_statement)
            print(ans_model)
            print("---" * 10)

        gen_prems = decompose_extraction(ans_model)
        recovered_gen_prems = gen_prems
        recovered_gen_prems = self.filter_claims(recovered_gen_prems)
        relations = ["decompose" for _ in range(len(recovered_gen_prems))]
        claimnode = ClaimNode(
            statement=tgt_statement,
            can_be_expanded=True,
            can_be_decomposed=False,
            child_nodes=recovered_gen_prems,
            prem_gen_method="decompose",
            initial_relations=relations,
            layer=layer,
        )

        return claimnode

    def statement_correction(self, tgt_statement: str, layer=1, verbose=False):
        """Generate child nodes by asking the LLM to correct the statement."""
        if self.backbone == "chatgpt":
            mask_query = self.chatgpt_perturb_prompt + f"\n\nStatement: {tgt_statement}"
            response = self.inference_api.query(
                prompt=mask_query, temperature=0.0, max_tokens=300, n=1
            )
            if response is None or response.choices[0].message.content is None:
                return None
        elif self.backbone == "llama":
            initial_resp = (
                "Yes, I understand the task. Please provide me with the target statement."
            )
            response = self.inference_api.query_with_split_examples(
                instruction=self.llama_perturb[:],
                initial_resp=initial_resp,
                test_example=tgt_statement,
                split_tag="**Examples:**",
                example_split_tag="\n\n",
                example_start_tag="Statement: ",
                temperature=0.0,
                max_tokens=300,
                n=1,
            )
        else:
            raise NotImplementedError

        ans_model = response.choices[0].message.content

        lines = ans_model.strip().split("\n")
        question = None
        masked_sentence = None
        for line in lines:
            if line.startswith("Masked statement"):
                masked_sentence = line.strip("Masked statement: ")
            elif line.startswith("Question"):
                question = line.strip("Question: ")
        if question is None:
            print("invalid perturbation: ", ans_model)
            return None

        qa_query = question
        response = self.inference_api.query(prompt=qa_query, temperature=0.7, max_tokens=200, n=5)
        if response is None:
            return None
        answers = [response.choices[x].message.content for x in range(len(response.choices))]

        fillin_sentences = []
        for ans in answers:
            if ans is None:
                continue
            if self.backbone == "chatgpt":
                fillin_query = self.chatgpt_perturb_fillin_prompt.format(
                    background=ans, statement=tgt_statement
                )
            elif self.backbone == "llama":
                fillin_query = self.llama_perturb_fillin_prompt.format(
                    background=ans, statement=tgt_statement
                )
            else:
                raise NotImplementedError
            response = self.inference_api.query(
                prompt=fillin_query, temperature=0.0, max_tokens=100, n=1
            )
            if self.backbone == "chatgpt":
                if response is None or response.choices[0].message.content is None:
                    return None
            model_output = response.choices[0].message.content
            model_output = model_output.strip()
            if self.backbone == "llama":
                model_output = model_output.replace("*", "")
            if model_output.startswith("Revised Answer:"):
                fillin_sentence = model_output[len("Revised Answer:") :].strip()
            elif model_output.startswith("Revised answer:"):
                fillin_sentence = model_output[len("Revised answer:") :].strip()
            elif model_output.startswith("Revised statement:"):
                fillin_sentence = model_output[len("Revised statement:") :].strip()
            elif "Revised statement: " in model_output:
                fillin_sentence = model_output.split("Revised statement: ")[-1].strip()
            else:
                continue
            fillin_sentences.append(fillin_sentence)

        fillin_sentences = self.filter_claims(fillin_sentences)
        if verbose:
            print(qa_query)
            print(masked_sentence)
            print(answers)
            print()
            print(fillin_query)
            print(fillin_sentences)

        claimnode = ClaimNode(
            statement=tgt_statement,
            can_be_expanded=False,
            child_nodes=fillin_sentences,
            prem_gen_method="perturbation",
            initial_relations=None,
            layer=layer,
        )
        return claimnode

    def chatgpt_generate_pipeline(
        self, root_statement: str, root_can_be_decompose=True, max_layers: int = 2,
    ):
        """
        Generate the whole belief tree for a given statement.

        Args:
            root_statement: the target sentence to be evaluated.
            root_can_be_decompose: whether the target sentence can be decomposed
                into multiple atomic statements. Only set this to be true for
                FactCheckGPT dataset, as it has been processed to ensure each sentence
                is about an atomic fact.

        Returns:
            premise_gen_logs: a list of dict, each element contains the parent node and
                the corresponding generated child nodes.
        """

        premise_gen_logs = []
        prem_layer_dict = {}

        to_generate = [root_statement]
        visited_nodes = []
        can_be_decompose_dict = {root_statement: root_can_be_decompose}

        while True:
            if len(to_generate) == 0:
                break
            curr_sentence = to_generate[0]

            if curr_sentence not in prem_layer_dict:
                prem_layer_dict[curr_sentence] = 1
                curr_layer = 1
            else:
                curr_layer = prem_layer_dict[curr_sentence]
            if curr_layer > max_layers:
                removed = to_generate.pop(0)
                visited_nodes.append(removed)
                continue

            if curr_sentence in visited_nodes:
                to_generate.pop(0)
                continue

            can_decompose = can_be_decompose_dict[curr_sentence]
            grouped_node = self.premise_generate_chatgpt(
                curr_sentence, layer=curr_layer, can_be_decompose=can_decompose
            )
            visited_nodes.append(curr_sentence)
            if grouped_node is None:
                to_generate.pop(0)
                continue
            new_child_nodes, can_be_decompose_list = grouped_node.get_expandable_childnodes()

            to_generate.pop(0)
            for idx, prem in enumerate(new_child_nodes):
                to_generate.insert(0, prem)
                prem_layer_dict[prem] = curr_layer + 1
                can_be_decompose_dict[prem] = can_be_decompose_list[idx]

            premise_gen_logs.append(grouped_node.dumps())

        return premise_gen_logs
