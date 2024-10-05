import itertools
from typing import Optional

import numpy as np
import yaml

np.set_printoptions(precision=6, suppress=True)


class HMMTreeNode:
    def __init__(self, config_path: str, statement: str, model_confidence: float, layer=0) -> None:
        self.config_path = config_path
        self.load_config(config_path)
        self.statement = statement
        # list of HMMTreeNode
        self.child_nodes = []
        # NLI labels
        self.nli_relations = []
        # general relations by LLM
        self.general_relations = []
        # confidence score of the LLM
        self.prior_confidence = model_confidence
        # the layer index of the current node. 0 means the root node.
        self.layer = layer
        # A unique index for the node in the belief tree.
        self.node_idx = None

        # given the current node hidden is true, the condition prob of the
        # observation tree start from current node
        self.beta_true = None
        # given the current node hidden is false, the condition prob of the
        # observation tree start from current node.
        self.beta_false = None
        # given the current node hidden is true, the condition prob of the
        # observation tree start from each child node
        self.tilde_beta_true = []
        # given the current node hidden is false, the condition prob of the
        # observation tree start from each child node
        self.tilde_beta_false = []
        # given the parent hidden is true, the condition prob of the
        # hidden child is true
        self.all_p_child_given_parent_hidden_true = []
        # given the parent hidden is true, the condition prob of the
        # hidden child is false
        self.all_p_child_given_parent_hidden_false = []

    def load_config(self, config_path: str):
        assert config_path is not None
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = yaml.load(f, Loader=yaml.SafeLoader)
        self.config_dict = config_dict
        # The prior probability of a node being true
        self.HMM_init_distribution_true = config_dict["HMM_init_distribution_true"]
        # The prior probability of a node being false
        self.HMM_init_distribution_false = config_dict["HMM_init_distribution_false"]

        self.bins_sep = config_dict["bins_sep"]

        self.emission_probs_true = config_dict["emission_probs_true"]
        self.emission_probs_false = config_dict["emission_probs_false"]

        self.emission_probs_true_prior = config_dict.get(
            "prior_emission_probs_true", self.emission_probs_true
        )
        self.emission_probs_false_prior = config_dict.get(
            "prior_emission_probs_false", self.emission_probs_false
        )

    def update_node_id(self):
        def traverse(node: HMMTreeNode, prev_num):
            if len(node.child_nodes) == 0:
                return prev_num
            for childnode in node.child_nodes:
                prev_num += 1
                childnode.node_idx = prev_num
            for childnode in node.child_nodes:
                new_num = traverse(childnode, prev_num)
                prev_num = new_num
            return prev_num

        if self.layer == 0:
            self.node_idx = 1
        traverse(self, prev_num=1)

    def get_all_leaf_nodes(self):
        leaf_nodes = []

        def traverse(node: HMMTreeNode):
            if len(node.child_nodes) == 0:
                leaf_nodes.append(node.statement)
                return
            else:
                for c_node in node.child_nodes:
                    traverse(c_node)

        traverse(self)
        return leaf_nodes

    def print(self, printall=True, posteval=False):
        """
        A simple function to print the node attributes in the belief tree.
        """

        def print_tree(node: HMMTreeNode, print_obj: str):
            """
            Recursively visit all nodes in the belief tree and print the attribute
            specified by `print_obj`.

            Args:
                node: a HMMTreeNode object
                print_obj: select from `statement`, `index`, `state`, `upward`
            """
            if print_obj == "statement":
                if node.layer == 0:
                    print(f"RootNode: {node.statement}")

            if print_obj == "upward":
                if len(node.child_nodes) != 0:
                    print(f"{node.node_idx}")
                    print(f"    beta_true: {node.beta_true}")
                    print(f"    beta_false: {node.beta_false}")
                    print(f"    tilde_beta_true: {node.tilde_beta_true}")
                    print(f"    tilde_beta_false: {node.tilde_beta_false}")
                    print()

            if len(node.child_nodes) == 0:
                return
            for idx, childnode in enumerate(node.child_nodes):
                if print_obj == "index":
                    print(
                        f"{node.node_idx} --> {childnode.node_idx}: "
                        f"{node.nli_relations[idx]} + {node.general_relations[idx]}"
                    )
                elif print_obj == "statement":
                    print(f"Node {childnode.node_idx}: {childnode.statement}")
                elif print_obj == "state":
                    print(
                        f"State of node {childnode.node_idx}: "
                        f"confidence {childnode.prior_confidence}"
                    )
                elif print_obj == "upward":
                    continue
                else:
                    raise NotImplementedError

            for childnode in node.child_nodes:
                print_tree(childnode, print_obj=print_obj)

        if self.node_idx is None:
            self.update_node_id()
        if printall:
            print_tree(self, print_obj="index")
            print_tree(self, print_obj="statement")
            print_tree(self, print_obj="state")
        if posteval:
            print_tree(self, print_obj="upward")

    def add_node_via_statement(
        self,
        parent_statement: str,
        child_statement: str,
        conf: float,
        nli_rel: str,
        general_rel: str,
    ):
        """
        Given a parent node and a child node, search in the belief tree and mount the child node
        under the parent node.

        Args:
            parent_statement: the statement on the parent node.
            child_statement: the statement on the child node.
            conf: the confidence score on the child node.
            nli_rel: the NLI relation between the parent node and the child node.
            general_rel: the premise generation method for that child node.

        Returns:
            layer_index: the layer index of the child node. -1 means the parent node
                is not found in the tree.
        """
        if parent_statement.strip().lower() == child_statement.strip().lower():
            ## do not add the node
            return 0
        if self.statement == parent_statement:
            newnode = HMMTreeNode(
                config_path=self.config_path,
                statement=child_statement,
                model_confidence=conf,
                layer=self.layer + 1,
            )
            self.child_nodes.append(newnode)
            self.nli_relations.append(nli_rel)
            self.general_relations.append(general_rel)
            return newnode.layer
        else:
            for childnode in self.child_nodes:
                insert_layer = childnode.add_node_via_statement(
                    parent_statement, child_statement, conf, nli_rel, general_rel
                )
                if insert_layer > 0:
                    return insert_layer
            return -1

    def get_p_observation_given_hidden(
        self, hidden_state: bool, observation: Optional[float],
    ):
        """
        Compute p(S_u|Z_u), the probability of the confidence score given the hidden states of
            the node (true or false).

        Args:
            hidden_state: the hidden state of the current node (true or false)
            observation: the value observation variable (i.e, LLM's confidence on the node)

        Returns:
            emission_probability: p(S_u = observation | Z_u = hidden_state)
        """
        if observation is None:
            return 1.0
        bin_idx = -1
        if observation < self.bins_sep[0]:
            bin_idx = 0
        else:
            for idx in range(len(self.bins_sep)):
                if idx == len(self.bins_sep) - 1:
                    end_point = 1.1
                else:
                    end_point = self.bins_sep[idx + 1]
                if observation > self.bins_sep[idx] and observation < end_point:
                    bin_idx = idx + 1
                    break
        if hidden_state:
            return self.emission_probs_true[bin_idx]
        else:
            return self.emission_probs_false[bin_idx]

    def get_p_child_state_given_parent_state(
        self, nli_rel: str, parent_state: bool, child_state: bool
    ):
        """
        Compute the transition probability between the parent node and the child node.

        Args:
            nli_rel: parent-to-child | child-to-parent nli relations, seprated by `-`
            parent_state: the hidden state of the parent node (true or false)
            child_state: the hidden state of the child node (true or false)
        """
        if parent_state:
            return self.get_p_child_state_given_parent_true(nli_rel, child_state)
        else:
            return self.get_p_child_state_given_parent_false(nli_rel, child_state)

    def get_p_child_state_given_parent_true(self, nli_rel: str, child_state: bool):
        if nli_rel == "entail-entail":
            if child_state:
                return 1.0
            else:
                return 0.0

        elif nli_rel == "entail-neutral":
            if child_state:
                return 1.0
            else:
                return 0.0

        elif "contradict" in nli_rel:
            if child_state:
                return 0.0
            else:
                return 1.0

        else:
            if child_state:
                return self.HMM_init_distribution_true
            else:
                return self.HMM_init_distribution_false

    def get_p_child_state_given_parent_false(self, nli_rel: str, child_state: bool):
        if nli_rel == "entail-entail":
            if child_state:
                return 0.0
            else:
                return 1.0

        elif nli_rel == "neutral-entail":
            if child_state:
                return 0.0
            else:
                return 1.0

        else:
            if child_state:
                return self.HMM_init_distribution_true
            else:
                return self.HMM_init_distribution_false

    def get_beta(self, parent_state: bool):
        """
        Compute beta(z, u) = p(S_{\tau{u}} | Z_u = parent_state), which is the joint probability
            of the child nodes confidence scores given the state of their parent node, u.

        Args:
            parent_state: the hidden state of the parent node (true or false).
        Returns:
            the probability beta(z, u).
        """
        if parent_state:
            p_tree_obervation_given_hidden = self.get_p_observation_given_hidden(
                True, observation=self.prior_confidence
            )
        else:
            p_tree_obervation_given_hidden = self.get_p_observation_given_hidden(
                False, observation=self.prior_confidence
            )

        # no child nodes: directly return P(confidence|hidden)
        if len(self.child_nodes) == 0:
            return p_tree_obervation_given_hidden

        list_p_childtree_observation_given_hidden = []
        list_p_child_given_parent_hidden = []

        rel_with_first_child = self.general_relations[0]
        if rel_with_first_child == "decompose" and not parent_state:
            child_observation_prob = self.get_cond_dependent_beta_parent_false()
            list_p_child_given_parent_hidden.append(child_observation_prob)
        elif rel_with_first_child == "decompose" and parent_state:
            child_observation_prob = self.get_cond_dependent_beta_parent_true()
            list_p_child_given_parent_hidden.append(child_observation_prob)
        else:
            for child_idx, childnode in enumerate(self.child_nodes):

                child_p_tree_obervation_given_hidden_false = childnode.get_beta(False)
                p_child_false_given_parent_state = self.get_p_child_state_given_parent_state(
                    nli_rel=self.nli_relations[child_idx],
                    parent_state=parent_state,
                    child_state=False,
                )

                child_p_tree_obervation_given_hidden_true = childnode.get_beta(True)
                p_child_true_given_parent_state = self.get_p_child_state_given_parent_state(
                    nli_rel=self.nli_relations[child_idx],
                    parent_state=parent_state,
                    child_state=True,
                )

                curr_p_childtree_observation_given_hidden = (
                    child_p_tree_obervation_given_hidden_false * p_child_false_given_parent_state
                    + child_p_tree_obervation_given_hidden_true * p_child_true_given_parent_state
                )
                list_p_childtree_observation_given_hidden.append(
                    curr_p_childtree_observation_given_hidden
                )
                list_p_child_given_parent_hidden.append(
                    (p_child_false_given_parent_state, p_child_true_given_parent_state)
                )

            child_observation_prob = 1
            for prob in list_p_childtree_observation_given_hidden:
                child_observation_prob *= prob
        total_prob = child_observation_prob * p_tree_obervation_given_hidden
        # *10 to prevent numerical underflow. The posterior probability will not
        # be influenced as it is the division between two probabilities, both of
        # which are enlarged by 10 times
        total_prob = total_prob * 10

        list_p_childtree_observation_given_hidden.append(p_tree_obervation_given_hidden)

        if parent_state:
            self.all_p_child_given_parent_hidden_true = np.array(list_p_child_given_parent_hidden)
            self.tilde_beta_true = np.array(list_p_childtree_observation_given_hidden)
            self.beta_true = total_prob
        else:
            self.all_p_child_given_parent_hidden_false = np.array(list_p_child_given_parent_hidden)
            self.tilde_beta_false = np.array(list_p_childtree_observation_given_hidden)
            self.beta_false = total_prob

        return total_prob

    def get_cond_dependent_beta_parent_false(self):
        """
        For child node generated by statement decomposition, compute the probability
            beta(z=false, u) = p(S_{\tau{u}} | Z_u = false).

        Returns:
            the probability beta(z=false, u).
        """
        child_p_tree_obervation_given_hidden_false_list = []
        child_p_tree_obervation_given_hidden_true_list = []
        for child_idx, childnode in enumerate(self.child_nodes):
            child_p_tree_obervation_given_hidden_false = childnode.get_beta(False)
            child_p_tree_obervation_given_hidden_false_list.append(
                child_p_tree_obervation_given_hidden_false
            )

            child_p_tree_obervation_given_hidden_true = childnode.get_beta(True)
            child_p_tree_obervation_given_hidden_true_list.append(
                child_p_tree_obervation_given_hidden_true
            )

        all_states = [[False, True] for _ in range(len(self.child_nodes))]
        combinations = list(itertools.product(*all_states))[:-1]
        for x in combinations:
            assert False in x

        single_prob = 1.0 / len(combinations)

        list_child_p_tree_obervation_given_hidden = []
        for combination in combinations:  ## a list of child node true/false
            prob = single_prob
            for child_idx, child_state in enumerate(combination):
                if child_state:
                    prob *= child_p_tree_obervation_given_hidden_true_list[child_idx]
                else:
                    prob *= child_p_tree_obervation_given_hidden_false_list[child_idx]
            list_child_p_tree_obervation_given_hidden.append(prob)
        return np.sum(list_child_p_tree_obervation_given_hidden)

    def get_cond_dependent_beta_parent_true(self):
        """
        For child node generated by statement decomposition, compute the probability
            beta(z=true, u) = p(S_{\tau{u}} | Z_u = true).

        Returns:
            the probability beta(z=true, u).
        """
        child_p_tree_obervation_given_hidden_true_list = []
        for _child_idx, childnode in enumerate(self.child_nodes):
            child_p_tree_obervation_given_hidden_true = childnode.get_beta(True)
            child_p_tree_obervation_given_hidden_true_list.append(
                child_p_tree_obervation_given_hidden_true
            )
        prob = 1.0
        for child_prob in child_p_tree_obervation_given_hidden_true_list:
            prob *= child_prob
        return prob

    def get_posterior(self,):
        """
        Compute the posterior probability of the statement being true. After we have defined
            `get_beta` which returns beta(z, u), the computation of the posterior probability
            will be quite straightforward.

        Returns:
            the posterior probability of the root node being true given the confidence scores
                of the child nodes.
        """
        cond_true_prob = self.get_beta(True)
        cond_false_prob = self.get_beta(False)
        joint_true_prob = cond_true_prob * self.HMM_init_distribution_true
        joint_false_prob = cond_false_prob * self.HMM_init_distribution_false

        posteior_true_prob = joint_true_prob / (joint_true_prob + joint_false_prob + 1e-15)
        return posteior_true_prob
