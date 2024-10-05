## Hallucination Detection with Belief Tree Propagation
This is the official implementation for the paper, "A Probabilistic Framework for LLM Hallucination Detection via Belief Tree Propagation".

### Requirements

The dependency packages can be found in `requirements.txt` file. One can use `pip install -r requirements.txt` to configure the environment. We use python 3.10 to run the experiments.

### Running the experiments

The overall pipeline is: build the belief tree via prompting the LLM $\rightarrow$ prompt the LLM for its confidence score $\rightarrow$ compute the posterior probability $\rightarrow$ evaluate the performance. Before running experiments, you need to configure your OpenAI API key by setting the `OPENAI_API_KEY` environment variable.

1. Belief tree generation

```sh
python generate_belief_tree.py --dataset=wikibio --backbone=chatgpt
```
Use `python generate_belief_tree.py --helpfull` to see the choices for dataset and backbone.

By default, the generated belief trees will be stored at `logs/belief_trees/{dataset}_{backbone}.json`

2. Prompt the LLM for its confidence score
Similarly, you can specify the dataset name and the backbone LLM used for the experiment in the command line:
```sh
python confidence_estimation.py --dataset=wikibio --backbone=chatgpt
```
By default, the generated belief trees will be stored at `logs/conf_estimation/{dataset}_{backbone}.json`

3. Use the NLI model to label the edge type (the relationship between a parent node and a child node)
```sh
python tools/label_edges.py --dataset=wikibio --backbone=chatgpt
```

4. Compute the posterior probabilities
```sh
python hmm_forward.py --dataset=wikibio --backbone=chatgpt
```

5. Performance evaluation
```sh
python tools/compute_metrics.py --dataset=wikibio --backbone=chatgpt
```



### Citation

```
@article{hou2024probabilistic,
  title={A Probabilistic Framework for LLM Hallucination Detection via Belief Tree Propagation},
  author={Hou, Bairu and Zhang, Yang and Andreas, Jacob and Chang, Shiyu},
  journal={arXiv preprint arXiv:2406.06950},
  year={2024}
}```