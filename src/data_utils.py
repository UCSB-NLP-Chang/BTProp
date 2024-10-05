import json


def load_felm_science_processed_human():
    with open("eval_dataset/felm_science_preprocess_v4.json", "r", encoding="utf-8") as f:
        data_list = json.load(f)
    print("num loaded: ", len(data_list))
    return data_list


def load_wikibio_processed_human():
    # this example contains some words about the world war II and NAZI,
    # and chatgpt may refuse to answer it.
    skip_indices = [
        126,
    ]
    with open("eval_dataset/wikibio_preprocess_v2.json", "r", encoding="utf-8") as f:
        data_list = json.load(f)
    data_list = [data_list[x] for x in range(len(data_list)) if x not in skip_indices]
    return data_list


def load_factcheckgpt_processed_human():
    filepath = "eval_dataset/Factbench.jsonl"
    datalist = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if len(line) <= 0:
                continue
            data_dict = json.loads(line.strip())
            if data_dict["source"] == "factcheckgpt":
                process_data_dict = {
                    "prompt": data_dict["prompt"],
                    "response": data_dict["response"],
                    "segmented_response": data_dict["claims"],
                    "labels": data_dict["claim_labels"],
                }

                datalist.append(process_data_dict)
    return datalist
