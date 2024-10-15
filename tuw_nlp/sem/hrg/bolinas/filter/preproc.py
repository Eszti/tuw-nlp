import json
import os.path


def get_gold_labels(preproc_dir, sen_idx):
    gold_labels = []
    preproc_path = os.path.join(preproc_dir, str(sen_idx), "preproc")
    files = [fn for fn in os.listdir(preproc_path) if fn.endswith("_gold_labels.json")]
    for fn in files:
        with open(os.path.join(preproc_path, fn)) as f:
            gold_labels.append(json.load(f))
    return gold_labels
