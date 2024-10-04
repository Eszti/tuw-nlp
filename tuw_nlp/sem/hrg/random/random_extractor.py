import argparse
import json
import os
import random
import sys
from collections import defaultdict

import stanza

from tuw_nlp.sem.hrg.common.conll import get_sen_from_conll_sen
from tuw_nlp.sem.hrg.common.io import create_sen_dir
from tuw_nlp.sem.hrg.common.wire_extraction import get_wire_extraction
from tuw_nlp.sem.hrg.predict.postprocess import add_arg_idx
from tuw_nlp.text.utils import gen_tsv_sens


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-d", "--data-dir", type=str)
    parser.add_argument("-c", "--config", type=str)
    return parser.parse_args()


def main(data_dir, config_json, with_nr_ex_stat=True, verb_pred=True):
    random.seed(10)

    stat_dir = os.path.dirname(os.path.realpath(__file__))
    config = json.load(open(config_json))
    k_max = config.get("k_max", 10)
    non_verbs = defaultdict(list)

    train_nr_ex = f"{stat_dir}/train_stat/train_nr_ex.json"
    with open(train_nr_ex) as f:
        nr_ex_stat = json.load(f)
    to_del = []
    for k, v in nr_ex_stat.items():
        if int(k) > k_max:
            to_del.append(k)
    for k in to_del:
        del nr_ex_stat[k]

    if verb_pred:
        nlp = stanza.Pipeline(
            lang="en",
            processors="tokenize,pos",
            tokenize_pretokenized=True,
        )

    seq_stats = {}
    for m in config["models"]:
        seq_stats[m] = json.load(open(f"{stat_dir}/train_stat/train_seq_dist_{m}.json"))
    for m, seq_stat in seq_stats.items():
        print(f"{m}\nstat len: {len(seq_stat)}")
        print(f"{sorted(seq_stat)}")

    last_sen = ""
    first = config.get("first", None)
    last = config.get("last", None)
    for sen_idx, sen in enumerate(gen_tsv_sens(sys.stdin)):
        if first is not None and sen_idx < first:
            continue
        if last is not None and last < sen_idx:
            break
        sen_txt = get_sen_from_conll_sen(sen)
        if with_nr_ex_stat and sen_txt == last_sen:
            continue
        last_sen = sen_txt

        print(f"Processing sen {sen_idx}")
        sen_dir = create_sen_dir(f"{data_dir}/{config['out_dir']}", sen_idx)
        predict_dir = os.path.join(sen_dir, "predict")

        for m, seq_stat in seq_stats.items():
            model_dir = f"{predict_dir}/{m}"
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            wire_fn = f"{model_dir}/sen{sen_idx}_wire.json"
            get_extractions(
                m,
                seq_stat,
                nlp,
                nr_ex_stat,
                sen,
                sen_idx,
                sen_txt,
                verb_pred,
                with_nr_ex_stat,
                non_verbs,
                wire_fn,
            )
    print(non_verbs)


def get_extractions(
        model,
        seq_stat,
        nlp,
        nr_ex_stat,
        sen,
        sen_idx,
        sen_txt,
        verb_pred,
        with_nr_ex_stat,
        non_verbs,
        wire_fn,
):
    extracted = defaultdict(list)

    sen_len_for_stat = str(len(sen))
    if sen_len_for_stat not in seq_stat:
        while sen_len_for_stat not in seq_stat:
            sen_len_for_stat = str(int(sen_len_for_stat) - 1)
    nr_seq = len(seq_stat[sen_len_for_stat])

    nr_to_gen = 1
    if with_nr_ex_stat:
        nr_ex = random.choices(list(nr_ex_stat.keys()), list(nr_ex_stat.values()))
        nr_to_gen = int(nr_ex[0])

    used_rnd_indexes = set()
    for i in range(nr_to_gen):
        rnd_idx = random.randrange(nr_seq)
        while rnd_idx in used_rnd_indexes:
            rnd_idx = random.randrange(nr_seq)
        used_rnd_indexes.add(rnd_idx)

        pred_seq = seq_stat[sen_len_for_stat][rnd_idx]
        if len(sen) > len(pred_seq):
            pred_seq += "_"
            pred_seq += "_".join(["O"] * (len(sen) - len(pred_seq)))

        extracted_labels = {str(i + 1): l for i, l in enumerate(pred_seq.split("_"))}
        if verb_pred:
            parsed_doc = nlp(sen_txt)
            pos_tags = [w.pos for w in parsed_doc.sentences[0].words]
            p_idx_l = [i for i, label in extracted_labels.items() if label == "P"]
            verbs = [str((i + 1)) for i, t in enumerate(pos_tags) if t == "VERB"]
            if verbs:
                for p_idx in p_idx_l:
                    if pos_tags[int(p_idx) - 1] != "VERB":
                        new_p_idx = verbs[random.randrange(0, len(verbs))]
                        extracted_labels[new_p_idx] = "P"
                        extracted_labels[p_idx] = "O"
            else:
                non_verbs[sen_idx].append((p_idx_l, pos_tags))
        if model == "boa":
            add_arg_idx(extracted_labels, len(pred_seq))
        extracted[sen_txt].append(get_wire_extraction(
            extracted_labels,
            sen_txt, i + 1,
            sen_idx,
            extractor=f"random_{model}"
        ))
    with open(wire_fn, "w") as f:
        json.dump(extracted, f, indent=4)


if __name__ == "__main__":
    args = get_args()
    main(args.data_dir, args.config)
