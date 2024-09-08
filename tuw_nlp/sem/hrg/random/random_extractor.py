import argparse
import json
import os
import random
from collections import defaultdict

import stanza

from tuw_nlp.sem.hrg.common.predict import add_arg_idx
from tuw_nlp.sem.hrg.common.wire_extraction import get_wire_extraction
from tuw_nlp.text.utils import gen_tsv_sens

random.seed(42)


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-d", "--dev-fn", type=str)
    parser.add_argument("-o", "--out-fn", type=str)
    return parser.parse_args()


def main(dev_fn, out_fn, with_nr_ex_stat=True, verb_pred=True):
    stat_dir = os.path.dirname(os.path.realpath(__file__))
    train_seq_dist = f"{stat_dir}/train_stat/train_seq_dist.json"
    train_nr_ex = f"{stat_dir}/train_stat/train_nr_ex.json"
    with open(train_seq_dist) as f:
        seq_stat = json.load(f)
    with open(train_nr_ex) as f:
        nr_ex_stat = json.load(f)
    to_del = []
    for k, v in nr_ex_stat.items():
        if int(k) > 10:
            to_del.append(k)
    for k in to_del:
        del nr_ex_stat[k]
    print(f"stat len: {len(seq_stat)}")
    print(f"{sorted(seq_stat)}")
    extracted = defaultdict(list)
    random.seed(10)
    last_sen_txt = ""
    if verb_pred:
        nlp = stanza.Pipeline(
            lang="en",
            processors="tokenize,pos",
            tokenize_pretokenized=True,
        )

    with open(dev_fn) as f:
        for sen_idx, sen in enumerate(gen_tsv_sens(f)):
            print(f"Processing sen {sen_idx}")
            sen_txt = " ".join([line[1] for line in sen])
            if with_nr_ex_stat and sen_txt == last_sen_txt:
                continue
            sen_len = str(len(sen))
            nr_seq = len(seq_stat[sen_len])
            nr_to_gen = 1
            if with_nr_ex_stat:
                nr_ex = random.choices(list(nr_ex_stat.keys()), list(nr_ex_stat.values()))
                nr_to_gen = int(nr_ex[0])
            for i in range(nr_to_gen):
                rnd_idx = random.randrange(nr_seq)
                pred_seq = seq_stat[sen_len][rnd_idx]
                extracted_labels = {str(i+1): l for i, l in enumerate(pred_seq)}
                if verb_pred:
                    parsed_doc = nlp(sen_txt)
                    pos_tags = [w.pos for w in parsed_doc.sentences[0].words]
                    p_idx_l = [i for i, label in extracted_labels.items() if label == "P"]
                    verbs = [str((i+1)) for i, t in enumerate(pos_tags) if t == "VERB"]
                    if verbs:
                        for p_idx in p_idx_l:
                            if pos_tags[int(p_idx)-1] != "VERB":
                                new_p_idx = verbs[random.randrange(0, len(verbs))]
                                extracted_labels[new_p_idx] = "P"
                                extracted_labels[p_idx] = "O"
                add_arg_idx(extracted_labels, len(pred_seq))
                extracted[sen_txt].append(get_wire_extraction(extracted_labels, sen_txt, i+1, sen_idx, extractor="random"))
            last_sen_txt = sen_txt

    with open(out_fn,"w") as f:
        json.dump(extracted, f, indent=4)


if __name__ == "__main__":
    args = get_args()
    main(args.dev_fn, args.out_fn)
