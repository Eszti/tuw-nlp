import argparse
import os
from collections import Counter, defaultdict


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-i", "--input-dir", type=str)
    parser.add_argument("-o", "--out-fn", type=str)
    parser.add_argument("-st", "--stat-fn", type=str)
    parser.add_argument("-s", "--size", type=int)
    return parser.parse_args()


def main(input_dir, out_fn, size, stat_fn):
    grammar = defaultdict(Counter)
    for sen_dir in os.listdir(input_dir):
        filename = os.path.join(input_dir, sen_dir, f"sen{sen_dir}.hrg")
        if os.path.exists(filename):
            with open(filename) as f:
                lines = f.readlines()
            for line in lines:
                rule = line.strip()
                nt = rule[0]
                grammar[nt][rule] += 1
    with open(out_fn, "w") as f:
        write_rule(f, grammar, "weight", size)
    if stat_fn:
        with open(stat_fn, "w") as f:
            write_rule(f, grammar, "cnt", size)
    print(f"Unique rules: {get_total(grammar)}")
    for nt, prods in grammar.items():
        print(f"{nt}: {len(prods)}")


def write_rule(f, grammar, numeric_info=None, size=None):
    factor = 1
    if size:
        all_prods = get_total(grammar)
        factor = size / all_prods
    for prod, cnt in grammar["S"].most_common(n=int(round(factor*len(grammar["S"])))):
        if not numeric_info:
            f.write(f"{prod}\n")
        elif numeric_info == "cnt":
            f.write(f"{prod}\t{cnt}\n")
        elif numeric_info == "weight":
            w = round(cnt / grammar["S"].total(), 2)
            if w < 0.01:
                w = 0.01
            f.write(f"{prod}\t{w}\n")
    for nt, prods in grammar.items():
        if nt == "S":
            continue
        for prod, cnt in prods.most_common(n=int(round(factor*len(prods)))):
            if not numeric_info:
                f.write(f"{prod}\n")
            elif numeric_info == "cnt":
                f.write(f"{prod}\t{cnt}\n")
            elif numeric_info == "weight":
                w = round(cnt / prods.total(), 2)
                if w < 0.01:
                    w = 0.01
                f.write(f"{prod}\t{w}\n")


def get_total(grammar):
    all_prods = 0
    for nt, prods in grammar.items():
        all_prods += len(prods)
    return all_prods


if __name__ == "__main__":
    args = get_args()
    main(args.input_dir, args.out_fn, args.size, args.stat_fn)
