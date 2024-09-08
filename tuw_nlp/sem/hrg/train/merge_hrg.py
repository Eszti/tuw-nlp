import argparse
import os
from collections import Counter, defaultdict


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-i", "--input-dir", type=str)
    parser.add_argument("-o", "--out-dir", type=str)
    parser.add_argument("-s", "--size", type=int)
    return parser.parse_args()


def cut_grammar(grammar, size):
    all_prods = get_total(grammar)
    factor = size / all_prods
    new_grammar = defaultdict(Counter)
    for nt, prods in grammar.items():
        for prod, cnt in prods.most_common(n=int(round(factor*len(prods)))):
            new_grammar[nt][prod] = cnt
    return new_grammar


def add_weights(grammar):
    new_grammar = defaultdict(list)
    for nt, prods in grammar.items():
        for prod, cnt in prods.most_common():
            w = round(cnt / prods.total(), 2)
            if w < 0.01:
                w = 0.01
            new_grammar[nt].append((prod, cnt, w))
    return new_grammar


def write_rules(f, grammar, numeric_info=None):
    for (prod, cnt, w) in grammar["S"]:
        if not numeric_info:
            f.write(f"{prod}\n")
        elif numeric_info == "cnt":
            f.write(f"{prod}\t{cnt}\n")
        elif numeric_info == "weight":
            f.write(f"{prod}\t{w}\n")
    for nt, prods in grammar.items():
        if nt == "S":
            continue
        for (prod, cnt, w) in prods:
            if not numeric_info:
                f.write(f"{prod}\n")
            elif numeric_info == "cnt":
                f.write(f"{prod}\t{cnt}\n")
            elif numeric_info == "weight":
                f.write(f"{prod}\t{w}\n")


def get_total(grammar):
    all_prods = 0
    for nt, prods in grammar.items():
        all_prods += len(prods)
    return all_prods


def main(input_dir, out_dir, size):
    grammar = defaultdict(Counter)
    for sen_dir in os.listdir(input_dir):
        filename = os.path.join(input_dir, sen_dir, f"sen{sen_dir}.hrg")
        if os.path.exists(filename):
            with open(filename) as f:
                lines = f.readlines()
            for line in lines:
                rule = line.strip()
                nt = line.split(" ")[0]
                grammar[nt][rule] += 1
    if size:
        grammar = cut_grammar(grammar, size)
    grammar = add_weights(grammar)
    size_str = str(size) if size is not None else "all"
    with open(f"{out_dir}/grammar_{size_str}.hrg", "w") as f:
        write_rules(f, grammar, "weight")
    with open(f"{out_dir}/grammar_{size_str}.stat", "w") as f:
        write_rules(f, grammar, "cnt")
    print(f"Unique rules: {get_total(grammar)}")
    for nt, prods in grammar.items():
        print(f"{nt}: {len(prods)}\t({round(len(prods)/get_total(grammar), 3)})")


if __name__ == "__main__":
    args = get_args()
    main(args.input_dir, args.out_dir, args.size)
