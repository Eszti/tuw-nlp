import argparse
import json
import os
from collections import Counter

from tuw_nlp.sem.hrg.common.conll import get_pos_tags
from tuw_nlp.sem.hrg.common.io import get_range
from tuw_nlp.sem.hrg.common.report import make_markdown_table


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-g", "--gold")
    parser.add_argument("-d", "--data-dir")
    parser.add_argument("-c", "--config")
    parser.add_argument("-rd", "--report-dir")
    return parser.parse_args()


def get_all_pos_tags(preproc_dir):
    all_pos_tags = {}
    for sen_dir in get_range(preproc_dir, None, None):
        parsed_doc_file = f"{preproc_dir}/{sen_dir}/preproc/sen{sen_dir}_parsed.conll"
        all_pos_tags[sen_dir] = get_pos_tags(parsed_doc_file)
    return all_pos_tags


def get_verb_pos_dist(extractions, pos_tags):
    cnt = Counter()
    for sen, ex_list in extractions.items():
        for ex in ex_list:
            rel_indexes = ex["rel"]["indexes"]
            sen_id = ex["sen_id"]
            for idx in rel_indexes:
                while sen_id not in pos_tags:
                    sen_id -= 1
                cnt[pos_tags[sen_id][str(idx)]] += 1
    table = [[
        "POS tag",
        "number of pred words",
    ]]
    for pos, nr in cnt.items():
        table.append([pos, nr])
    table.append(["sum", sum(cnt.values())])
    return make_markdown_table(table)


def main(gold_path, data_dir, config_json, report_dir):
    config = json.load(open(config_json))
    pos_tags = get_all_pos_tags(f"{data_dir}/{config['preproc_dir']}")

    report = "# Verb stat\n"

    gold_stat = get_verb_pos_dist(json.load(open(gold_path)), pos_tags)
    report += "## Gold\n"
    report += f"{gold_stat}\n"

    for c in config["models"]:
        report += f"## {c['name']}\n"
        if c.get("ignore") and c["ignore"]:
            continue
        for chart_filter in c["bolinas_chart_filters"]:
            for pp in c["postprocess"]:
                report += f"### {chart_filter} - {pp}\n"
                in_dir = f"{data_dir}/{config['extractions_dir']}/{c['name']}"
                if chart_filter:
                    in_dir += f"/{chart_filter}"
                if pp:
                    in_dir += f"/{pp}"
                files = [i for i in os.listdir(in_dir) if i.endswith("all.json")]
                assert len(files) == 1
                fn = f"{in_dir}/{files[0]}"
                model_stat = get_verb_pos_dist(json.load(open(fn)), pos_tags)
                report += f"{model_stat}\n"

    with open(f"{report_dir}/pred_pos_tag_stat.md", "w") as f:
        f.writelines(report)


if __name__ == "__main__":
    args = get_args()
    main(
        args.gold,
        args.data_dir,
        args.config,
        args.report_dir,
    )
