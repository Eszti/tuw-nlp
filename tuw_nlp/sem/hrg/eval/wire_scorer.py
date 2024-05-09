import argparse
import json
from collections import defaultdict


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-g", "--gold")
    parser.add_argument("-p", "--predicted")
    return parser.parse_args()


def main(gold_path, pred_path):
    gold = json.load(open(gold_path))
    all_predictions = json.load(open(pred_path))
    # check_all_predictions(all_predictions, gold.keys())
    common = check_keys(gold.keys(), all_predictions.keys())
    keep_only_common(gold, common)
    keep_only_common(all_predictions, common)
    
    predictions_by_model = split_tuples_by_extractor(gold.keys(), all_predictions)
    models = predictions_by_model.keys()

    reports = {}
    for m in models:
        report = ""
        metrics, raw_match_scores = eval_system(gold, predictions_by_model[m])
        with open("raw_scores/" + m + "_prec_scores.dat", "w") as f:
            f.write(str(raw_match_scores[0]))
        with open("raw_scores/" + m + "_rec_scores.dat", "w") as f:
            f.write(str(raw_match_scores[1]))
        prec, rec = metrics['precision'], metrics['recall']
        f1_score = f1(prec, rec)
        exactmatch_prec = metrics['exactmatches_precision'][0] / metrics['exactmatches_precision'][1]
        exactmatch_rec = metrics['exactmatches_recall'][0] / metrics['exactmatches_recall'][1]
        report += ("System {} prec/rec/f1: {:.1%} {:.1%} {:.3f}"
                   .format(m, prec, rec, f1_score))
        report += ("\nSystem {} prec/rec of matches only (non-matches): {:.0%} {:.0%} ({})"
                   .format(m, metrics['precision_of_matches'], metrics['recall_of_matches'], metrics['matches']))
        report += ("\n{} were exactly correct, out of {} predicted / the reference {}."
                   .format(metrics['exactmatches_precision'][0],
                           metrics['exactmatches_precision'][1], metrics['exactmatches_recall'][1]))
        report += ("\nExact-match prec/rec/f1: {:.1%} {:.1%} {:.3f}"
                   .format(exactmatch_prec, exactmatch_rec, f1(exactmatch_prec, exactmatch_rec)))
        reports[f1_score] = report
    sorted_reports = [a[1] for a in sorted(reports.items(), reverse=True)]
    print("\n" + "\n\n".join(sorted_reports))


def eval_system(gold, predictions):
    results = {}
    for s, reference_tuples in gold.items():
        predicted_tuples = predictions.get(s, [])
        results[s] = sentence_match(reference_tuples, predicted_tuples)

    prec_num, prec_denom = 0, 0
    rec_num, rec_denom = 0, 0
    exactmatches_precnum, exactmatches_precdenom = 0, 0
    exactmatches_recnum, exactmatches_recdenom = 0, 0
    tot_prec_of_matches, tot_rec_of_matches = 0, 0
    for s in results.values():
        prec_num += s['precision'][0]
        prec_denom += s['precision'][1]
        rec_num += s['recall'][0]
        rec_denom += s['recall'][1]
        exactmatches_precnum += s['exact_match_precision'][0]
        exactmatches_precdenom += s['exact_match_precision'][1]
        exactmatches_recnum += s['exact_match_recall'][0]
        exactmatches_recdenom += s['exact_match_recall'][1]
        tot_prec_of_matches += sum(s['precision_of_matches'])
        tot_rec_of_matches += sum(s['recall_of_matches'])
    precision_scores = [v for s in results.values() for v in s['precision_of_matches']]
    recall_scores = [v for s in results.values() for v in s['recall_of_matches']]
    raw_match_scores = [precision_scores, recall_scores]
    matches = len(precision_scores)
    metrics = {
        'precision': prec_num / prec_denom,
        'recall': rec_num / rec_denom,
        'matches': matches,
        'precision_of_matches': tot_prec_of_matches / matches,
        'recall_of_matches': tot_rec_of_matches / matches,
        'exactmatches_precision': [exactmatches_precnum, exactmatches_precdenom],
        'exactmatches_recall': [exactmatches_recnum, exactmatches_recdenom]
    }
    return metrics, raw_match_scores


def f1(prec, rec):
    try:
        return 2 * prec * rec / (prec + rec)
    except ZeroDivisionError:
        return 0


def sentence_match(gold, predicted):
    exact_match_scores = [[None for _ in predicted] for __ in gold]
    scores = [[None for _ in predicted] for __ in gold]
    for i, gt in enumerate(gold):
        for j, pt in enumerate(predicted):
            exact_match_scores[i][j] = tuple_exact_match(pt, gt)
            scores[i][j] = tuple_match(pt, gt)
    scoring_metrics = aggregate_scores_greedily(scores)
    exact_match_summary = aggregate_exact_matches(exact_match_scores)
    scoring_metrics['exact_match_precision'] = exact_match_summary['precision']
    scoring_metrics['exact_match_recall'] = exact_match_summary['recall']
    return scoring_metrics


def aggregate_scores_greedily(scores):
    matches = []
    while True:
        max_s = 0
        gold, pred = None, None
        for i, gold_ss in enumerate(scores):
            if i in [m[0] for m in matches]:
                continue
            for j, pred_s in enumerate(scores[i]):
                if j in [m[1] for m in matches]:
                    continue
                if pred_s and f1(*pred_s) > max_s:
                    max_s = f1(*pred_s)
                    gold = i
                    pred = j
        if max_s == 0:
            break
        matches.append([gold, pred])
    prec_scores = [scores[i][j][0] for i, j in matches]
    rec_scores = [scores[i][j][1] for i, j in matches]
    total_prec = sum(prec_scores)
    total_rec = sum(rec_scores)
    scoring_metrics = {"precision": [total_prec, len(scores[0])],
                       "recall": [total_rec, len(scores)],
                       "precision_of_matches": prec_scores,
                       "recall_of_matches": rec_scores
                       }
    return scoring_metrics


def aggregate_exact_matches(match_matrix):
    recall = [sum([any(gold_matches) for gold_matches in match_matrix], 0), len(match_matrix)]
    if len(match_matrix[0]) == 0:
        precision = [0, 0]
    else:
        precision = [sum([any([g[i] for g in match_matrix]) for i in range(len(match_matrix[0]))], 0),
                     len(match_matrix[0])]
    metrics = {'precision': precision,
               'recall': recall}
    return metrics


def tuple_exact_match(t, gt):
    for part in ['arg1', 'rel']:
        if not t[part] == gt[part]:
            return False
    if gt['arg2+']:
        if not t.get('arg2+', False):
            return False
        for i, p in enumerate(gt['arg2+']):
            if len(t['arg2+']) > i and t['arg2+'][i] != p:
                return False
    return True


def tuple_match(t, gt):
    precision = [0, 0]
    recall = [0, 0]
    for part in ['arg1', 'rel']:
        predicted_words = t[part].split()
        gold_words = gt[part].split()
        if not predicted_words:
            if gold_words:
                return False
            else:
                continue
        matching_words = sum(1 for w in predicted_words if w in gold_words)
        if matching_words == 0:
            return False
        precision[0] += matching_words
        precision[1] += len(predicted_words)
        recall[0] += matching_words
        recall[1] += len(gold_words)
    if gt['arg2+']:
        for i, gold_words in enumerate(gt['arg2+']):
            recall[1] += len(gold_words)
            if t.get("arg2+", False) and len(t['arg2+']) > i:
                predicted_words = t['arg2+'][i].split()
                matching_words = sum(1 for w in predicted_words if w in gold_words)
                precision[0] += matching_words
                precision[1] += len(predicted_words)
                recall[0] += matching_words
    prec = precision[0] / precision[1]
    rec = recall[0] / recall[1]
    return [prec, rec]


def split_tuples_by_extractor(gold, tuples):
    systems = sorted(list(set(t['extractor'] for st in tuples.values() for t in st)))
    predictions_by_model = {e: {} for e in systems}
    for s in gold:
        for t in tuples[s]:
            if s in predictions_by_model[t['extractor']]:
                predictions_by_model[t['extractor']][s].append(t)
            else:
                predictions_by_model[t['extractor']][s] = [t]
    return predictions_by_model


def check_all_predictions(extracted, gold_keys):
    extracted_by_model = defaultdict(set)
    for k, v in extracted.items():
        for t in v:
            extracted_by_model[t["extractor"]].add(k)
    common = None
    for model, keys in extracted_by_model.items():
        if common is None:
            common = set(keys)
        else:
            common &= keys
    diff = set()
    for model, keys in extracted_by_model.items():
        print(model)
        print(len(keys))
        print('\n'.join(keys - common))
        print("\n")
        diff |= (keys - common)
    print("diff")
    print('\n'.join(diff))
    print("\ndiff - gold")
    print('\n'.join(diff - gold_keys))
    exit()
    pass


def check_keys(gold, extracted):
    print(f"Keys in gold: {len(gold)}")
    print(f"Keys in extracted: {len(extracted)}")
    found = 0
    not_found = 0
    common = set()
    for s in gold:
        if s not in extracted:
            not_found += 1
        else:
            found += 1
            common.add(s)
    print("Keys from gold")
    print(f"found: {found}")
    print(f"not found: {not_found}")
    found = 0
    not_found = 0
    for s in extracted:
        if s not in gold:
            not_found += 1
        else:
            found += 1
    print("Keys from extracted")
    print(f"found: {found}")
    print(f"not found: {not_found}")
    return common


def keep_only_common(tuples, common):
    diff = tuples.keys() - common
    for k in diff:
        del tuples[k]
        

if __name__ == "__main__":
    args = get_args()
    main(args.gold, args.predicted)
