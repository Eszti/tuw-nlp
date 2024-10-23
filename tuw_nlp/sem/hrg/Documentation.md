# Method documentation

## Proof of concept 

First, we try to find a top estimate for our system in order to validate our concept.

Our system on the 23rd of Oct. 2024:

1. We [train](train/train.py) a hyperedge replacement [grammar](train/grammar) (HRG) using the [lsoie dataset](https://github.com/Jacobsolawetz/large-scale-oie/tree/master/dataset_creation/lsoie_data) on the triplet induced sub-graphs of the UD graph of a sentence. We create one rule per word and use the nonterminals `S`, `A`, `P` and `X` (no label). 

```python
# TBD
```

2. We create different cuts of this grammar (gr100, gr200 and gr300) using the top 100, 200 and 300 rules by keeping the original distribution of nonterminals and norming the weighs per nonterminal.

```python
# TBD
```

3. Using this grammar, first we [parse](bolinas/parse/parse.py) the UD graphs on the dev set, saving the resulting charts as an intermediary output. We prune the parsing above 10.000 steps for gr100 and gr200 and above 50.000 steps for gr300. The parsing takes from 1 hour to one day (gr300), see more [here](bolinas/parse/log).

```python
# TBD
```

4. We [search](bolinas/kbest/kbest.py) for the top k best derivations in the chart. We apply different filters on the chart: `basic` (no filtering), `max` (searching only among the largest derivations), or classic retrieval metrics `precision`, `recall` and `f1-score`, where we cheat by using the gold data and returning for each gold entry only the one derivation with the highest respective score. To calculate these scores we use the same triplet matching and scoring function as in the evaluation step. Our system returns at most k node-label maps for a sentence, where a label corresponds to a nonterminal symbol. This mapping requires some [postprocessing](postproc/postproc.py), a predicate resolution in case no predicate is found and an argument grouping and indexing step, since we only have `A` as nonterminal. For `precision`, `recall` and `f1-score` filters this postprocessing step has to be done before calculating the scores. We also try argument permutation, in which case we try all possible argument indexing for the identified argument groups. This search takes from 1 our to 2.5 days, see more [here](bolinas/kbest/log).

```python
# TBD
```

5. After the k best derivations are found we [predict](predict/predict.py) the labels, where we apply the necessary [postprocessing](postproc/postproc.py) steps (for `basic` and `max`). There is a possibility to implement further postprocessing strategies, as for now `keep` (resolving predicate only if not present, forming argument groups as continuous A-label word spans and indexing these groups from left to right) is our only strategy.

```python
# TBD
```

6. Once all sentences are predicted, we [merge](predict/merge.py) them into one json per model.

```python
# TBD
```

7. We implement a [random extractor](random/random_extractor.py) that uses the [artefacts](random/train_stat) of the training dataset (distribution of the number of extractions per sentence, and distribution of labels per length of the sentence) and assures that the predicate is a verb.  

```python
# TBD
```

8. We [evaluate](eval/eval.py) our system using a slightly modified version of the [scorer](eval/wire_scorer.py) from the [WiRe paper](https://aclanthology.org/W19-4002/) (since lsoie triples does not necessarily have a second argument, common words are only needed for predicates and first arguments in order for two triplets to match). We present the results of [all](eval/reports/dev_all.md) our systems and a filtered table for the [top estimation](eval/reports/dev_best.md).

```python
# TBD
```

9. We compare our [results](test/reports/eval.md) on the test set with [baseline systems](https://github.com/Jacobsolawetz/large-scale-oie/tree/master/large_scale_oie/evaluation) made available in the repository for the [lsoie paper](https://aclanthology.org/2021.eacl-main.222/).

```python
# TBD
```

10. We calculate some [statistics](dev_stat/run_all_stat.py) on the dev set for quantitative and qualitative analysis.

```python
# TBD
```