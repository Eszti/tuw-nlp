# Method documentation

## Proof of concept 

First, we try to find a top estimate for our system in order to validate our concept.

### Train a grammar

 We [train](steps/train/train.py) a hyperedge replacement [grammar](pipeline/output/grammar) (HRG) using the [lsoie dataset](https://github.com/Jacobsolawetz/large-scale-oie/tree/master/dataset_creation/lsoie_data) on the triplet induced sub-graphs of the UD graph of a sentence. We create one rule per word and use the nonterminals `S`, `A`, `P` and `X` (no label). 

```bash
# Get the data
export DATA_DIR=$HOME/data
mkdir $DATA_DIR
cd $DATA_DIR
# Download and unzip the lsoie data into a folder called lsoie_data

# Preprocess the train data
python steps/preproc/preproc.py  -d $DATA_DIR -c pipeline/config/preproc_train.json

# Train the grammar
python steps/train/train.py -d $DATA_DIR -c pipeline/config/train_per_word.json
```

We [create](steps/train/hrg.py) different cuts of this grammar using the top 100, 200 and 300 rules by keeping the original distribution of nonterminals and norming the weighs per nonterminal.

```bash
python steps/train/hrg.py -d $DATA_DIR -c pipeline/config/hrg.json
```

#### Run the whole train pipeline

```bash
python pipeline/pipeline.py -d $DATA_DIR -c pipeline/config/pipeline_train.json
```

### Predict with the grammar on dev

First, we [preprocess](steps/preproc/preproc.py) the dev data as well.

```bash
python steps/preproc/preproc.py -d $DATA_DIR -c pipeline/config/preproc_dev.json
```

Using the grammar, first we [parse](steps/bolinas/parse/parse.py) the UD graphs on the dev set, saving the resulting charts as an intermediary output. We prune the parsing above 50.000 steps. The parsing takes 4-10 hours, see more [here](pipeline/log).

```bash
python steps/bolinas/parse/parse.py -d $DATA_DIR -c pipeline/config/parse_100.json
```

We [search](steps/bolinas/kbest/kbest.py) for the top k best derivations in the chart. We apply different filters on the chart: `basic` (no filtering), `max` (searching only among the largest derivations), or classic retrieval metrics `precision`, `recall` and `f1-score`, where we cheat by using the gold data and returning for each gold entry only the one derivation with the highest respective score. To calculate these scores we use the same triplet matching and scoring function as in the evaluation step. Our system returns at most k node-label maps for a sentence, where a label corresponds to a nonterminal symbol. This mapping requires some [postprocessing](steps/postproc/postproc.py), a predicate resolution in case no predicate is found and an argument grouping and indexing step, since we only have `A` as nonterminal. For `precision`, `recall` and `f1-score` filters this postprocessing step has to be done before calculating the scores. We also try argument permutation, in which case we try all possible argument indexing for the identified argument groups. This search takes from 1 our to 2.5 days, see more [here](pipeline/log).

```bash
python steps/bolinas/kbest/kbest.py -d $DATA_DIR -c pipeline/config/kbest_100.json
```

After the k best derivations are found we [predict](steps/predict/predict.py) the labels, where we apply the necessary [postprocessing](steps/postproc/postproc.py) steps (for `basic` and `max`). There is a possibility to implement further postprocessing strategies, as for now `keep` (resolving predicate only if not present, forming argument groups as continuous A-label word spans and indexing these groups from left to right) is our only strategy.

```bash
python steps/predict/predict.py -d $DATA_DIR -c pipeline/config/predict_100.json 
```

Once all sentences are predicted, we [merge](steps/predict/merge.py) them into one json per model.

```bash
python steps/predict/merge.py -d $DATA_DIR -c pipeline/config/merge_100.json
```

#### Run the whole predict pipeline on dev

```bash
# Hrg - 100
python pipeline/pipeline.py -d $DATA_DIR -c pipeline/config/pipeline_dev_100.json

# Hrg - 200
python pipeline/pipeline.py -d $DATA_DIR -c pipeline/config/pipeline_dev_200.json

# Hrg - 300
python pipeline/pipeline.py -d $DATA_DIR -c pipeline/config/pipeline_dev_300.json
```

### Create random predictions for comparison

We implement a [random extractor](steps/random/random_extractor.py) that uses the [artefacts](pipeline/output/artefacts) of the training dataset (distribution of the number of extractions per sentence, and distribution of labels per length of the sentence) and assures that the predicate is a verb.  

```bash
# Extract artefacts
python steps/random/artefacts.py -d $DATA_DIR -c pipeline/config/artefacts_train.json

# Get random extractions
python steps/random/random_extractor.py -d $DATA_DIR -c pipeline/config/random_dev.json

# Merge the extractions
python steps/predict/merge.py -d $DATA_DIR -c pipeline/config/merge_dev_random.json

# Or run as a pipeline
python pipeline/pipeline.py -d $DATA_DIR -c pipeline/config/pipeline_dev_random.json
```

### Evaluate the predictions

We [evaluate](steps/eval/eval.py) our system using a slightly modified version of the [scorer](steps/eval/wire_scorer.py) from the [WiRe paper](https://aclanthology.org/W19-4002/) (since lsoie triples does not necessarily have a second argument, common words are only needed for predicates and first arguments in order for two triplets to match). We present the results of [all](eval/reports/dev_all.md) our systems and a filtered table for the [top estimation](eval/reports/dev_best.md).

```bash
python steps/eval/eval.py -d $DATA_DIR -c pipeline/config/eval_dev_all.json
```

We calculate some [statistics](steps/stat/run_all_stat.py) (distribution of extractions per sentence, predicate recognition, rule usage) for quantitative and qualitative analysis. See output [here](pipeline/output/stat).

```bash
python steps/stat/run_all_stat.py -d $DATA_DIR -c pipeline/config/stat_dev.json
```

### Compare the results with baselines on the test set

We compare our [results](test/reports/eval.md) on the test set with [baseline systems](https://github.com/Jacobsolawetz/large-scale-oie/tree/master/large_scale_oie/evaluation) made available in the repository for the [lsoie paper](https://aclanthology.org/2021.eacl-main.222/).

```python
# TBD
```
