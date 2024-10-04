
def get_labels_str(sen, model):
    ret = []
    for line in sen:
        if model == "boa":
            ret.append(line[-1][0])
        elif model == "argidx":
            ret.append(line[-1].split("-")[0])
    return "_".join(ret)


def get_pos_tags(fn):
    with open(fn) as f:
        lines = f.readlines()
    ret = {}
    for line in lines:
        line = line.strip()
        fields = line.split('\t')
        if len(fields) > 1:
            ret[fields[0]] = fields[3]
    return ret


def get_sen_from_conll_sen(sen):
    return " ".join([line[1] for line in sen])


def get_sen_text_from_conll_file(conll):
    with open(conll) as f:
        lines = f.readlines()
    return " ".join([line.strip().split("\t")[1] for line in lines])
