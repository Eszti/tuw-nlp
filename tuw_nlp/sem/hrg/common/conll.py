
def get_labels_str(sen):
    ret = ""
    for line in sen:
        ret += line[-1][0]
    return ret


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


def get_sen_from_conll_file(conll):
    with open(conll) as f:
        lines = f.readlines()
    return " ".join([line.strip().split("\t")[1] for line in lines])
