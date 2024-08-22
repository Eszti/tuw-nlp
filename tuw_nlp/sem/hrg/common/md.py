from cmath import isclose

from numpy import dtype


def make_markdown_table(array, bold=None):
    if bold is None:
        bold = list()
    markdown = "\n" + str("| ")

    for e in array[0]:
        to_add = " " + str(e) + str(" |")
        markdown += to_add
    markdown += "\n"

    markdown += "|"
    for i in range(len(array[0])):
        markdown += str("-------------- | ")
    markdown += "\n"

    for i, entry in enumerate(array[1:]):
        markdown += str("| ")
        for j, e in enumerate(entry):
            if type(e) == int or type(e) == tuple:
                e = str(e)
            elif type(e) == float or type(e) == dtype("float64"):
                e = f"{e:.4f}"
            if (i, j) in bold:
                e = f"**{e}**"
            to_add = e + str(" | ")
            markdown += to_add
        markdown += "\n"

    return markdown + "\n"


def find_best_in_column(table, headers):
    ret = []
    for header in headers:
        j = table[0].index(header)
        best = 0.0
        best_i = []
        for i, row in enumerate(table[1:]):
            f1_score = row[j]
            if isclose(best, f1_score, rel_tol=1e-04):
                best_i.append(i)
            elif f1_score > best:
                best = f1_score
                best_i = [i]
        ret += [(i, j) for i in best_i]
    return ret

