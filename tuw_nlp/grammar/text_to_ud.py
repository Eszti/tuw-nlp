from typing import Any, Dict

from tuw_nlp.graph.ud_graph import UDGraph
from tuw_nlp.text.pipeline import CachedStanzaPipeline, CustomStanzaPipeline


class TextToUD:
    def __init__(self, lang, nlp_cache, cache_dir=None, pretokenized=False):
        if lang == "de":
            nlp = CustomStanzaPipeline(
                processors="tokenize,mwt,pos,lemma,depparse", pretokenized=pretokenized
            )
        elif lang == "en":
            nlp = CustomStanzaPipeline(
                "en",
                processors="tokenize,mwt,pos,lemma,depparse",
                pretokenized=pretokenized,
            )
        elif lang == "en_bio":
            nlp = CustomStanzaPipeline("en", package="craft", pretokenized=pretokenized)
        assert lang, "TextTo4lang does not have lang set"

        self.lang = lang
        self.pretokenized = pretokenized
        self.nlp = CachedStanzaPipeline(nlp, nlp_cache)

    def __call__(self, text, ssplit=True):
        for sen in self.nlp(text, ssplit=ssplit).sentences:
            tokens = [token.text for token in sen.tokens]
            ud_graph = UDGraph(sen, sen.text, tokens)

            yield ud_graph

    def get_params(self) -> Dict[str, Any]:
        return {"lang": self.lang, "pretokenized": self.pretokenized}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.nlp.save_cache_if_changed()


if __name__ == "__main__":
    # for testing

    toks = ("People", "who", "'ve", "been", "giving", "them", "money")
    with TextToUD(lang="en", nlp_cache="test_cache", pretokenized=True) as t_t_ud:
        for graph in t_t_ud(toks):
            print(graph)
