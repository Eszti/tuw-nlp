from tuw_nlp.sem.hrg.common.script.script import Script
from tuw_nlp.sem.hrg.stat.k_stat import KStat
from tuw_nlp.sem.hrg.stat.pred_eval import PredEval
from tuw_nlp.sem.hrg.stat.rule_stat import RuleStat


class RunAllStat(Script):
    def __init__(self):
        super().__init__(description="Script to run all dev statistics at once.", log=False)
        self.config_dir = self._get_subdir("config", create=False)

    def _run_loop(self):
        print("Calculate k stat")
        KStat(f"{self.config_dir}/{self.config['k_stat_config']}").run()
        print("Evaluate predicate resolution")
        PredEval(f"{self.config_dir}/{self.config['pred_eval_config']}").run()
        print("Calculate rule stat")
        RuleStat(f"{self.config_dir}/{self.config['rule_stat_config']}").run()


if __name__ == "__main__":
    RunAllStat().run()
