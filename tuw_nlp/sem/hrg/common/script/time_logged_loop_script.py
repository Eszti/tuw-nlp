import json
import os.path
import time
from abc import ABC
from datetime import datetime

from tuw_nlp.sem.hrg.common.io import log_to_console_and_log_lines
from tuw_nlp.sem.hrg.common.script.loop_script import LoopScriptOnPreprocessed


class TimeLoggedLoopScript(LoopScriptOnPreprocessed, ABC):
    def __init__(self, data_dir, config_json):
        super().__init__(data_dir, config_json)

        self.start_time = time.time()
        self.log_lines = [f"Execution start: {datetime.now()}\n", f"{json.dumps(self.config, indent=4)}\n"]

        log_dir = f"{os.path.dirname(os.path.dirname(os.path.realpath(config_json)))}/log"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.log_file = f"{log_dir}/{config_json.split('/')[-1].split('.json')[0]}.log"

    def after_loop(self):
        log_to_console_and_log_lines(
            f"\nFirst sentence to process: {self.first_sen_to_proc}"
            f"\nLast sentence to process: {self.last_sen_to_proc}",
            self.log_lines
        )
        log_to_console_and_log_lines(f"\nExecution finish: {datetime.now()}", self.log_lines)
        elapsed_time = time.time() - self.start_time
        log_to_console_and_log_lines(
            f"Elapsed time: {round(elapsed_time / 60)} min {round(elapsed_time % 60)} sec\n",
            self.log_lines
        )
        with open(self.log_file, "w") as f:
            f.writelines(self.log_lines)
