import json
import time
from abc import abstractmethod
from datetime import datetime

from tuw_nlp.sem.hrg.common.io import log_to_console_and_log_lines
from tuw_nlp.sem.hrg.common.script.script import Script


class BolinasScript(Script):
    def __init__(self, data_dir, config_json, log_file_prefix):
        super().__init__(data_dir, config_json)
        self.start_time = None
        self.log_lines = None
        self.log_file = f"{log_file_prefix}{self.config['model_dir']}.log"
        self.last_sen_to_proc = None
        self.first_sen_to_proc = None

    @abstractmethod
    def run_loop(self):
        pass

    def before_loop(self):
        self.start_time = time.time()
        self.log_lines = [f"Execution start: {datetime.now()}\n", f"{json.dumps(self.config, indent=4)}\n"]

    def after_loop(self):
        log_to_console_and_log_lines(f"\nFirst sentence to process: {self.first_sen_to_proc}", self.log_lines)
        log_to_console_and_log_lines(f"Last sentence to process: {self.last_sen_to_proc}", self.log_lines)

        log_to_console_and_log_lines(f"\nExecution finish: {datetime.now()}", self.log_lines)
        elapsed_time = time.time() - self.start_time
        time_str = f"Elapsed time: {round(elapsed_time / 60)} min {round(elapsed_time % 60)} sec\n"
        log_to_console_and_log_lines(time_str, self.log_lines)
        with open(self.log_file, "w") as f:
            f.writelines(self.log_lines)
