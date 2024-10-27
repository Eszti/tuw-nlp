import json
import os
import time
from abc import ABC, abstractmethod
from datetime import datetime


class Script(ABC):
    def __init__(self, data_dir, config_json, log=False):
        self.data_dir = data_dir
        self.config = json.load(open(config_json))
        self.log = log
        if log:
            self.start_time = time.time()
            self.log_lines = [f"Execution start: {datetime.now()}\n", f"{json.dumps(self.config, indent=4)}\n"]

            log_dir = f"{os.path.dirname(os.path.dirname(os.path.realpath(config_json)))}/log"
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            self.log_file = f"{log_dir}/{config_json.split('/')[-1].split('.json')[0]}.log"
        self.first_sen_to_proc = None
        self.last_sen_to_proc = None

    def run(self):
        self._before_loop()
        self._run_loop()
        self._after_loop()

    def _before_loop(self):
        pass

    @abstractmethod
    def _run_loop(self):
        raise NotImplemented

    def _after_loop(self):
        if self.log:
            self._log(
                f"\nFirst sentence to process: {self.first_sen_to_proc}"
                f"\nLast sentence to process: {self.last_sen_to_proc}"
            )
            self._log(f"\nExecution finish: {datetime.now()}")
            elapsed_time = time.time() - self.start_time
            self._log(f"Elapsed time: {round(elapsed_time / 60)} min {round(elapsed_time % 60)} sec\n")
            with open(self.log_file, "w") as f:
                f.writelines(self.log_lines)

    def _log(self, line, log_lines=None):
        if not log_lines:
            self.log_lines.append(f"{line}\n")
        else:
            log_lines.append(f"{line}\n")
        print(line)
