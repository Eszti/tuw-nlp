import json
import os
import time
from abc import ABC, abstractmethod
from datetime import datetime


class Script(ABC):
    def __init__(self, data_dir, config_json, log):
        self.data_dir = data_dir
        self.parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(config_json)))
        self.config_name = config_json.split('/')[-1].split('.json')[0]
        self.config = json.load(open(config_json))
        self.log = log
        if log:
            self.start_time = time.time()
            self.log_lines = [f"Execution start: {datetime.now()}\n", f"{json.dumps(self.config, indent=4)}\n"]
            self.log_file = f"{self._get_subdir('log')}/{self.config_name}.log"
        self.first_sen_to_proc = None
        self.last_sen_to_proc = None
        self.out_dir = f"{self.data_dir}/{self.config['out_dir']}" if "out_dir" in self.config else None

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
            log_lines = self.log_lines
        log_lines.append(f"{line}\n")
        print(line)

    def _get_subdir(self, name, parent_dir=None, create=True):
        if parent_dir is None:
            parent_dir = self.parent_dir
        subdir = f"{parent_dir}/{name}"
        if not os.path.exists(subdir):
            if create:
                os.makedirs(subdir)
            else:
                raise RuntimeError(f"{subdir} does not exist")
        return subdir
