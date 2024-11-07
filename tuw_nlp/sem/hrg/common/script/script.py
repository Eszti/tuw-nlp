import argparse
import json
import os
import time
from abc import ABC, abstractmethod
from datetime import datetime


class Script(ABC):
    def __init__(self, description, log, config=None):
        args = self._get_data_dir_and_config_args(description)
        if config is None:
            self.config_json = args.config
        else:
            self.config_json = config
        self.data_dir = args.data_dir
        self.parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(self.config_json)))
        self.config_name = self.config_json.split('/')[-1].split('.json')[0]
        self.config = json.load(open(self.config_json))
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

    @staticmethod
    def _add_filter_and_postprocess(name, chart_filter, postprocess, delim="/"):
        if chart_filter:
            name += f"{delim}{chart_filter}"
        if postprocess:
            name += f"{delim}{postprocess}"
        return name

    @staticmethod
    def _get_data_dir_and_config_args(desc=""):
        parser = argparse.ArgumentParser(description=desc)
        parser.add_argument("-d", "--data-dir", type=str)
        parser.add_argument("-c", "--config", type=str, default="")
        return parser.parse_args()
