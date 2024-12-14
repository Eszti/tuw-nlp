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
        self.pipeline_dir = os.path.dirname(os.path.dirname(os.path.realpath(self.config_json)))
        self.script_output_root = f"{self.pipeline_dir}/output"
        self.config_name = self.config_json.split('/')[-1].split('.json')[0]
        self.config = json.load(open(self.config_json))
        self.log = log
        if log:
            self.log_file = f"{self._get_subdir('log', parent_dir=self.pipeline_dir)}/{self.config_name}.log"
            if os.path.exists(self.log_file):
                os.remove(self.log_file)
            self.start_time = time.time()
            self._log(f"Execution start: {datetime.now()}\n{json.dumps(self.config, indent=4)}\n")
        self.first_sen_to_proc = None
        self.last_sen_to_proc = None
        self.out_dir = f"{self.data_dir}/{self.config['out_dir']}" if "out_dir" in self.config else None
        self.first = self.config.get("first", None)
        self.last = self.config.get("last", None)

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
            if self.first_sen_to_proc is not None or self.last_sen_to_proc is not None:
                self._log(
                    f"\nFirst sentence to process: {self.first_sen_to_proc}"
                    f"\nLast sentence to process: {self.last_sen_to_proc}",
                    print_to_std=True,
                )
            self._log(f"\nExecution finish: {datetime.now()}", print_to_std=True)
            elapsed_time = time.time() - self.start_time
            self._log(
                f"Elapsed time: {round(elapsed_time / 60)} min {round(elapsed_time % 60)} sec\n",
                print_to_std=True
            )

    def _log(self, line, log_lines=None, print_to_std=False):
        new_line = f"{line}\n"
        if log_lines is None:
            with open(self.log_file, "a") as f:
                f.write(new_line)
        else:
            log_lines.append(new_line)
        if print_to_std:
            print(line)

    def _get_subdir(self, name, parent_dir=None, create=True):
        if parent_dir is None:
            parent_dir = self.script_output_root
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
