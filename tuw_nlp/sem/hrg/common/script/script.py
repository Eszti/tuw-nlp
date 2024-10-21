import json
from abc import ABC, abstractmethod


class Script(ABC):
    def __init__(self, data_dir, config_json):
        self.data_dir = data_dir
        self.config = json.load(open(config_json))
        self.first = self.config.get("first", None)
        self.last = self.config.get("last", None)

    def run(self):
        self.before_loop()
        self.run_loop()
        self.after_loop()

    @abstractmethod
    def before_loop(self):
        pass

    @abstractmethod
    def run_loop(self):
        pass

    @abstractmethod
    def after_loop(self):
        pass
