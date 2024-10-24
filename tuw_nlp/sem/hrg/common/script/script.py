import json
from abc import ABC, abstractmethod


class Script(ABC):
    def __init__(self, data_dir, config_json):
        self.data_dir = data_dir
        self.config = json.load(open(config_json))

    @abstractmethod
    def run(self):
        raise NotImplemented
