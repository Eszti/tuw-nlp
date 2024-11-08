from tuw_nlp.sem.hrg.common.script.pipeline import Pipeline


class RunPipeline(Pipeline):
    def __init__(self):
        super().__init__(description="Script to run a pipeline.", log=True)

    def _run_loop(self):
        for step in self.steps:
            step_name = step['name']
            self._log(f"Processing step {step_name}")
            step_class = self.name_to_class[step_name]
            config = f"{self.parent_dir}/steps/{step_name}/config/{step['config']}"
            step_class(config=config).run()


if __name__ == "__main__":
    RunPipeline().run()
