"""Baseline fallback; authors replace Runner with a complete implementation."""


class Runner:
    def __init__(self, task, config, parameters, reference):
        self.reference = reference

    def run(self, *inputs):
        return self.reference(*inputs)
