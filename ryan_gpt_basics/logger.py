"""Weights & Biases logging wrapper."""

import time


class Logger:
    """Simple wandb logger with timing utilities."""
    
    def __init__(self, project: str = "gpt", name: str = None, config: dict = None):
        import wandb
        self._wandb = wandb
        self._start = time.time()
        self._run = wandb.init(project=project, name=name, config=config)
    
    def log(self, step: int, metrics: dict):
        """Log metrics at given step."""
        self._wandb.log({"step": step, "time": self.elapsed_time(), **metrics}, step=step)
    
    def elapsed_time(self) -> float:
        """Seconds since logger creation."""
        return time.time() - self._start
    
    def finish(self):
        """End the wandb run."""
        self._wandb.finish()
