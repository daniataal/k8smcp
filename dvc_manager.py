import subprocess
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

class DVCManager:
    """Simple wrapper for DVC CLI commands."""

    def __init__(self, repo_dir: Optional[str] = None):
        self.repo_dir = repo_dir

    def _run(self, args: List[str]) -> dict:
        cmd = ["dvc"] + args
        try:
            result = subprocess.run(
                cmd,
                cwd=self.repo_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            return {"status": "success", "stdout": result.stdout}
        except subprocess.CalledProcessError as e:
            logger.error(f"DVC command failed: {e.stderr}")
            return {"status": "error", "stderr": e.stderr}

    def init(self):
        return self._run(["init"])

    def add(self, path: str):
        return self._run(["add", path])

    def push(self):
        return self._run(["push"])

    def pull(self):
        return self._run(["pull"])

    def status(self):
        return self._run(["status"])

    def repro(self):
        return self._run(["repro"])

    def run(self, dvc_args: List[str]):
        return self._run(dvc_args)
