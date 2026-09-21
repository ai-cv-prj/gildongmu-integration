"""Non-overwriting JSONL sidecar, published with the completed video."""
import json
import os
from pathlib import Path
import tempfile

class RiskLog:
    def __init__(self, video_output):
        self.path = Path(video_output).with_suffix(".risk.jsonl")
        if self.path.exists():
            raise FileExistsError(f"Risk log already exists: {self.path}")
        self.file = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path.parent,
                     prefix=f".{self.path.stem}.", suffix=".partial.jsonl", delete=False)
        self.temporary = Path(self.file.name)
        self.published = False

    def write(self, prediction):
        self.file.write(json.dumps(prediction, ensure_ascii=False, allow_nan=False)+"\n")

    def publish(self):
        self.file.close()
        os.link(self.temporary, self.path)
        self.published = True

    def finish(self, committed):
        self.file.close()
        # Only remove a link created by this operation, never a pre-existing/replaced result.
        if self.published and not committed and self.path.exists() and os.path.samefile(self.path,self.temporary):
            self.path.unlink()
        self.temporary.unlink(missing_ok=True)
