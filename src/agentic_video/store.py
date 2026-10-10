"""Atomic JSON metadata store. Intended for one API process / one host."""
import json
import os
from pathlib import Path
from uuid import UUID
from .models import Project, ProjectStatus, ShotStatus


class FileProjectStore:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def directory(self, project_id: str) -> Path:
        try:
            if str(UUID(project_id)) != project_id:
                raise ValueError("noncanonical project id")
        except (ValueError, AttributeError) as exc:
            raise KeyError("project not found") from exc
        return self.root / project_id

    def load(self, project_id: str) -> Project:
        path = self.directory(project_id) / "project.json"
        try:
            return Project.model_validate_json(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise KeyError("project not found") from exc

    def save(self, project: Project) -> None:
        directory = self.directory(project.project_id)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "project.json"
        tmp = directory / f".project.{os.getpid()}.tmp"
        tmp.write_text(project.model_dump_json(indent=2), encoding="utf-8")
        os.replace(tmp, path)

    def recover_interrupted(self) -> None:
        """No task survives a process restart; mark interrupted work retryable."""
        for path in self.root.glob("*/project.json"):
            try:
                project = self.load(path.parent.name)
            except (KeyError, ValueError, OSError, json.JSONDecodeError):
                continue
            if project.status in {ProjectStatus.GENERATING, ProjectStatus.STITCHING}:
                project.status = ProjectStatus.FAILED
                project.error = "Process restarted during generation; retry the operation"
                for shot in project.shots:
                    if shot.status.value == "GENERATING":
                        shot.status = ShotStatus.FAILED
                        shot.error = "Interrupted; retry generation"
                self.save(project)

    def asset(self, project_id: str, filename: str) -> Path:
        """Only files created by this application may be read as project assets."""
        if not filename or Path(filename).name != filename or filename.startswith("."):
            raise KeyError("asset not found")
        path = self.directory(project_id) / filename
        if path.suffix != ".mp4" or not path.is_file():
            raise KeyError("asset not found")
        return path
