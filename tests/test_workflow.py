import asyncio
from pathlib import Path

import pytest

from agentic_video.models import (
    CreateProjectRequest, ProjectStatus, ShotStatus, UpdateShotRequest,
)
from agentic_video.parsers import RuleStoryboardParser
from agentic_video.service import VideoService, WorkflowConflict
from agentic_video.store import FileProjectStore


class FakeBackend:
    def __init__(self):
        self.calls = []
        self.fail = False

    async def generate(self, prompt, destination, duration):
        self.calls.append(prompt)
        if self.fail:
            raise RuntimeError("simulated GPU failure")
        destination.write_bytes(f"video:{prompt}:{duration}".encode())


class FakeComposer:
    def __init__(self):
        self.calls = 0

    def compose(self, clips, output):
        self.calls += 1
        output.write_bytes(b"|".join(p.read_bytes() for p, _ in clips))


async def drain(service):
    while service._tasks:
        await asyncio.gather(*list(service._tasks))


def test_end_to_end_and_single_shot_regeneration(tmp_path):
    async def scenario():
        backend = FakeBackend()
        composer = FakeComposer()
        service = VideoService(
            FileProjectStore(tmp_path), RuleStoryboardParser(),
            backend, composer, max_shots=5
        )
        p = await service.create(CreateProjectRequest(
            script="Morning coffee. A customer smiles. Steam rises.",
            shot_count=3
        ))
        assert p.status == ProjectStatus.DRAFT
        assert len(p.shots) == 3
        await service.enqueue(p.project_id)
        await drain(service)
        p = service.get(p.project_id)
        assert p.status == ProjectStatus.READY
        assert len(backend.calls) == 3
        first_clip = p.shots[0].video_path
        second_clip = p.shots[1].video_path
        previous_final = p.final_video_path
        await service.edit(
            p.project_id, 2, UpdateShotRequest(prompt="Close-up of coffee foam")
        )
        assert service.get(p.project_id).status == ProjectStatus.DRAFT
        await service.enqueue(p.project_id)
        await drain(service)
        p = service.get(p.project_id)
        assert p.status == ProjectStatus.READY
        assert len(backend.calls) == 4
        assert composer.calls == 2
        assert p.shots[0].video_path == first_clip
        assert p.shots[1].video_path != second_clip
        assert p.final_video_path != previous_final
        assert not (tmp_path / p.project_id / previous_final).exists()
        assert not (tmp_path / p.project_id / second_clip).exists()

        backend.fail = True
        await service.enqueue(p.project_id, shot_id=2)
        await drain(service)
        failed = service.get(p.project_id)
        assert failed.status == ProjectStatus.FAILED
        assert failed.shots[1].status == ShotStatus.FAILED
        assert failed.shots[0].video_path == first_clip
        backend.fail = False
        await service.enqueue(p.project_id)
        await drain(service)
        assert service.get(p.project_id).status == ProjectStatus.READY
        await service.close()
    asyncio.run(scenario())


def test_edit_during_generation_fails(tmp_path):
    async def scenario():
        class WaitingBackend(FakeBackend):
            async def generate(self, prompt, destination, duration):
                await asyncio.sleep(0.3)
                await super().generate(prompt, destination, duration)
        service = VideoService(
            FileProjectStore(tmp_path), RuleStoryboardParser(),
            WaitingBackend(), FakeComposer()
        )
        p = await service.create(CreateProjectRequest(script="Scene", shot_count=1))
        await service.enqueue(p.project_id)
        with pytest.raises(WorkflowConflict):
            await service.edit(
                p.project_id, 1, UpdateShotRequest(prompt="new scene")
            )
        await drain(service)
        await service.close()
    asyncio.run(scenario())


def test_store_recovery_and_path_traversal(tmp_path):
    store = FileProjectStore(tmp_path)
    with pytest.raises(KeyError):
        store.load("../outside")
    with pytest.raises(KeyError):
        store.asset("../outside", "file.mp4")

    async def scenario():
        service = VideoService(store, RuleStoryboardParser(), FakeBackend(), FakeComposer())
        p = await service.create(CreateProjectRequest(script="Scene"))
        p.status = ProjectStatus.GENERATING
        p.shots[0].status = ShotStatus.GENERATING
        store.save(p)
        store.recover_interrupted()
        recovered = store.load(p.project_id)
        assert recovered.status == ProjectStatus.FAILED
        assert recovered.shots[0].status == ShotStatus.FAILED
    asyncio.run(scenario())


def test_targeted_retry_does_not_render_other_dirty_shots(tmp_path):
    async def scenario():
        backend = FakeBackend()
        service = VideoService(
            FileProjectStore(tmp_path), RuleStoryboardParser(),
            backend, FakeComposer(), max_shots=5
        )
        project = await service.create(CreateProjectRequest(
            script="Scene one. Scene two.", shot_count=2
        ))
        await service.enqueue(project.project_id)
        await drain(service)
        original = service.get(project.project_id)
        assert original.status == ProjectStatus.READY
        await service.edit(
            project.project_id, 1, UpdateShotRequest(prompt="new first shot")
        )
        await service.edit(
            project.project_id, 2, UpdateShotRequest(prompt="new second shot")
        )
        dirty = service.get(project.project_id)
        target_revision = dirty.shots[0].revision
        await service.enqueue(project.project_id, shot_id=1)
        await drain(service)
        after = service.get(project.project_id)
        assert after.shots[0].revision == target_revision  # no double increment
        assert after.shots[0].status == ShotStatus.READY
        assert after.shots[1].status == ShotStatus.PENDING
        assert len(backend.calls) == 3  # 2 original + only the specified target
        assert after.status == ProjectStatus.DRAFT
        await service.enqueue(project.project_id)
        await drain(service)
        assert service.get(project.project_id).status == ProjectStatus.READY
        assert len(backend.calls) == 4
        await service.close()
    asyncio.run(scenario())
