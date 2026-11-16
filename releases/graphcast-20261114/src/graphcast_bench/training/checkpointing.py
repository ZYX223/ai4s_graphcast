"""Publish independently verifiable training snapshots, using only the stdlib.

Workers provide the device snapshot and exact reload checks. JSON is the commit
record: a data file without its matching record is never a resumable checkpoint.
"""
import hashlib
import json
import os
from pathlib import Path
import pickle
import tempfile
import time

SCHEMA = "graphcast.training_checkpoint.v1"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sync_directory(path):
    descriptor = os.open(str(path), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_json(path, record):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", prefix=".checkpoint-", suffix=".tmp",
                                         dir=str(path.parent), delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(record, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        sync_directory(path.parent)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


class CheckpointWriter:
    """Save on global update multiples and always at the selected stage end."""

    def __init__(self, output, mode, data_sha256, plan, tiny, rollout, save_every,
                 snapshot, verify, preflight=False):
        if save_every is not None and (type(save_every) is not int or save_every < 1):
            raise ValueError("--save-every must be a positive integer")
        self.output = Path(output)
        self.metadata = dict(mode=mode, data_sha256=data_sha256, plan=plan, tiny=tiny)
        self.rollout = rollout
        self.preflight = preflight
        self.save_every = save_every
        self.snapshot = snapshot
        self.verify = verify
        self.events = []

    def save(self, params, state, update, final=False):
        if not final and (self.save_every is None or update % self.save_every):
            return None
        started, begin_ns = time.perf_counter(), int(time.time() * 1e9)
        folder = self.output / "checkpoints"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / ("checkpoint-%04d.pkl" % update)
        commit = path.with_suffix(".json")
        if commit.exists():
            raise ValueError("Refusing to overwrite committed checkpoint: " + str(path))
        params_host, state_host = self.snapshot((params, state))
        payload = dict(self.metadata, params=params_host, opt_state=state_host, updates=update)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(prefix=".checkpoint-", suffix=".tmp",
                                             dir=str(folder), delete=False) as stream:
                temporary = Path(stream.name)
                pickle.dump(payload, stream, protocol=4)
                stream.flush()
                os.fsync(stream.fileno())
            reload = self.verify(temporary, params_host, state_host)
            if reload.get("status") != "passed":
                raise ValueError("Checkpoint reload validation failed")
            record = dict(schema=SCHEMA, status="completed", rollout=self.rollout,
                          mode=self.metadata["mode"], tiny=self.metadata["tiny"],
                          preflight=self.preflight,
                          data_sha256=self.metadata["data_sha256"],
                          plan_sha256=hashlib.sha256(json.dumps(self.metadata["plan"],
                              sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                          rows=[dict(update=update)], checkpoint_file=path.name,
                          checkpoint_reload=reload, checkpoint_sha256=file_sha256(temporary))
            temporary.replace(path)
            sync_directory(folder)
            publish_json(commit, record)
            publish_json(folder / "latest.json", dict(update=update, checkpoint=path.name))
            if final:
                # Same filesystem: preserve the historical path without copying model state.
                alias = self.output / ".checkpoint.pkl.tmp"
                try:
                    os.link(str(path), str(alias))
                    alias.replace(self.output / "checkpoint.pkl")
                    sync_directory(self.output)
                finally:
                    if alias.exists():
                        alias.unlink()
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        self.events.append(dict(update=update, final=final,
                                path=str(path.relative_to(self.output)),
                                seconds=time.perf_counter() - started,
                                begin_ns=begin_ns, end_ns=int(time.time() * 1e9)))
        return record

    def summary(self):
        return dict(save_every=self.save_every, saves=self.events,
                    total_seconds=sum(event["seconds"] for event in self.events),
                    timing_scope="Snapshot, write and reload verification; excluded from formal update timing; included in total_main_seconds")
