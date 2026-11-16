"""Manage container identity, mounts, and lifecycle; runtime dispatches model execution."""

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from .catalog import checked_path, sha256


def work_directory(bundle, work):
    # Each recipient owns a separate work tree for locks, caches and checkpoints.
    # Keep it outside the shared bundle, which is mounted read-only in Docker.
    bundle, work = Path(bundle).resolve(), Path(work).expanduser().resolve()
    if bundle == work or bundle in work.parents or work in bundle.parents:
        raise ValueError("Work directory must be separate from the delivery materials")
    work.mkdir(parents=True, exist_ok=True, mode=0o700)
    if work.stat().st_uid != os.getuid() or not os.access(str(work), os.W_OK):
        raise ValueError("Work directory must be owned and writable by the current user")
    return work


def container_spec(bundle, work, relative, manifest):
    """Derive container identity and mounts from the user, paths, and environment manifest."""
    identity = dict(
        bundle=str(Path(bundle).resolve()),
        work=str(work),
        environment=relative,
        manifest=sha256(Path(bundle) / relative / "manifest.json"),
        uid=os.getuid(),
        gid=os.getgid(),
        launch_policy="graphcast.container.v2",
    )
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    name = "graphcast-{}-{}".format(os.getuid(), digest[:12])
    # Each tuple is (host path, container path, read-only); separate materials from outputs.
    mounts = [(str(Path(bundle).resolve()), "/bundle", True), (str(work), "/work", False)]
    if relative.startswith("environments/bw1100/"):
        sockets = Path(work) / ".containers" / name / "sockets"
        sockets.mkdir(parents=True, exist_ok=True, mode=0o700)
        mounts.append((str(sockets), "/tmp/flagtree-r10-0", False))
    for mount in manifest["mounts"]:
        source = checked_path(bundle, str(Path(relative) / mount["source"]))
        if not source.exists():
            raise ValueError("Missing environment mount: " + str(source))
        mounts.append((str(source), mount["target"], True))
    return dict(name=name, identity=digest, mounts=mounts)


def ensure_container(bundle, work, relative, manifest):
    """Resolve the pinned image, validate or create the container, and return its name.

    The caller must hold this environment's session lock under work. On return,
    only the idle process remains; a later docker exec starts the workload.
    """
    spec = container_spec(bundle, work, relative, manifest)
    accepted = [manifest["image"]]
    if manifest.get("image_config_digest"):
        accepted.append(manifest["image_config_digest"])

    def installed_image():
        for candidate in accepted:
            result = subprocess.run(
                ["docker", "image", "inspect", candidate, "--format", "{{.Id}}"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )
            if result.returncode == 0 and result.stdout.strip() in accepted:
                return result.stdout.strip()
        return None

    # Verify installed image digests; load the offline delivery archive only if needed.
    image = installed_image()
    if image is None:
        archive = checked_path(
            bundle, str(Path(relative) / manifest.get("image_archive", "runtime-image.tar"))
        )
        expected = manifest.get("image_archive_sha256")
        if expected and sha256(archive) != expected:
            raise ValueError("Docker image archive SHA256 mismatch")
        subprocess.check_call(["docker", "load", "-i", str(archive)])
        image = installed_image()
    if image is None:
        raise ValueError("Docker image identity mismatch")
    raw = subprocess.run(
        ["docker", "inspect", spec["name"]],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    if raw.returncode == 0:
        # A matching name is insufficient: verify image, label, platform options, and mounts.
        value = json.loads(raw.stdout)[0]
        if (
            value["Image"] != image
            or value["Config"]["Labels"].get("graphcast.delivery") != spec["identity"]
        ):
            raise ValueError("Existing container does not match the delivery")
        unconfined = "seccomp=unconfined" in (value["HostConfig"].get("SecurityOpt") or [])
        if unconfined != relative.startswith("environments/bw1100/"):
            raise ValueError("Existing container security options differ from platform policy")
        found = {m["Destination"]: (m["Source"], not m["RW"]) for m in value["Mounts"]}
        if any(found.get(dst) != (src, ro) for src, dst, ro in spec["mounts"]):
            raise ValueError("Existing container mounts changed")
        if not value["State"]["Running"]:
            subprocess.check_call(["docker", "start", spec["name"]], stdout=subprocess.DEVNULL)
        initialize(name=spec["name"], relative=relative, python=manifest["python"])
        return spec["name"]
    # Create an idle container; the CLI stops it after a run and restarts it for reuse.
    cmd = [
        "docker",
        "run",
        "-d",
        "--name",
        spec["name"],
        "--label",
        "graphcast.delivery=" + spec["identity"],
        "--network",
        "none",
        "--ipc",
        "private",
        "--shm-size",
        "4g",
        "--cpus",
        "8",
        "--memory",
        "128g",
        "--ulimit",
        "core=0",
    ]
    if relative.startswith("environments/bw1100/"):
        cmd += ["--device=/dev/kfd", "--device=/dev/dri", "--security-opt", "seccomp=unconfined"]
    else:
        cmd += ["--runtime", "nvidia", "--gpus", "all"]
    for src, dst, ro in spec["mounts"]:
        if ":" in src or "," in src:
            raise ValueError("Docker bind paths cannot contain colon/comma")
        cmd += ["--mount", "type=bind,src={},dst={}{}".format(src, dst, ",readonly" if ro else "")]
    for key, value in manifest["environment"].items():
        cmd += ["-e", key + "=" + value]
    cmd += [
        "-e",
        "PYTHONDONTWRITEBYTECODE=1",
        "-e",
        "PYTHONNOUSERSITE=1",
        "--entrypoint",
        "/bin/bash",
        image,
        "-c",
        "exec sleep infinity",
    ]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL)
    initialize(name=spec["name"], relative=relative, python=manifest["python"])
    return spec["name"]


def initialize(name, relative, python):
    """Reject leftover tasks, then clean this BW1100 session's private compiler sockets."""
    # docker run may return while its initial bash is still exec'ing sleep.
    for attempt in range(10):
        processes = subprocess.check_output(
            ["docker", "top", name, "-eo", "pid,args"], universal_newlines=True
        )
        rows = [line for line in processes.splitlines()[1:] if line.strip()]
        if len(rows) == 1 and rows[0].split()[1:] == ["sleep", "infinity"]:
            break
        if attempt == 9:
            raise RuntimeError("Delivery container has active processes; this session cannot be reused")
        time.sleep(0.2)
    if relative.startswith("environments/bw1100/"):
        # A prior stopped compiler may leave socket files. This path is a bind
        # of this exact container's dedicated scratch directory, never shared.
        cleanup = (
            'import os,shutil; from pathlib import Path; p=Path("/tmp/flagtree-r10-0"); '
            "[(shutil.rmtree(str(x)) if x.is_dir() and not x.is_symlink() else x.unlink()) for x in p.iterdir()]; "
            "os.chown(str(p),0,0); os.chmod(str(p),0o700)"
        )
        subprocess.check_call(["docker", "exec", name, python, "-c", cleanup])


def exec_command(name, python, release, args, environment=None):
    """Build argv for delivery.py inside the container without executing a shell.

    Paths in release/args are already container paths; python comes from the manifest.
    --inside <context.json> runs a workload; --inside-check checks dependencies.
    """
    cmd = ["docker", "exec", "-w", release]
    for key, value in (environment or {}).items():
        cmd += ["-e", key + "=" + str(value)]
    return cmd + [name, python, release + "/scripts/delivery.py"] + args
