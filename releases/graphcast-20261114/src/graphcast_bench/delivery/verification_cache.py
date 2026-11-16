"""Optional local hash evidence, bound to a material root and its manifest."""
import hashlib
import json
from pathlib import Path
import tempfile


def cache_path(directory, root):
    key = hashlib.sha256(str(root).encode()).hexdigest()
    return Path(directory) / (key + '.json')


def load(path, identity):
    try:
        record = json.loads(path.read_text())
        if record.get('identity') == identity and isinstance(record.get('files'), dict):
            return record['files']
    except (OSError, ValueError, AttributeError):
        pass
    return {}


def invalidate(path):
    """Retire old evidence before checking; a failed check must not revive it."""
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return True
    except OSError:
        return False


def save(path, identity, files):
    """Publish complete evidence atomically; cache failures never reject valid materials."""
    temporary = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode='w', dir=str(path.parent), delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(dict(identity=identity, files=files), stream, separators=(',', ':'))
        temporary.replace(path)
    except OSError:
        pass
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except OSError:
                pass


def fingerprint(path):
    value = path.stat()
    return [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns]
