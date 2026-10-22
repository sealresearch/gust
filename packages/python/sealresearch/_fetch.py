"""Finding, downloading and checking model files. The registry (_registry.json) is the same one the sealpup command line
reads: the model ids, and for each model the Hugging Face repository and every file with its size and SHA-256."""
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request

_REGISTRY = None


def registry():
    global _REGISTRY
    if _REGISTRY is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_registry.json")
        if not os.path.exists(path):  # a checkout of the repository
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "cli", "registry.json")
        with open(path, encoding="utf-8") as f:
            _REGISTRY = json.load(f)
    return _REGISTRY


def resolve(model_id):
    """The registry entry of a model, by id or alias, and its canonical id."""
    reg = registry()
    model_id = reg.get("aliases", {}).get(model_id, model_id)
    entry = reg["models"].get(model_id)
    if entry is None:
        raise KeyError(f"no model called {model_id!r}; the registry has {', '.join(sorted(reg['models']))}")
    return model_id, entry


def cache_dir():
    """Where downloads are kept: $GUST_CACHE, else the platform's cache folder."""
    if os.environ.get("GUST_CACHE"):
        return os.environ["GUST_CACHE"]
    home = os.path.expanduser("~")
    if sys.platform.startswith("win"):
        return os.path.join(os.environ.get("LOCALAPPDATA") or os.path.join(home, "AppData", "Local"), "gust", "cache")
    if sys.platform == "darwin":
        return os.path.join(home, "Library", "Caches", "gust")
    return os.path.join(os.environ.get("XDG_CACHE_HOME") or os.path.join(home, ".cache"), "gust")


def _check(data, file):
    if len(data) != file["size"]:
        raise ValueError(f"{file['name']} is {len(data)} bytes, expected {file['size']}")
    got = hashlib.sha256(data).hexdigest()
    if got != file["sha256"]:
        raise ValueError(f"{file['name']} does not match its checksum (got {got[:12]}, expected {file['sha256'][:12]})")


def _download(url, file, progress):
    req = urllib.request.Request(url, headers={"User-Agent": "sealresearch-python"})
    try:
        with urllib.request.urlopen(req, timeout=60) as res:
            chunks, loaded = [], 0
            while True:
                chunk = res.read(1 << 20)
                if not chunk:
                    break
                chunks.append(chunk)
                loaded += len(chunk)
                if progress:
                    progress(file["name"], loaded, file["size"])
    except urllib.error.HTTPError as e:
        raise OSError(f"{file['name']}: {e.code} {e.reason} from {url}") from None
    data = b"".join(chunks)
    _check(data, file)
    return data


def obtain(model_id, entry, *, directory=None, cache=True, base_url="https://huggingface.co", names=None, verify=True,
           progress=None):
    """The files of a model as [(name, bytes)]: read from `directory`, or from the cache, or downloaded, checked and cached.
    `names` limits it to some of the files."""
    out = []
    for file in entry["files"]:
        if names is not None and file["name"] not in names:
            continue
        if directory is not None:
            with open(os.path.join(directory, file["name"]), "rb") as f:
                data = f.read()
            if verify:
                _check(data, file)
            out.append((file["name"], data))
            continue
        folder = None
        if cache:
            folder = os.path.join(cache if isinstance(cache, str) else cache_dir(), model_id)
            path = os.path.join(folder, file["name"])
            try:
                with open(path, "rb") as f:
                    data = f.read()
                _check(data, file)
                if progress:
                    progress(file["name"], file["size"], file["size"])
                out.append((file["name"], data))
                continue
            except (OSError, ValueError):
                pass  # missing or damaged: download it again
        url = f"{base_url.rstrip('/')}/{entry['repo']}/resolve/{entry.get('revision', 'main')}/{file['name']}"
        data = _download(url, file, progress)
        if folder:
            os.makedirs(folder, exist_ok=True)
            part = f"{path}.{os.getpid()}.part"
            with open(part, "wb") as f:
                f.write(data)
            os.replace(part, path)
        out.append((file["name"], data))
    return out
