"""Models, streams and the values that go in and out of them. Nothing here is specific to a model: a model says what it takes
and gives in its description (Model.info), and this module moves typed buffers."""
import array
import ctypes
import json
from ctypes import byref, c_int, c_void_p, string_at

from . import _fetch
from ._lib import (BYTES, IMAGE_GRAY8, IMAGE_RGBA8, JSON, PCM_F32, PCM_I16, STATUS, TEXT, Buffer, File, lib)

__all__ = ["GustError", "Model", "Stream", "Pcm", "Image", "models", "describe", "set_threads", "runtime"]


class GustError(RuntimeError):
    """A failure reported by the library. `status` is its name ("input", "files", "unsupported", ...)."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code
        self.status = STATUS.get(code, str(code))


def _check(code):
    if code:
        message = lib.gust_model_last_error() or b""
        raise GustError(code, message.decode("utf-8", "replace") or f"failed ({STATUS.get(code, code)})")


def _numpy():
    try:
        import numpy
        return numpy
    except ImportError:
        return None


class Pcm:
    """Audio: interleaved samples, float32 or int16, with their rate and channel count.

    `samples` can be a numpy array (float32 or int16), an array.array ('f' or 'h'), a list of floats, or bytes together with
    `dtype` ("float32" or "int16"). Outputs come back as numpy arrays when numpy is installed, as array.array otherwise."""

    def __init__(self, samples, rate, channels=1, dtype=None):
        self.rate, self.channels = int(rate), int(channels)
        np = _numpy()
        if isinstance(samples, (bytes, bytearray, memoryview)):
            if dtype not in ("float32", "int16"):
                raise ValueError('bytes need dtype="float32" or dtype="int16"')
            code = "f" if dtype == "float32" else "h"
            arr = array.array(code)
            arr.frombytes(bytes(samples))
            samples = arr
        elif np is not None and isinstance(samples, np.ndarray):
            if str(samples.dtype) not in ("float32", "int16"):
                samples = samples.astype("float32")
        elif isinstance(samples, array.array):
            if samples.typecode not in ("f", "h"):
                raise ValueError("an array.array of samples must be 'f' (float32) or 'h' (int16)")
        else:
            samples = array.array("f", samples)
        self.samples = samples

    @property
    def dtype(self):
        s = self.samples
        return "int16" if (getattr(s, "typecode", None) == "h" or str(getattr(s, "dtype", "")) == "int16") else "float32"

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        return f"Pcm({len(self)} samples, {self.dtype}, {self.rate} Hz, {self.channels} ch)"


class Image:
    """A frame: `pixels` (bytes or a uint8 numpy array), `width`, `height` and a `format` of "gray8" or "rgba8"."""

    def __init__(self, pixels, width, height, format="gray8"):
        if format not in ("gray8", "rgba8"):
            raise ValueError('format must be "gray8" or "rgba8"')
        self.pixels, self.width, self.height, self.format = pixels, int(width), int(height), format

    def __repr__(self):
        return f"Image({self.width}x{self.height}, {self.format})"


def _view(obj):
    """(address, byte count, object to keep alive) of anything that holds bytes."""
    if hasattr(obj, "ctypes") and hasattr(obj, "nbytes"):  # numpy
        if not obj.flags["C_CONTIGUOUS"]:
            obj = obj.copy(order="C")
        return obj.ctypes.data, obj.nbytes, obj
    if isinstance(obj, array.array):
        address, count = obj.buffer_info()
        return address, count * obj.itemsize, obj
    if isinstance(obj, bytes):
        return ctypes.cast(ctypes.c_char_p(obj), c_void_p).value or 0, len(obj), obj
    if isinstance(obj, memoryview):
        obj = bytes(obj)
        return ctypes.cast(ctypes.c_char_p(obj), c_void_p).value or 0, len(obj), obj
    if isinstance(obj, bytearray):
        raw = (ctypes.c_char * len(obj)).from_buffer(obj)
        return ctypes.addressof(raw), len(obj), raw
    raise TypeError(f"cannot pass a {type(obj).__name__} to a model")


def _buffer(value, keep):
    b = Buffer()
    b.size = ctypes.sizeof(Buffer)
    if isinstance(value, Pcm):
        b.type = PCM_I16 if value.dtype == "int16" else PCM_F32
        address, n, k = _view(value.samples)
        b.rate, b.channels = value.rate, value.channels
    elif isinstance(value, Image):
        b.type = IMAGE_RGBA8 if value.format == "rgba8" else IMAGE_GRAY8
        address, n, k = _view(value.pixels)
        b.width, b.height = value.width, value.height
    elif isinstance(value, str):
        b.type = TEXT
        data = value.encode("utf-8")
        address, n, k = _view(data)
    elif isinstance(value, (bytes, bytearray, memoryview)):
        b.type = BYTES
        address, n, k = _view(value)
    else:  # anything else is sent as JSON: dicts, lists, numbers, booleans, None
        b.type = JSON
        data = json.dumps(value, ensure_ascii=False).encode("utf-8")
        address, n, k = _view(data)
    b.data, b.len = address, n
    keep.append(k)
    return b


def _buffers(values):
    keep = []
    arr = (Buffer * max(len(values), 1))(*[_buffer(v, keep) for v in values])
    return arr, len(values), keep


def _decode(b):
    raw = string_at(b.data, b.len) if b.len else b""
    if b.type == JSON:
        return json.loads(raw.decode("utf-8"))
    if b.type == TEXT:
        return raw.decode("utf-8")
    if b.type in (PCM_F32, PCM_I16):
        np = _numpy()
        if np is not None:
            samples = np.frombuffer(raw, dtype="float32" if b.type == PCM_F32 else "int16").copy()
        else:
            samples = array.array("f" if b.type == PCM_F32 else "h")
            samples.frombytes(raw)
        return Pcm(samples, b.rate, b.channels)
    if b.type in (IMAGE_GRAY8, IMAGE_RGBA8):
        return Image(raw, b.width, b.height, "rgba8" if b.type == IMAGE_RGBA8 else "gray8")
    return raw


def _outputs(result):
    """Copies a result into Python values and frees it."""
    try:
        return [_decode(lib.gust_result_get(result, i).contents) for i in range(lib.gust_result_count(result))]
    finally:
        lib.gust_result_free(result)


def _options(options):
    return None if not options else json.dumps(options).encode("utf-8")


def models():
    """The ids of the models in this build of the library."""
    ids, out, i = lib.gust_model_ids(), [], 0
    while ids[i]:
        out.append(ids[i].decode())
        i += 1
    return out


def describe(model_id):
    """What a model is, takes and gives, as a dict, before it is opened."""
    model_id = _fetch.registry().get("aliases", {}).get(model_id, model_id)
    text = lib.gust_model_describe(model_id.encode())
    if text is None:
        _check(2)
    return json.loads(text.decode("utf-8"))


def runtime():
    """The instruction set this build runs, such as "x86-64 AVX-512 VNNI"."""
    return lib.gust_model_runtime().decode()


def set_threads(threads):
    """Threads for the operators, for the whole process (0 asks for the number of cores, up to 8). The answers do not
    depend on it. Returns the count in use."""
    return lib.gust_model_set_threads(int(threads))


class Model:
    """A loaded model.

        model = Model.load("relay-1")
        model.run({"text": "...", "tools": [...]})      # one input in, one value out (a list when there are several)
    """

    def __init__(self, handle, model_id):
        self._h = handle
        self.id = model_id
        self.info = json.loads(lib.gust_model_info(handle).decode("utf-8"))

    @classmethod
    def load(cls, model_id, *, options=None, threads=None, directory=None, cache=True, base_url="https://huggingface.co",
             optional=True, verify=True, progress=None):
        """Finds the model's files (in `directory`, in the cache, or downloaded from Hugging Face and checked against the
        registry's SHA-256), then opens it.

        options    what the model's description lists under options.open, as a dict
        threads    threads for the operators, for the whole process (see set_threads)
        directory  read the files from this folder instead of downloading them
        cache      True for the default cache folder, a path, or False to download every time
        optional   also fetch the files the model lists as optional (Relay's prefilter, for example)
        progress   called as progress(file_name, bytes_so_far, total_bytes)"""
        model_id, entry = _fetch.resolve(model_id)
        info = describe(model_id)
        wanted = info.get("files", {})
        names = None
        if wanted:
            names = set(wanted.get("required", []))
            if optional:
                names |= set(wanted.get("optional", []))
        files = _fetch.obtain(model_id, entry, directory=directory, cache=cache, base_url=base_url, names=names, verify=verify,
                              progress=progress)
        if threads is not None:
            set_threads(threads)
        return cls.open(model_id, files, options=options)

    @classmethod
    def open(cls, model_id, files=(), *, options=None):
        """Opens a model from files you already have in memory (a dict or a list of (name, bytes)): no registry, no download.
        `load` is this plus finding the files."""
        files = list(files.items()) if isinstance(files, dict) else list(files)
        arr = (File * max(len(files), 1))()
        keep = []
        for i, (name, data) in enumerate(files):
            n = name.encode()
            data = bytes(data)
            keep += [n, data]
            arr[i].name, arr[i].data, arr[i].len = n, ctypes.cast(ctypes.c_char_p(data), c_void_p), len(data)
        handle = c_void_p()
        _check(lib.gust_model_open(model_id.encode(), arr, len(files), _options(options), byref(handle)))
        return cls(handle, model_id)

    def run(self, *inputs, options=None):
        """One call: the inputs go in, the outputs come back. A single output is returned as itself, several as a list.
        What a model takes is in `model.info["inputs"]`. Python values go in as JSON (dict, list, number), text (str), bytes,
        audio (Pcm) or images (Image)."""
        self._open()
        arr, n, keep = _buffers(inputs)
        result = c_void_p()
        _check(lib.gust_model_run(self._h, arr, n, _options(options), byref(result)))
        out = _outputs(result)
        return out[0] if len(out) == 1 else out

    __call__ = run

    def stream(self, options=None):
        """A stream for models that work on live input (speech in, audio out). See Stream."""
        self._open()
        handle = c_void_p()
        _check(lib.gust_stream_open(self._h, _options(options), byref(handle)))
        return Stream(handle, self)

    @property
    def memory_bytes(self):
        """Bytes the loaded model holds (0 when the model does not say)."""
        self._open()
        return int(lib.gust_model_memory(self._h))

    def _open(self):
        if not self._h:
            raise GustError(9, "the model is closed")

    def close(self):
        if self._h:
            lib.gust_model_close(self._h)
            self._h = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        return f"Model({self.id!r})"


class Stream:
    """Push input as it arrives, pull what is ready, finish when the input ends.

        with model.stream() as s:
            for chunk in audio_chunks:
                s.push(chunk)
                for out in s.pull(): ...        # may be empty
            s.finish()
            for out in s.drain(): ...           # everything that is left
    """

    def __init__(self, handle, model):
        self._h = handle
        self._model = model  # keeps the model object alive
        self.done = False

    def push(self, *inputs):
        arr, n, keep = _buffers(inputs)
        _check(lib.gust_stream_push(self._h, arr, n))

    def pull(self):
        """The outputs that are ready now, as a list (possibly empty). `done` is True once the stream is finished and drained."""
        result, done = c_void_p(), c_int()
        _check(lib.gust_stream_pull(self._h, byref(result), byref(done)))
        self.done = bool(done.value)
        return _outputs(result)

    def finish(self):
        _check(lib.gust_stream_finish(self._h))

    def drain(self):
        """After finish(): yields every remaining output until the stream is done."""
        while True:
            outs = self.pull()
            yield from outs
            if self.done:
                return

    def close(self):
        if self._h:
            lib.gust_stream_close(self._h)
            self._h = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
