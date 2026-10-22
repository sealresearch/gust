"""The shared library and its C interface (sdk/include/gust_model.h), through ctypes. Nothing here knows about any one model."""
import ctypes
import os
import platform
import sys
from ctypes import POINTER, Structure, c_char_p, c_int, c_size_t, c_uint32, c_void_p

ABI_VERSION = 1

STATUS = {
    0: "ok", 1: "argument", 2: "unknown_model", 3: "files", 4: "open", 5: "input", 6: "run", 7: "unsupported",
    8: "memory", 9: "state",
}

BYTES, TEXT, JSON, PCM_F32, PCM_I16, IMAGE_GRAY8, IMAGE_RGBA8 = range(7)


class Buffer(Structure):
    _fields_ = [("size", c_size_t), ("type", c_int), ("data", c_void_p), ("len", c_size_t), ("rate", c_uint32),
                ("channels", c_uint32), ("width", c_uint32), ("height", c_uint32)]


class File(Structure):
    _fields_ = [("name", c_char_p), ("data", c_void_p), ("len", c_size_t)]


def platform_name():
    """The folder name of this platform's library in the repository, and the name a wheel uses."""
    machine = platform.machine().lower()
    arm = machine in ("arm64", "aarch64")
    if sys.platform.startswith("win"):
        return "windows-x64"
    if sys.platform == "darwin":
        return "macos-arm64" if arm else "macos-x64"
    return "linux-arm64" if arm else "linux-x64"


def _file_name():
    return "gust.dll" if sys.platform.startswith("win") else "libgust.dylib" if sys.platform == "darwin" else "libgust.so"


def _candidates():
    if os.environ.get("GUST_LIB"):
        yield os.environ["GUST_LIB"]
    here = os.path.dirname(os.path.abspath(__file__))
    yield os.path.join(here, "_native", _file_name())
    # A checkout of the repository: packages/python/sealresearch -> out/lib/<platform>/
    yield os.path.join(here, "..", "..", "..", "out", "lib", platform_name(), _file_name())


def _load():
    tried = []
    for path in _candidates():
        path = os.path.abspath(path)
        tried.append(path)
        if os.path.exists(path):
            if sys.platform.startswith("win") and hasattr(os, "add_dll_directory"):
                os.add_dll_directory(os.path.dirname(path))
            return ctypes.CDLL(path)
    raise OSError("the Gust library was not found; looked in:\n  " + "\n  ".join(tried) +
                  "\nSet GUST_LIB to its path, or install a wheel for this platform.")


lib = _load()

_sigs = {
    "gust_model_abi_version": (c_uint32, []),
    "gust_model_last_error": (c_char_p, []),
    "gust_model_ids": (POINTER(c_char_p), []),
    "gust_model_describe": (c_char_p, [c_char_p]),
    "gust_model_set_threads": (c_uint32, [c_uint32]),
    "gust_model_open": (c_int, [c_char_p, POINTER(File), c_size_t, c_char_p, POINTER(c_void_p)]),
    "gust_model_info": (c_char_p, [c_void_p]),
    "gust_model_run": (c_int, [c_void_p, POINTER(Buffer), c_size_t, c_char_p, POINTER(c_void_p)]),
    "gust_model_close": (None, [c_void_p]),
    "gust_stream_open": (c_int, [c_void_p, c_char_p, POINTER(c_void_p)]),
    "gust_stream_push": (c_int, [c_void_p, POINTER(Buffer), c_size_t]),
    "gust_stream_pull": (c_int, [c_void_p, POINTER(c_void_p), POINTER(c_int)]),
    "gust_stream_finish": (c_int, [c_void_p]),
    "gust_stream_close": (None, [c_void_p]),
    "gust_result_count": (c_size_t, [c_void_p]),
    "gust_result_get": (POINTER(Buffer), [c_void_p, c_size_t]),
    "gust_result_free": (None, [c_void_p]),
}
for _name, (_res, _args) in _sigs.items():
    _fn = getattr(lib, _name)
    _fn.restype = _res
    _fn.argtypes = _args

if lib.gust_model_abi_version() != ABI_VERSION:
    raise OSError(f"the Gust library speaks model API version {lib.gust_model_abi_version()}, this package needs {ABI_VERSION}")
