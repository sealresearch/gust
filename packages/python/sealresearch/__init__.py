"""Seal Research's on-device models, from Python: one interface for every model, on the Gust runtime.

    import sealresearch as sr

    relay = sr.Model.load("relay-1")                  # downloads the files the first time, checks them, keeps them
    relay.run({"text": "Show me the Eiffel Tower on the map", "tools": tools})

The same calls work for every model: what a model takes in and gives back is in `model.info`.
"""
from ._fetch import cache_dir, registry
from .model import GustError, Image, Model, Pcm, Stream, describe, models, runtime, set_threads

__version__ = "0.1.0"
__all__ = ["GustError", "Image", "Model", "Pcm", "Stream", "cache_dir", "describe", "models", "registry", "runtime", "set_threads"]
