"""Tests of the Python package. They need the model files: RELAY_MODEL_DIR is a checkout of Hugging Face's sealresearch/relay.
Downloads are served from a local HTTP server that mimics Hugging Face. Run: RELAY_MODEL_DIR=... python -m unittest discover tests"""
import http.server
import json
import os
import shutil
import tempfile
import threading
import unittest

import sealresearch as sr

DIR = os.environ.get("RELAY_MODEL_DIR")
TOOLS = [
    {"name": "show_map", "description": "Shows a location on the map.",
     "params": [{"name": "query", "type": "string", "description": "The place to show.", "required": True}]},
    {"name": "turn_on_flashlight", "description": "Turns the flashlight on.", "params": []},
]
REQUEST = {"text": "Show me the Eiffel Tower on the map", "tools": TOOLS}
EXPECTED = [{"tool": "show_map", "args": {"query": "Eiffel Tower"}}]


def serve(corrupt=None):
    """A local stand-in for huggingface.co/<repo>/resolve/<revision>/<file>. Returns (url, hits, stop)."""
    hits = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            name = self.path.rsplit("/", 1)[-1]
            path = os.path.join(DIR, name)
            if not os.path.exists(path):
                self.send_error(404)
                return
            hits.append(name)
            with open(path, "rb") as f:
                body = bytearray(f.read())
            if name == corrupt:
                body[len(body) // 2] ^= 0xFF
            self.send_response(200)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(bytes(body))

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    def stop():
        server.shutdown()
        server.server_close()

    return f"http://127.0.0.1:{server.server_address[1]}", hits, stop


@unittest.skipUnless(DIR, "set RELAY_MODEL_DIR to a checkout of the Hugging Face repository")
class ModelTests(unittest.TestCase):
    def test_library_and_registry(self):
        self.assertIn("relay-1", sr.models())
        info = sr.describe("relay")  # an alias
        self.assertEqual(info["id"], "relay-1")
        self.assertEqual(info["kind"], "tool-calling")
        self.assertIn("relay-prefilter.bin", info["files"]["optional"])
        with self.assertRaises(sr.GustError) as e:
            sr.describe("no-such-model")
        self.assertEqual(e.exception.status, "unknown_model")

    def test_runtime_and_memory(self):
        self.assertTrue(sr.runtime())
        with sr.Model.load("relay-1", directory=DIR) as model:
            self.assertGreater(model.memory_bytes, 10_000_000)

    def test_run_from_a_folder(self):
        with sr.Model.load("relay-1", directory=DIR) as model:
            self.assertEqual(model.run(REQUEST), EXPECTED)
            self.assertEqual(model(REQUEST), EXPECTED)  # __call__
            self.assertEqual(model.run({"text": "Tell me a joke", "tools": TOOLS}), [])
            self.assertEqual(model.info["id"], "relay-1")

    def test_options_and_run_modes(self):
        with sr.Model.load("relay-1", directory=DIR, options={"prefilter": False, "k": 5}) as model:
            self.assertEqual(model.run(REQUEST), EXPECTED)
            explained = model.run(REQUEST, options={"mode": "explain"})
            self.assertIn("calls", explained)
            self.assertIn("why", explained["calls"][0])

    def test_errors_carry_the_library_message(self):
        with sr.Model.load("relay-1", directory=DIR) as model:
            with self.assertRaises(sr.GustError) as e:
                model.run("not a request")  # text that is not a JSON request
            self.assertEqual(e.exception.status, "input")
            self.assertIn("JSON", str(e.exception))
            with self.assertRaises(sr.GustError) as e:
                model.run(sr.Pcm([0.0, 0.1], 16000))
            self.assertEqual(e.exception.status, "input")
            with self.assertRaises(sr.GustError) as e:
                model.stream()
            self.assertEqual(e.exception.status, "unsupported")
        with self.assertRaises(sr.GustError) as e:
            model.run(REQUEST)  # closed
        self.assertEqual(e.exception.status, "state")

    def test_same_answers_on_one_thread_and_four(self):
        texts = ["Show me the Eiffel Tower on the map", "Turn on the flashlight and show Paris", "Set an alarm", "hello"]
        with sr.Model.load("relay-1", directory=DIR, threads=1) as model:
            one = [model.run({**REQUEST, "text": t}) for t in texts]
            self.assertEqual(sr.set_threads(4), 4)
            four = [model.run({**REQUEST, "text": t}) for t in texts]
            sr.set_threads(1)
        self.assertEqual(one, four)

    def test_two_python_threads_at_once(self):
        results, errors = [], []
        with sr.Model.load("relay-1", directory=DIR) as model:
            def work():
                try:
                    for _ in range(8):
                        results.append(model.run(REQUEST))
                except Exception as e:  # pragma: no cover
                    errors.append(e)
            threads = [threading.Thread(target=work) for _ in range(4)]
            [t.start() for t in threads]
            [t.join() for t in threads]
        self.assertEqual(errors, [])
        self.assertEqual(len(results), 32)
        self.assertTrue(all(r == EXPECTED for r in results))

    def test_download_check_and_cache(self):
        url, hits, stop = serve()
        cache = tempfile.mkdtemp(prefix="gust-py-")
        try:
            progress = []
            with sr.Model.load("relay-1", base_url=url, cache=cache, progress=lambda *a: progress.append(a)) as model:
                self.assertEqual(model.run(REQUEST), EXPECTED)
            self.assertEqual(sorted(hits), sorted(f["name"] for f in sr.registry()["models"]["relay-1"]["files"]))
            self.assertTrue(any(name == "relay.gust" and loaded == total for name, loaded, total in progress))
            n = len(hits)
            with sr.Model.load("relay-1", base_url=url, cache=cache) as model:
                self.assertEqual(model.run(REQUEST), EXPECTED)
            self.assertEqual(len(hits), n, "the second load should come from the cache")
            # a damaged cached file is fetched again
            path = os.path.join(cache, "relay-1", "relay-route.gust")
            with open(path, "r+b") as f:
                f.seek(100)
                f.write(b"\x00\x01\x02")
            with sr.Model.load("relay-1", base_url=url, cache=cache) as model:
                self.assertEqual(model.run(REQUEST), EXPECTED)
            self.assertEqual(len(hits), n + 1)
            # without the optional prefilter
            hits.clear()
            shutil.rmtree(cache)
            with sr.Model.load("relay-1", base_url=url, cache=cache, optional=False) as model:
                self.assertEqual(model.run(REQUEST), EXPECTED)
            self.assertNotIn("relay-prefilter.bin", hits)
        finally:
            stop()
            shutil.rmtree(cache, ignore_errors=True)

    def test_refuses_a_download_that_does_not_match(self):
        url, hits, stop = serve(corrupt="relay-args.gust")
        cache = tempfile.mkdtemp(prefix="gust-py-")
        try:
            with self.assertRaisesRegex(ValueError, "relay-args.gust does not match its checksum"):
                sr.Model.load("relay-1", base_url=url, cache=cache)
        finally:
            stop()
            shutil.rmtree(cache, ignore_errors=True)

    def test_missing_files_are_a_library_error(self):
        # the registry check passes with verify off, but Relay needs all its files
        with self.assertRaises(FileNotFoundError):
            sr.Model.load("relay-1", directory=tempfile.gettempdir())


class ValueTests(unittest.TestCase):
    def test_pcm(self):
        p = sr.Pcm([0.0, 0.5, -0.5], 24000, 1)
        self.assertEqual((len(p), p.dtype, p.rate, p.channels), (3, "float32", 24000, 1))
        import array
        q = sr.Pcm(array.array("h", [1, 2, 3, 4]), 16000, 2)
        self.assertEqual(q.dtype, "int16")
        r = sr.Pcm(array.array("f", [1.0, 2.0]).tobytes(), 8000, dtype="float32")
        self.assertEqual(len(r), 2)
        with self.assertRaises(ValueError):
            sr.Pcm(b"\x00\x00", 8000)
        with self.assertRaises(ValueError):
            sr.Image(b"", 1, 1, "bgr")


@unittest.skipUnless("toy-gain" in sr.models(), "build the library with GUST_TOY=1 to test streams and audio")
class ToyModelTests(unittest.TestCase):
    """Models that have nothing to do with Relay, built into test libraries: text in and out, and streaming audio."""

    def test_text_model(self):
        with sr.Model.open("toy-upper", options={"suffix": "!"}) as m:
            self.assertEqual(m.info["id"], "toy-upper")
            self.assertEqual(m.run("hello"), "HELLO!")
            with self.assertRaises(sr.GustError) as e:
                m.run({"not": "text"})
            self.assertEqual(e.exception.status, "input")
            with self.assertRaises(sr.GustError) as e:
                m.stream()
            self.assertEqual(e.exception.status, "unsupported")

    def test_audio_stream(self):
        import array
        with sr.Model.open("toy-gain", options={"gain": 3}) as m:
            with m.stream() as s:
                self.assertEqual(s.pull(), [])
                self.assertFalse(s.done)
                s.push(sr.Pcm(array.array("f", [1, 2, 3, 4]), 16000))
                (out,) = s.pull()
                self.assertIsInstance(out, sr.Pcm)
                self.assertEqual((out.rate, out.channels, out.dtype), (16000, 1, "float32"))
                self.assertEqual(list(out.samples), [3, 6, 9, 12])
                s.push(sr.Pcm([-1.0, 0.5], 16000))
                s.finish()
                with self.assertRaises(sr.GustError) as e:
                    s.push(sr.Pcm([0.0], 16000))
                self.assertEqual(e.exception.status, "state")
                rest = list(s.drain())
                self.assertTrue(s.done)
                self.assertEqual(list(rest[0].samples), [-3.0, 1.5])
                with self.assertRaises(sr.GustError) as e:
                    s.push("text")
                self.assertIn(e.exception.status, ("state", "input"))

    def test_a_stream_outlives_its_model(self):
        import array
        m = sr.Model.open("toy-gain")
        s = m.stream()
        m.close()
        s.push(sr.Pcm(array.array("f", [1.0, 2.0]), 8000))
        s.finish()
        outs = list(s.drain())
        self.assertEqual(list(outs[0].samples), [2.0, 4.0])
        s.close()

    def test_numpy_audio_when_available(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is not installed")
        with sr.Model.open("toy-gain") as m, m.stream() as s:
            s.push(sr.Pcm(np.array([0.25, -0.5], dtype="float32"), 22050))
            (out,) = s.pull()
            self.assertIsInstance(out.samples, np.ndarray)
            self.assertEqual(out.samples.tolist(), [0.5, -1.0])


if __name__ == "__main__":
    unittest.main()
