"""Asks the same requests through the two ways sealpup can reach a model, the program started for each request and the Gust
library in the process, and checks that the calls are identical.

    python cli/scripts/compare_engines.py PATH/TO/sealpup

Relay must be installed (sealpup install relay-1). The library is the one SEALPUP_LIBRARY names, or the installed one."""
import json
import os
import subprocess
import sys

REQUESTS = [
    "Set an alarm for 6:45 tomorrow and text Sam I'm running late",
    "What is the weather in Rotterdam?",
    "Show me the Eiffel Tower on the map",
    "Turn on the flashlight",
    "Turn the volume up to 80 percent",
    "Play some Bowie",
    "Send a message to Anna saying I will be late, and set a timer for ten minutes",
    "Tell me a joke",
    "Wake me up at 7 on Friday",
    "Call mom",
    "Remind me to buy milk at 5pm",
    "What's the weather like in Tokyo tomorrow and show me the Tokyo Tower",
    "Set the volume to 20, then play jazz",
    "Text +31 6 1234 5678 that the package arrived",
]


def ask(sealpup, way, text):
    env = dict(os.environ, SEALPUP_ENGINE=way)
    p = subprocess.run(
        [sealpup, "run", "relay-1", "--json", "--now", "2026-10-18T09:00:00", "--threads", "4", text],
        capture_output=True,
        text=True,
        env=env,
    )
    if p.returncode != 0:
        raise SystemExit(f"{way} failed on {text!r}: {p.stderr.strip()[-300:]}")
    return json.loads(p.stdout.strip().splitlines()[-1])


def main():
    sealpup = sys.argv[1]
    same = 0
    for text in REQUESTS:
        runner, library = ask(sealpup, "runner", text), ask(sealpup, "library", text)
        ok = runner["calls"] == library["calls"] and runner["held_back"] == library["held_back"]
        same += ok
        print("same     " if ok else "DIFFERENT", text[:60], "->", [c["tool"] for c in library["calls"]])
        if not ok:
            print("  runner :", json.dumps(runner["calls"])[:200])
            print("  library:", json.dumps(library["calls"])[:200])
    print(f"\n{same} of {len(REQUESTS)} requests give identical calls through the program and through the library")
    if same != len(REQUESTS):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
