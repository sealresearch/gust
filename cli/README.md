# seal

Install Seal's on-device models and run them.

```sh
seal install relay-1
seal run relay-1 "Set an alarm for 6:45 tomorrow"
```

```
  relay-1 · built-in phone tools (9 tools) · 8 threads

  › Show me the Eiffel Tower on the map

  ✓ show_map
      query  "Eiffel Tower"  copied     100%

  19.9 ms · 9 tools read · loaded in 48 ms (21.2 MB)
```

`seal` is one small program with no runtime to install. It downloads a model from Hugging Face, downloads the program
that runs it from a release of this repository, checks both, and keeps them in `~/.seal`. Everything after that runs
offline.

## Install

macOS and Linux (and Git Bash on Windows):

```sh
curl -fsSL https://raw.githubusercontent.com/sealresearch/gust/main/cli/install.sh | sh
```

Windows (PowerShell):

```powershell
irm https://raw.githubusercontent.com/sealresearch/gust/main/cli/install.ps1 | iex
```

Homebrew:

```sh
brew install sealresearch/tap/seal
```

The scripts download the program for your platform from the latest release, check it against the release's
`SHA256SUMS`, and put it in `~/.local/bin` (`%LOCALAPPDATA%\Programs\seal` on Windows). `SEAL_VERSION` picks a release
and `SEAL_INSTALL` the folder. From source: `cargo install --path cli --locked`.

## Commands

| | |
|---|---|
| `seal install <model>` | Download a model and the program that runs it. Files that are already here and intact are skipped; `--force` downloads again. |
| `seal run <model> [text]` | Ask a model. With no text on a terminal it starts a conversation. |
| `seal bench <model>` | Measure its speed on this machine, at 1, 2, 4 and 8 threads. |
| `seal list` | The models, and which are installed. |
| `seal info <model>` | What a model is, its size and licence, and where its files are. |
| `seal remove <model>` | Delete a model. |
| `seal doctor` | Verify every installed file against its checksum, and run each model once to check that it answers. |

Models can be named by id (`relay-1`) or by alias (`relay`).

### Asking Relay

`seal run` works the same way for every model. This section is about Relay, the first one. Relay turns a request and a list of tools into calls. Without `--tools`, `seal run` offers it a built-in set of phone
tools (alarms, messages, weather, maps, volume, music, flashlight) so you can try it straight away. To use your own,
give a file:

```sh
seal run relay-1 --tools tools.json "I want a large pizza with mushrooms"
```

A tools file is a JSON array of tools, or an object with a `tools` array. Tools can be written the way Relay reads them
(`name`, `description`, `params`) or as JSON Schema tools, the form most APIs use (`parameters`, or `input_schema`, with
or without a `function` wrapper):

```json
[
  {
    "name": "order_pizza",
    "description": "Order a pizza.",
    "parameters": {
      "type": "object",
      "properties": {
        "size": {"type": "string", "enum": ["small", "medium", "large"]},
        "toppings": {"type": "string", "description": "What goes on it."}
      },
      "required": ["size"]
    }
  }
]
```

The words of the request that a value was read from are underlined, and every value shows how it was found (copied from
the request, a number, a date and time, one of the options) and how sure Relay is.

| Option | |
|---|---|
| `--tools FILE` | The tools to offer. |
| `--threads N` | Threads to use. The default is the number of cores, up to 8. Every thread count gives the same answer. |
| `--now TIME` | The date Relay resolves "tomorrow" against, like `2026-10-18T09:00:00`. The default is the time now. |
| `--json` | One line of JSON for a script: the calls, any held back, and the timings. |
| `--lines` | Read standard input as one request per line. With `--json`, the output is one JSON line per request. |
| `-v` | Also show every tool's score for the request. |
| `--no-prefilter` | Read every tool, instead of the best 10. |

Requests can also come from a pipe: `echo "Play some Bowie" | seal run relay-1`.

## How it keeps you safe

- **Weights** are checked against SHA-256 checksums that are compiled into this program, so a download that was changed
  on the way, or at the source, is refused and nothing is installed.
- **The program** is checked against the `SHA256SUMS` file of the release it comes from. That guards against a broken
  download, not against someone who can change the release itself.
- A file is written beside its destination and only moved into place once it has been checked.
- Nothing is sent anywhere when a model runs. The only network traffic is `install`.

## Settings

| Variable | |
|---|---|
| `SEAL_HOME` | Where models and programs are kept. The default is `~/.seal`. |
| `SEAL_HF_URL` | Where to download weights from instead of `https://huggingface.co`. |
| `SEAL_GITHUB_URL` | Where to download programs from instead of `https://github.com`. |
| `NO_COLOR` | Turn colour off. |
| `FORCE_COLOR` | Turn colour on when the output is not a terminal. |

Output is plain text when it is not a terminal. Exit codes: 0 for success, 1 for an error, 2 for a mistake in how the
command was used.

## Build, and try it before anything is published

```sh
cargo build --release     # target/release/seal
cargo test
```

`seal install` downloads from Hugging Face and from a release of this repository. To try everything before either
exists, `dev/mirror.sh` lays those files out in a folder on your machine and writes the settings that point `seal`
at it. Any URL setting also accepts a folder, as `file:///...`.

```sh
ZIG=/path/to/zig ./build.sh release        # the Relay program for every platform
(cd cli && cargo build --release)           # seal itself
cli/dev/mirror.sh                           # the mirror, in ~/.seal-dev

source ~/.seal-dev/env.sh                   # PowerShell: . ~/.seal-dev/env.ps1
seal install relay-1
```

The settings keep the models in `~/.seal-dev/home`, apart from a real `~/.seal`. `cli/install.sh` and `cli/install.ps1`
work against the same mirror, because the settings include `SEAL_GITHUB_URL` and `SEAL_VERSION`.

The Homebrew formula is generated for each release: `cli/scripts/formula.sh TAG SHA256SUMS`. The release workflow
attaches it as `seal.rb`; copy it into the `sealresearch/homebrew-tap` repository as `Formula/seal.rb`.

The list of models and their checksums is `registry.json`. Adding a model means adding an entry there.
