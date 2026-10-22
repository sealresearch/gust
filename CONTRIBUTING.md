# Contributing

Issues and pull requests are welcome. For a change to the runtime or to a model's host, please open an issue first so we
can agree on the approach. To add a model, see [models/README.md](models/README.md).

## Build and test

You need [zig](https://ziglang.org) (tested with 0.14.1), which is used as the C compiler for every target, and Node 22 or
newer for the JavaScript parts.

```sh
export ZIG=/path/to/zig
bash build.sh x64 wasm test-t5 test-temporal    # build into out/
out/t5.exe && out/temporal.exe                   # runtime tests
bash models/relay/tests/shared_arena.sh relay-model
```

`relay-model/` holds the Relay 1 files from [Hugging Face](https://huggingface.co/sealresearch/relay). The tests in
`runtime/tests` need no model files.

The `seal` command line is a Rust program in `cli/`: `cd cli && cargo test`. CI also runs `cargo fmt --check` and
`cargo clippy -- -D warnings`.

## One rule: outputs are bit-exact

Gust returns the same bits on every target and for any thread count. A change to a kernel, an operator or the host is
accepted only if every build still reproduces the reference output. In practice:

- Compile with `-ffp-contract=off`. Float expressions must round exactly as written.
- Integer kernels must be exact; a vector kernel must match the scalar one in `runtime/src/int8/kernels.c`.
- Run `runtime/tests/bitexact.sh` against the reference predictions before and after your change.

## Style

C is formatted with `clang-format` (`.clang-format`), JavaScript with Prettier (`.prettierrc`), Python with ruff
(`ruff.toml`). `.editorconfig` covers the rest.
