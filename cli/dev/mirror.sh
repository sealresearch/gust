#!/usr/bin/env bash
# Try sealpup before anything is published: lay out the files that Hugging Face and the GitHub release will serve in a
# folder on this machine, and write the settings that make sealpup read from it.
#
#   ZIG=/path/to/zig ./build.sh release        # the Relay program for every platform, in out/release/
#   (cd cli && cargo build --release)           # sealpup itself
#   cli/dev/mirror.sh [WEIGHTS_DIR [DEV_DIR]]
#
# WEIGHTS_DIR holds the Relay files from the Hugging Face repository (default: ../relay next to this repository).
# DEV_DIR is where the mirror, a separate SEALPUP_HOME and the settings go (default: ~/.sealpup-dev).
#
# Then, in a new shell:
#   source ~/.sealpup-dev/env.sh          # bash, zsh
#   . ~/.sealpup-dev/env.ps1              # PowerShell
#   sealpup install relay-1
set -euo pipefail

root=$(cd "$(dirname "$0")/../.." && pwd)
weights=${1:-$root/../relay}
dev=${2:-$HOME/.sealpup-dev}
# The release that holds sealpup and its library (the engine's tag), and the older one that holds the models' programs.
tag=$(sed -n '/"engines"/,/"runners"/ s/.*"tag": *"\([^"]*\)".*/\1/p' "$root/cli/registry.json" | head -n 1)
runner_tag=$(sed -n '/"runners"/,$ s/.*"tag": *"\([^"]*\)".*/\1/p' "$root/cli/registry.json" | head -n 1)
hf_repo=$(sed -n 's/.*"repo": *"\(sealresearch\/relay\)".*/\1/p' "$root/cli/registry.json" | head -n 1)

[[ -f $weights/relay.gust ]] || { echo "no Relay weights in $weights (pass the folder as the first argument)" >&2; exit 1; }
[[ -f $root/out/release/relay-linux-x64 ]] || { echo "no programs in $root/out/release; run: ZIG=... ./build.sh release" >&2; exit 1; }

hf="$dev/mirror/hf/$hf_repo/resolve/main"
gh="$dev/mirror/gh/sealresearch/gust/releases/download/$tag"
gh_runner="$dev/mirror/gh/sealresearch/gust/releases/download/$runner_tag"
rm -rf "$dev/mirror"
mkdir -p "$hf" "$gh" "$gh_runner"
cp "$weights"/relay* "$hf/"
cp "$root"/out/release/relay-* "$gh/"
cp "$root"/out/release/relay-* "$gh_runner/"
# the Gust library for every platform, if it was built (ZIG=... ./build.sh lib-release)
cp "$root"/out/release/gust-* "$gh/" 2>/dev/null || true
(cd "$gh_runner" && rm -f SHA256SUMS && sha256sum -- * > SHA256SUMS)

# sealpup itself, named for the platform it was built on, so cli/install.sh and cli/install.ps1 can be tried too.
case "$(uname -s)" in
  Linux) os=linux ;; Darwin) os=macos ;; *) os=windows ;;
esac
case "$(uname -m)" in
  arm64 | aarch64) arch=arm64 ;; *) arch=x64 ;;
esac
built="$root/cli/target/release/sealpup"
[[ -f $built.exe ]] && built="$built.exe"
if [[ -f $built ]]; then
  name="sealpup-$os-$arch"
  [[ $os == windows ]] && name="$name.exe"
  cp "$built" "$gh/$name"
fi
(cd "$gh" && rm -f SHA256SUMS && sha256sum -- * > SHA256SUMS)

# A path in the form the Rust program wants: C:/... on Windows, /... elsewhere.
here() { (cd "$1" && { pwd -W 2>/dev/null || pwd; }); }
mirror=$(here "$dev/mirror")
home=$(here "$dev")/home

cat > "$dev/env.sh" <<EOF
export SEALPUP_HOME="$home"
export SEALPUP_HF_URL="file://$mirror/hf"
export SEALPUP_GITHUB_URL="file://$mirror/gh"
export SEALPUP_VERSION="$tag"
echo "sealpup now reads from the local mirror ($mirror) and keeps its files in $home"
EOF
cat > "$dev/env.ps1" <<EOF
\$env:SEALPUP_HOME = "$home"
\$env:SEALPUP_HF_URL = "file://$mirror/hf"
\$env:SEALPUP_GITHUB_URL = "file://$mirror/gh"
\$env:SEALPUP_VERSION = "$tag"
Write-Host "sealpup now reads from the local mirror ($mirror) and keeps its files in $home"
EOF

echo "mirror ready in $dev"
echo "  source $dev/env.sh        (bash)"
echo "  . $dev/env.ps1            (PowerShell)"
echo "then: sealpup install relay-1"
