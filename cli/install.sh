#!/bin/sh
# Install the sealpup command line.
#
#   curl -fsSL https://raw.githubusercontent.com/sealresearch/gust/main/cli/install.sh | sh
#
# Settings, as environment variables:
#   SEALPUP_VERSION      a release tag such as v0.1.0 (default: the latest release)
#   SEALPUP_INSTALL      where to put the program (default: ~/.local/bin)
#   SEALPUP_GITHUB_URL   download from a mirror instead of https://github.com
set -eu

repo="sealresearch/gust"
base="${SEALPUP_GITHUB_URL:-https://github.com}"
dir="${SEALPUP_INSTALL:-$HOME/.local/bin}"

say() { printf '  %s\n' "$*"; }
die() { printf '\n  ✗ %s\n\n' "$*" >&2; exit 1; }

case "$(uname -s)" in
  Linux) os=linux ;;
  Darwin) os=macos ;;
  MINGW* | MSYS* | CYGWIN*) os=windows ;;
  *) die "sealpup does not have a build for $(uname -s)" ;;
esac
case "$(uname -m)" in
  x86_64 | amd64) arch=x64 ;;
  arm64 | aarch64) arch=arm64 ;;
  *) die "sealpup does not have a build for $(uname -m)" ;;
esac
asset="sealpup-$os-$arch"
exe=sealpup
if [ "$os" = windows ]; then
  [ "$arch" = x64 ] || die "on Windows only the x64 build exists"
  asset="$asset.exe"
  exe=sealpup.exe
fi

if [ -n "${SEALPUP_VERSION:-}" ]; then
  release="$base/$repo/releases/download/$SEALPUP_VERSION"
else
  release="$base/$repo/releases/latest/download"
fi

fetch() { # url file
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "$1" -o "$2" || die "cannot download $1"
  elif command -v wget >/dev/null 2>&1; then
    wget -qO "$2" "$1" || die "cannot download $1"
  else
    die "I need curl or wget to download sealpup"
  fi
}

sha256() { # file
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | cut -d' ' -f1
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | cut -d' ' -f1
  else
    die "I need sha256sum or shasum to check the download"
  fi
}

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

printf '\n'
say "Installing sealpup ($os $arch)"
fetch "$release/SHA256SUMS" "$tmp/SHA256SUMS"
fetch "$release/$asset" "$tmp/$asset"

expected=$(grep " \*\{0,1\}$asset\$" "$tmp/SHA256SUMS" | cut -d' ' -f1 | head -n 1)
[ -n "$expected" ] || die "the release has no checksum for $asset"
[ "$(sha256 "$tmp/$asset")" = "$expected" ] || die "the download does not match its checksum, so it was not installed"

mkdir -p "$dir"
mv "$tmp/$asset" "$dir/$exe"
chmod +x "$dir/$exe"

version=$("$dir/$exe" --version 2>/dev/null || true)
printf '\n  ✓ %s  %s\n' "${version:-sealpup}" "$dir/$exe"
case ":$PATH:" in
  *":$dir:"*) ;;
  *) printf '\n  %s is not on your PATH. Add it with:\n\n    export PATH="%s:$PATH"\n' "$dir" "$dir" ;;
esac
printf '\n  Next: sealpup install relay-1\n\n'
