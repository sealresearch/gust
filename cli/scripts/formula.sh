#!/usr/bin/env bash
# Print the Homebrew formula for a release of seal.
#
#   cli/scripts/formula.sh v0.1.0 path/to/SHA256SUMS > Formula/seal.rb
#
# SHA256SUMS is the file attached to the release. The formula goes in the tap repository, sealresearch/homebrew-tap,
# and people install it with: brew install sealresearch/tap/seal
set -euo pipefail

tag=${1:?usage: formula.sh TAG SHA256SUMS}
sums=${2:?usage: formula.sh TAG SHA256SUMS}
version=${tag#v}

sum() { # asset
  local s
  s=$(grep -E " \*?$1\$" "$sums" | cut -d' ' -f1 | head -n 1)
  [[ -n $s ]] || { echo "no checksum for $1 in $sums" >&2; exit 1; }
  echo "$s"
}

base="https://github.com/sealresearch/gust/releases/download/$tag"
mac_arm=$(sum seal-macos-arm64)
mac_x64=$(sum seal-macos-x64)
linux_arm=$(sum seal-linux-arm64)
linux_x64=$(sum seal-linux-x64)

cat <<EOF
class Seal < Formula
  desc "Install and run Seal's on-device models"
  homepage "https://seal.computer"
  version "$version"
  license "MIT"

  on_macos do
    on_arm do
      url "$base/seal-macos-arm64"
      sha256 "$mac_arm"
    end
    on_intel do
      url "$base/seal-macos-x64"
      sha256 "$mac_x64"
    end
  end

  on_linux do
    on_arm do
      url "$base/seal-linux-arm64"
      sha256 "$linux_arm"
    end
    on_intel do
      url "$base/seal-linux-x64"
      sha256 "$linux_x64"
    end
  end

  def install
    bin.install Dir["seal-*"].first => "seal"
  end

  test do
    assert_match "seal #{version}", shell_output("#{bin}/seal --version")
  end
end
EOF
