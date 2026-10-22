#!/usr/bin/env bash
# Print the Homebrew formula for a release of sealpup.
#
#   cli/scripts/formula.sh v0.1.0 path/to/SHA256SUMS > Formula/sealpup.rb
#
# SHA256SUMS is the file attached to the release. The formula goes in the tap repository, sealresearch/homebrew-tap,
# and people install it with: brew install sealresearch/tap/sealpup
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
mac_arm=$(sum sealpup-macos-arm64)
mac_x64=$(sum sealpup-macos-x64)
linux_arm=$(sum sealpup-linux-arm64)
linux_x64=$(sum sealpup-linux-x64)

cat <<EOF
class Seal < Formula
  desc "Install and run Seal's on-device models"
  homepage "https://seal.computer"
  version "$version"
  license "MIT"

  on_macos do
    on_arm do
      url "$base/sealpup-macos-arm64"
      sha256 "$mac_arm"
    end
    on_intel do
      url "$base/sealpup-macos-x64"
      sha256 "$mac_x64"
    end
  end

  on_linux do
    on_arm do
      url "$base/sealpup-linux-arm64"
      sha256 "$linux_arm"
    end
    on_intel do
      url "$base/sealpup-linux-x64"
      sha256 "$linux_x64"
    end
  end

  def install
    bin.install Dir["sealpup-*"].first => "sealpup"
  end

  test do
    assert_match "sealpup #{version}", shell_output("#{bin}/sealpup --version")
  end
end
EOF
