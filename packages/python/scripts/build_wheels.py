"""Builds one wheel per platform from the shared libraries that `build.sh lib` puts in out/lib/<platform>/.

    ZIG=zig bash build.sh lib
    python packages/python/scripts/build_wheels.py [platform ...]     # wheels go to packages/python/dist/

A platform's wheel holds that platform's library under sealresearch/_native/ and the registry (cli/registry.json)."""
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(PKG))
# repository folder -> (library file, wheel platform tag)
PLATFORMS = {
    "windows-x64": ("gust.dll", "win_amd64"),
    "linux-x64": ("libgust.so", "manylinux2014_x86_64"),
    "linux-arm64": ("libgust.so", "manylinux2014_aarch64"),
    "macos-x64": ("libgust.dylib", "macosx_10_13_x86_64"),
    "macos-arm64": ("libgust.dylib", "macosx_11_0_arm64"),
}


def build(platform, dist):
    lib, tag = PLATFORMS[platform]
    src = os.path.join(REPO, "out", "lib", platform, lib)
    if not os.path.exists(src):
        raise SystemExit(f"missing {src}; build it with: ZIG=zig bash build.sh lib")
    with tempfile.TemporaryDirectory() as tmp:
        work = os.path.join(tmp, "pkg")
        shutil.copytree(PKG, work, ignore=shutil.ignore_patterns("dist", "build", "*.egg-info", "__pycache__", "_native", "tests", "scripts"))
        os.makedirs(os.path.join(work, "sealresearch", "_native"))
        shutil.copy(src, os.path.join(work, "sealresearch", "_native", lib))
        shutil.copy(os.path.join(REPO, "cli", "registry.json"), os.path.join(work, "sealresearch", "_registry.json"))
        shutil.copy(os.path.join(REPO, "LICENSE"), os.path.join(work, "LICENSE"))
        subprocess.run([sys.executable, "setup.py", "-q", "bdist_wheel", "--plat-name", tag, "--dist-dir", dist], cwd=work, check=True)


if __name__ == "__main__":
    dist = os.path.join(PKG, "dist")
    os.makedirs(dist, exist_ok=True)
    for platform in sys.argv[1:] or PLATFORMS:
        build(platform, dist)
        print("built", platform)
    for name in sorted(os.listdir(dist)):
        print(" ", name, os.path.getsize(os.path.join(dist, name)) // 1024, "KB")
