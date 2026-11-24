from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "rust" / "Cargo.toml"
COMMITTED_WASM = ROOT / "rust" / "microcosm.wasm"
TARGET = "wasm32-unknown-unknown"
PACKAGE = "microcosm-wasm"
PROFILE = "wasm"
ARTIFACT = "microcosmwasm"
RUSTFLAGS_ENV = ("RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "CARGO_BUILD_RUSTFLAGS", "CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUSTFLAGS")

def fail(message: str) -> NoReturn:
    print(f"build.py: error: {message}", file=sys.stderr)
    sys.exit(1)

def run(args: list[str]) -> str:
    try:
        result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
    except OSError as error:
        fail(f"could not run {args[0]}: {error}")
    if result.returncode != 0:
        fail(f"{' '.join(args)} exited with {result.returncode}: {result.stderr.strip()}")
    return result.stdout

def find_tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        fail(f"{name} was not found on PATH; install Rust with rustup (https://rustup.rs)")
    return path

def check_environment() -> None:
    for name in RUSTFLAGS_ENV:
        if name in os.environ:
            fail(f"{name} is set; it would replace the rustflags in .cargo/config.toml, so unset it and run again")

def rustc_info(rustc: str) -> tuple[str, str, Path]:
    verbose = run([rustc, "-vV"]).splitlines()
    fields = dict(line.split(": ", 1) for line in verbose[1:] if ": " in line)
    commit = fields.get("commit-hash", "unknown")
    if commit == "unknown":
        fail("rustc -vV reports no commit-hash, so the standard library paths cannot be remapped")
    sysroot = Path(run([rustc, "--print", "sysroot"]).strip())
    return verbose[0], commit, sysroot

def remap_flags(commit: str, sysroot: Path) -> list[str]:
    cargo_home = Path(os.environ.get("CARGO_HOME") or Path.home() / ".cargo").resolve()
    remaps = (
        (ROOT, "."),
        (sysroot / "lib" / "rustlib" / "src" / "rust", f"/rustc/{commit}"),
        (cargo_home / "registry" / "src", "/cargo/registry/src"),
        (cargo_home / "git" / "checkouts", "/cargo/git/checkouts")
    )
    return [f"--remap-path-prefix={source}={dest}" for source, dest in remaps]

def build(cargo: str, rustflags: list[str]) -> Path:
    config = f"target.{TARGET}.rustflags=[{', '.join(json.dumps(flag) for flag in rustflags)}]"
    args = [
        cargo, "build", "--manifest-path", str(MANIFEST), "-p", PACKAGE, "--profile", PROFILE, "--target", TARGET,
        "--locked", "--message-format=json-render-diagnostics", "--config", config
    ]
    artifact = None
    try:
        process = subprocess.Popen(args, cwd=ROOT, stdout=subprocess.PIPE, text=True)
    except OSError as error:
        fail(f"could not run cargo: {error}")
    assert process.stdout is not None
    for line in process.stdout:
        if not line.startswith("{"):
            print(line, end="")
            continue
        message = json.loads(line)
        if message.get("reason") == "compiler-artifact" and message["target"]["name"] == ARTIFACT:
            artifact = next((Path(name) for name in message["filenames"] if name.endswith(".wasm")), artifact)
    if process.wait() != 0:
        fail(f"cargo build exited with {process.returncode}")
    if artifact is None:
        fail(f"cargo reported no .wasm artifact for {ARTIFACT}")
    return artifact

def summarize(label: str, path: Path, data: bytes, rustc_version: str) -> None:
    print(f"{label}: {path}")
    print(f" - raw: {len(data):,} bytes")
    print(f" - gzip -9: {len(gzip.compress(data, 9, mtime=0)):,} bytes")
    print(f" - sha256: {hashlib.sha256(data).hexdigest()}")
    print(f" - {rustc_version}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Build rust/microcosm.wasm from the Rust sources.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="build and confirm that rust/microcosm.wasm matches, without writing anything")
    mode.add_argument("--out", type=Path, metavar="PATH", help="write the binary to PATH instead of rust/microcosm.wasm")
    args = parser.parse_args()
    check_environment()
    cargo = find_tool("cargo")
    rustc = find_tool("rustc")
    rustc_version, commit, sysroot = rustc_info(rustc)
    artifact = build(cargo, remap_flags(commit, sysroot))
    data = artifact.read_bytes()
    if args.check:
        committed = COMMITTED_WASM.read_bytes() if COMMITTED_WASM.is_file() else b""
        summarize("Built", artifact, data, rustc_version)
        print(f"Compared with: {COMMITTED_WASM}")
        if committed != data:
            print("The committed binary is out of date:")
            print(f" - committed sha256: {hashlib.sha256(committed).hexdigest()}")
            print(f" - built sha256: {hashlib.sha256(data).hexdigest()}")
            print("Run python scripts/build.py to rebuild it.")
            sys.exit(1)
        print("The committed binary is up to date.")
        return
    out = args.out.resolve() if args.out else COMMITTED_WASM
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    summarize("Wrote", out, data, rustc_version)

if __name__ == "__main__":
    main()
