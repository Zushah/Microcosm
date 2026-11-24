from __future__ import annotations
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import NoReturn

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = str(ROOT / "rust" / "Cargo.toml")
CHECKS = (
    ("Build all", ["build", "--manifest-path", MANIFEST, "--all-targets"]),
    ("Build wasm32", ["build", "--manifest-path", MANIFEST, "--target", "wasm32-unknown-unknown"]),
    ("Format", ["fmt", "--manifest-path", MANIFEST, "--all", "--", "--check"]),
    ("Lint", ["clippy", "--manifest-path", MANIFEST, "--all-targets", "--", "-D", "warnings"]),
    ("Test", ["test", "--manifest-path", MANIFEST, "--", "--test-threads=1"])
)
IN_GITHUB_ACTIONS = os.environ.get("GITHUB_ACTIONS") == "true"

def fail(message: str) -> NoReturn:
    print(f"test.py: error: {message}", file=sys.stderr)
    sys.exit(1)

def run_check(cargo: str, name: str, args: list[str]) -> tuple[bool, float]:
    title = f"{name}: cargo {' '.join(args)}"
    print(f"::group::{title}" if IN_GITHUB_ACTIONS else f"==> {title}", flush=True)
    start = time.monotonic()
    try:
        code = subprocess.run([cargo, *args], cwd=ROOT).returncode
    except OSError as error:
        fail(f"could not run cargo: {error}")
    elapsed = time.monotonic() - start
    if IN_GITHUB_ACTIONS:
        print("::endgroup::", flush=True)
    return code == 0, elapsed

def main() -> None:
    cargo = shutil.which("cargo")
    if cargo is None:
        fail("cargo was not found on PATH; install Rust with rustup (https://rustup.rs)")
    results = [(name, *run_check(cargo, name, args)) for name, args in CHECKS]
    print("Summary:")
    for name, passed, seconds in results:
        print(f" - {name}: {'passed' if passed else 'FAILED'} ({seconds:.1f} s)")
    failed = [name for name, passed, _ in results if not passed]
    if failed:
        if IN_GITHUB_ACTIONS:
            print(f"::error::Failed: {', '.join(failed)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
