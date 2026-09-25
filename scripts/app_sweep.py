"""Pre-push gate: compile + AppTest every Streamlit entry point.

    .venv/bin/python scripts/app_sweep.py

Two independent checks per file, because `AppTest.from_file` alone is NOT a
sufficient gate: a page with a SyntaxError prints a traceback to stderr but
leaves `AppTest.exception` empty, so it would otherwise "pass".

1. `py_compile` — catches syntax errors (the false-pass case).
2. `AppTest.from_file(path, default_timeout=180).run()` — catches runtime
   exceptions raised while rendering the page.

Exit code 0 only when every file is clean.
"""
from __future__ import annotations

import py_compile
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest  # noqa: E402


def targets() -> list[Path]:
    files = [ROOT / "app.py"]
    files += sorted((ROOT / "pages").glob("*.py"))
    return [f for f in files if f.exists()]


def compile_check(path: Path) -> str | None:
    try:
        with tempfile.NamedTemporaryFile(suffix=".pyc", delete=True) as tmp:
            py_compile.compile(str(path), cfile=tmp.name, doraise=True)
        return None
    except py_compile.PyCompileError as exc:
        return f"{type(exc).__name__}: {exc}"


def run_check(path: Path) -> tuple[bool, str]:
    try:
        at = AppTest.from_file(str(path), default_timeout=180)
        at.run()
    except Exception as exc:                       # noqa: BLE001 - report anything
        return False, f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=4)}"
    if at.exception:
        msgs = []
        for e in at.exception:
            detail = getattr(e, "stack_trace", "") or str(e)
            msgs.append(f"{e.value}\n{detail[-900:]}")
        return False, "\n---\n".join(msgs)
    return True, ""


def main() -> int:
    files = targets()
    failures = 0
    print(f"AppTest sweep · {len(files)} Streamlit entry point(s)\n")
    for path in files:
        name = path.relative_to(ROOT).as_posix()
        err = compile_check(path)
        if err:
            failures += 1
            print(f"FAIL (syntax) {name}\n    {err.splitlines()[-1]}")
            continue
        ok, msg = run_check(path)
        if ok:
            print(f"ok           {name}")
        else:
            failures += 1
            print(f"FAIL (run)   {name}")
            for line in msg.splitlines():
                print(f"    {line}")
    clean = len(files) - failures
    print(f"\n{clean}/{len(files)} clean")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
