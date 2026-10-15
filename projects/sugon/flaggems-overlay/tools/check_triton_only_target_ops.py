#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reject Native target-compute calls from Triton-only operator sources."""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Sequence


_FORBIDDEN_TERMINALS = frozenset({"redispatch", "get_kernel", "call_boxed"})
_TARGET_COMPUTE_NAMES = frozenset(
    {
        "add",
        "addmm",
        "dgemm",
        "gemm",
        "linear",
        "matmul",
        "mm",
        "npu_linear",
        "sgemm",
    }
)
_DIRECT_TORCH_COMPUTE = frozenset(
    {
        "torch.add",
        "torch.addmm",
        "torch.matmul",
        "torch.mm",
    }
)
_VENDOR_ROOTS = frozenset(
    {
        "cann",
        "cublas",
        "cuda",
        "hcblas",
        "hip",
        "hipblas",
        "mudnn",
        "musa",
        "rocblas",
        "torch_musa",
        "torch_npu",
    }
)


@dataclass(frozen=True)
class Violation:
    path: Path
    line: int
    symbol: str


def _dotted_name(node: ast.AST, aliases: dict[str, str]) -> str | None:
    if isinstance(node, ast.Name):
        return aliases.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        owner = _dotted_name(node.value, aliases)
        return f"{owner}.{node.attr}" if owner else node.attr
    return None


def _is_forbidden_call(symbol: str) -> bool:
    parts = symbol.split(".")
    if parts[-1] in _FORBIDDEN_TERMINALS:
        return True
    if symbol in _DIRECT_TORCH_COMPUTE:
        return True

    lowered = [part.lower() for part in parts]
    if len(parts) >= 4 and parts[:2] == ["torch", "ops"]:
        return bool(_TARGET_COMPUTE_NAMES.intersection(lowered[2:]))
    return parts[0] in _VENDOR_ROOTS and bool(
        _TARGET_COMPUTE_NAMES.intersection(lowered[1:])
    )


class _CallAudit(ast.NodeVisitor):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.aliases: dict[str, str] = {}
        self.violations: list[Violation] = []

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        for imported in node.names:
            bound_name = imported.asname or imported.name.split(".", 1)[0]
            self.aliases[bound_name] = imported.name
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        if node.module is not None:
            for imported in node.names:
                if imported.name == "*":
                    continue
                bound_name = imported.asname or imported.name
                self.aliases[bound_name] = f"{node.module}.{imported.name}"
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        symbol = _dotted_name(node.func, self.aliases)
        if symbol is not None and _is_forbidden_call(symbol):
            self.violations.append(
                Violation(path=self.path, line=node.lineno, symbol=symbol)
            )
        self.generic_visit(node)


def _python_files(paths: Sequence[Path]) -> list[Path]:
    files: set[Path] = set()
    for path in paths:
        if path.is_dir():
            files.update(candidate for candidate in path.rglob("*.py") if candidate.is_file())
        else:
            files.add(path)
    return sorted(files, key=lambda path: str(path))


def audit_paths(paths: Sequence[Path]) -> list[Violation]:
    """Return every forbidden call found below the supplied Python paths."""

    violations: list[Violation] = []
    for path in _python_files(paths):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        audit = _CallAudit(path)
        audit.visit(tree)
        violations.extend(audit.violations)
    return sorted(
        violations,
        key=lambda item: (str(item.path), item.line, item.symbol),
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", type=Path, nargs="+")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        missing = [path for path in args.paths if not path.exists()]
        if missing:
            raise FileNotFoundError(", ".join(str(path) for path in missing))
        violations = audit_paths(args.paths)
    except (OSError, SyntaxError) as error:
        print(f"audit input error: {error}", file=sys.stderr)
        return 2

    for violation in violations:
        print(f"{violation.path}:{violation.line}: forbidden call {violation.symbol}")
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
