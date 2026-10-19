#!/usr/bin/env python3
"""Check packet integrity, application and Python syntax without importing MLX."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def syntax(node):
    return ast.dump(node, include_attributes=False)


def assignment(tree, name):
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
    )


def function(tree, name):
    return next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name
    )


def verify():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "BASE.json").read_text())
    path = Path(manifest["base_path"])
    base = (root / "base" / path).read_bytes()
    patch_path = root / "change.patch"
    checks = {}

    def check(name, condition):
        checks[name] = bool(condition)
        if not condition:
            raise RuntimeError("Packet verification failed: " + name)

    check("base_sha256", sha256(base) == manifest["base_blob_sha256"])
    check("patch_sha256", sha256(patch_path.read_bytes()) == manifest["patch_sha256"])
    check("base_mit_notice", base.startswith(b"# SPDX-License-Identifier: MIT\n"))

    with tempfile.TemporaryDirectory(prefix="flash1017-bf16-patch-") as directory:
        stage = Path(directory)
        target = stage / path
        target.parent.mkdir(parents=True)
        target.write_bytes(base)
        subprocess.run(
            ["git", "apply", "--check", str(patch_path)],
            cwd=stage, check=True, capture_output=True, text=True,
        )
        checks["git_apply_check"] = True
        subprocess.run(
            ["git", "apply", str(patch_path)],
            cwd=stage, check=True, capture_output=True, text=True,
        )
        proposed = target.read_bytes()
        check("only_expected_file", [p.relative_to(stage) for p in stage.rglob("*") if p.is_file()] == [path])
        check("proposed_sha256", sha256(proposed) == manifest["proposed_blob_sha256"])
        check("proposed_mit_notice", proposed.startswith(b"# SPDX-License-Identifier: MIT\n"))

        # compile() creates Python code objects; it executes no module or import.
        compile(base, str(path), "exec")
        compile(proposed, str(path), "exec")
        checks["python_syntax_compiles_without_execution"] = True
        before, after = ast.parse(base), ast.parse(proposed)
        before_imports = [syntax(n) for n in before.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        after_imports = [syntax(n) for n in after.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        check("imports_unchanged", before_imports == after_imports)
        check("original_metal_source_unchanged", syntax(assignment(before, "_SOURCE")) == syntax(assignment(after, "_SOURCE")))
        check("original_kernel_factory_unchanged", syntax(function(before, "_kernel")) == syntax(function(after, "_kernel")))

        old_function, new_function = function(before, "project_logits"), function(after, "project_logits")
        check("original_guard_and_setup_unchanged", [syntax(n) for n in old_function.body[:4]] == [syntax(n) for n in new_function.body[:4]])
        check("original_single_row_return_unchanged", syntax(old_function.body[-1]) == syntax(new_function.body[-1]))
        expected_assignment = ast.parse("queries = x.size // width").body[0]
        expected_condition = ast.parse("queries > 1", mode="eval").body
        check("new_branch_after_existing_one_to_five_guard", len(new_function.body) == len(old_function.body) + 2 and syntax(new_function.body[4]) == syntax(expected_assignment) and isinstance(new_function.body[5], ast.If) and syntax(new_function.body[5].test) == syntax(expected_condition) and not new_function.body[5].orelse)

        rows_source = ast.literal_eval(assignment(after, "_ROWS_SOURCE").value)
        kernel_manifest = manifest["reused_kernel"]
        check("metal_body_matches_frozen_local_kernel_hash", sha256(rows_source.encode()) == kernel_manifest["metal_source_sha256"])
        local_source_path = root / kernel_manifest["source_path"]
        local_source = local_source_path.read_bytes()
        check("packaged_local_source_sha256", sha256(local_source) == kernel_manifest["source_file_sha256"])
        local_kernel = ast.literal_eval(assignment(ast.parse(local_source), kernel_manifest["original_assignment"]).value)
        check("metal_body_identical_to_packaged_local_kernel", rows_source == local_kernel)

    return {
        "schema": "flash1017-bf16-rows2-5-cpu-verification/1",
        "status": "passed_cpu_packet_checks_only",
        "base_commit": manifest["base_commit"],
        "checks": checks,
        "patch_sha256": manifest["patch_sha256"],
        "proposed_blob_sha256": manifest["proposed_blob_sha256"],
        "not_run": [
            "MLX import or production-function execution",
            "Metal compilation or GPU kernel execution",
            "upstream focused or default test suite",
            "current-main real-model numerical or performance validation",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional receipt path")
    arguments = parser.parse_args()
    rendered = json.dumps(verify(), indent=2) + "\n"
    if arguments.output:
        arguments.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
