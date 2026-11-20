"""Dependency rules from README.md: no module imports from a later layer."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RANK = {"config": 0, "utils": 0, "data": 1, "features": 2, "qc": 3, "core": 4, "robustness": 5,
        "ml": 6, "advanced": 7, "plots": 8}
ALLOWED_ONLY = {"utils": {"config", "utils"}, "data": {"config", "data"},
                "features": {"config", "data", "features", "utils"}}


def project_imports(path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            yield node.module.split(".")[0]
        elif isinstance(node, ast.Import):
            for a in node.names:
                yield a.name.split(".")[0]


def test_no_upward_imports():
    problems = []
    for pkg, rank in RANK.items():
        folder = ROOT / pkg
        if not folder.is_dir():
            continue
        for f in folder.glob("*.py"):
            for imp in project_imports(f):
                if imp not in RANK:
                    continue
                if pkg in ALLOWED_ONLY and imp not in ALLOWED_ONLY[pkg]:
                    problems.append(f"{f.relative_to(ROOT)} imports {imp}")
                elif RANK[imp] > rank:
                    problems.append(f"{f.relative_to(ROOT)} imports later layer {imp}")
    assert not problems, "\n".join(problems)


def test_every_module_is_registered():
    import run_all
    registered = {run_all.short(m) for _, m, _ in run_all.registry()}
    analysis_modules = {f.stem for pkg in ["qc", "core", "robustness", "ml", "advanced", "plots"]
                        for f in (ROOT / pkg).glob("*.py") if f.stem != "__init__"}
    helpers = {"nested_cv", "style"}
    assert analysis_modules - helpers == registered
