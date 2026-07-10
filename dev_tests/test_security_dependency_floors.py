"""Regression checks for dependency versions that close audited advisories."""

from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parents[1]


def _requirements(path: Path) -> dict[str, Requirement]:
    parsed: dict[str, Requirement] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        requirement = Requirement(line)
        parsed[requirement.name.lower()] = requirement
    return parsed


def _assert_minimum(requirement: Requirement, minimum: str) -> None:
    lower_bounds = [
        Version(specifier.version)
        for specifier in requirement.specifier
        if specifier.operator in {">=", ">", "==", "~="}
    ]
    assert lower_bounds, f"{requirement.name} must declare a lower bound"
    assert max(lower_bounds) >= Version(minimum)


def test_engine_security_dependency_floors() -> None:
    requirements = _requirements(REPO_ROOT / "requirements.txt")

    for package, minimum in {
        "fastapi": "0.136.3",
        "starlette": "1.3.1",
        "python-multipart": "0.0.32",
        "aiohttp": "3.14.1",
        "cryptography": "48.0.1",
    }.items():
        _assert_minimum(requirements[package], minimum)


def test_saas_parser_security_dependency_floors() -> None:
    requirements = _requirements(REPO_ROOT / "saas" / "requirements.txt")

    _assert_minimum(requirements["fastapi"], "0.136.3")
    _assert_minimum(requirements["starlette"], "1.3.1")
    _assert_minimum(requirements["python-multipart"], "0.0.32")
