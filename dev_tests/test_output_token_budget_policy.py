import ast
import inspect
from pathlib import Path

from ai_service import AIService
from config import DEFAULT_OUTPUT_TOKEN_FALLBACK, config


def _set_model_specs(monkeypatch, *, output_tokens: int = 20000, safety_margin: float = 0.5) -> None:
    monkeypatch.setattr(
        config,
        "spec_catalog",
        {
            "model_specifications": {
                "fake": {
                    "budget-model": {
                        "model_id": "budget-model",
                        "output_tokens": output_tokens,
                    }
                }
            },
            "token_validation": {
                "default_max_output": 8192,
                "fallback_limits": {"output": 4096},
                "safety_margin": safety_margin,
            },
        },
    )


def test_output_budget_resolver_priority_and_cap(monkeypatch):
    _set_model_specs(monkeypatch, output_tokens=20000, safety_margin=0.5)

    requested = config.resolve_output_max_tokens(
        "budget-model",
        requested_max_tokens=7000,
        routed_max_tokens=6000,
        configured_default=5000,
        call_id="test.requested",
    )
    assert requested["max_tokens"] == 7000
    assert requested["source"] == "requested"

    routed = config.resolve_output_max_tokens(
        "budget-model",
        routed_max_tokens=6000,
        configured_default=5000,
        call_id="test.routing",
    )
    assert routed["max_tokens"] == 6000
    assert routed["source"] == "routing"

    configured = config.resolve_output_max_tokens(
        "budget-model",
        configured_default=5000,
        call_id="test.configured",
    )
    assert configured["max_tokens"] == 5000
    assert configured["source"] == "configured_default"

    model_limit = config.resolve_output_max_tokens("budget-model", call_id="test.model")
    assert model_limit["max_tokens"] == 10000
    assert model_limit["source"] == "model_safe_limit"

    capped = config.resolve_output_max_tokens(
        "budget-model",
        requested_max_tokens=30000,
        call_id="test.cap",
    )
    assert capped["max_tokens"] == 10000
    assert capped["was_adjusted"] is True
    assert capped["reason"] == "capped_to_model_safe_limit"


def test_output_budget_resolver_percentage_and_technical_fallback(monkeypatch):
    _set_model_specs(monkeypatch, output_tokens=20000, safety_margin=0.5)

    percentage = config.resolve_output_max_tokens(
        "budget-model",
        max_tokens_percentage=50,
        call_id="test.percentage",
    )
    assert percentage["max_tokens"] == 5000
    assert percentage["source"] == "percentage"
    assert percentage["percentage_used"] is True

    fallback = config.resolve_output_max_tokens(
        "unknown-budget-model",
        call_id="test.fallback",
    )
    assert fallback["max_tokens"] == DEFAULT_OUTPUT_TOKEN_FALLBACK
    assert fallback["source"] == "fallback"


def test_ai_service_public_generation_methods_do_not_default_to_8000():
    for method_name in (
        "generate_content",
        "generate_content_stream",
        "call_ai_with_validation_tools",
    ):
        parameter = inspect.signature(getattr(AIService, method_name)).parameters["max_tokens"]
        assert parameter.default is None


def _parent_functions(tree: ast.AST) -> dict[ast.AST, str]:
    parent_by_child: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parent_by_child[child] = parent

    functions: dict[ast.AST, str] = {}
    for node in ast.walk(tree):
        parent = parent_by_child.get(node)
        while parent is not None:
            if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions[node] = parent.name
                break
            parent = parent_by_child.get(parent)
    return functions


def test_no_unapproved_local_max_tokens_numeric_fallbacks():
    root = Path(__file__).resolve().parents[1]
    excluded = {
        ".git",
        ".pytest_cache",
        "__pycache__",
        "dev_tests",
        "docs",
        "demos",
        "templates",
        "saas",
        "gransabio_reimagined",
    }
    allowed_get_defaults = {
        ("config.py", "_validate_thinking_budget", 16384),
        ("ai_service.py", "health_check", 5),
        (str(Path("evidence_grounding") / "budget_scorer.py"), "score_claims", 5),
    }

    findings: list[str] = []
    for path in root.rglob("*.py"):
        relative = path.relative_to(root)
        if any(part in excluded for part in relative.parts):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(relative))
        parent_functions = _parent_functions(tree)

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for default in node.args.defaults + node.args.kw_defaults:
                    if isinstance(default, ast.Constant) and default.value == 8000:
                        arg_names = [arg.arg for arg in node.args.args] + [
                            arg.arg for arg in node.args.kwonlyargs
                        ]
                        if "max_tokens" in arg_names:
                            findings.append(f"{relative}:{node.lineno} max_tokens default 8000")

            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute) or node.func.attr != "get":
                continue
            if len(node.args) < 2:
                continue
            key, default = node.args[0], node.args[1]
            if not (isinstance(key, ast.Constant) and key.value == "max_tokens"):
                continue
            if not (isinstance(default, ast.Constant) and isinstance(default.value, int)):
                continue

            function_name = parent_functions.get(node, "")
            allow_key = (str(relative), function_name, default.value)
            if allow_key not in allowed_get_defaults:
                findings.append(
                    f"{relative}:{node.lineno} get('max_tokens', {default.value}) in {function_name}"
                )

    assert findings == []
