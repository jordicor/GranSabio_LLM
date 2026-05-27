from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

RUNTIME_FILES = [
    "ai_service.py",
    "arbiter.py",
    "auto_qa_planner.py",
    "feedback_memory.py",
    "gran_sabio.py",
    "preflight_validator.py",
    "qa_engine.py",
    "core/generation_processor.py",
    "core/generation_routes.py",
    "core/streaming_routes.py",
    "evidence_grounding/budget_scorer.py",
    "evidence_grounding/claim_extractor.py",
    "evidence_grounding/grounding_engine.py",
    "long_text/controller.py",
    "smart_edit/analyzer.py",
    "smart_edit/operations.py",
    "smart_edit/router.py",
    "client/sync_client.py",
    "client/async_client.py",
    "mcp_server/gransabio_mcp_server.py",
]

MODEL_LITERAL_PREFIXES = (
    "gpt-",
    "claude-",
    "gemini-",
    "grok-",
    "qwen",
    "text-embedding",
    "models/text-embedding",
)

MODEL_KEYS = {
    "model",
    "models",
    "model_id",
    "generator_model",
    "qa_models",
    "gran_sabio_model",
    "arbiter_model",
    "analysis_model",
    "embedding_model",
}


def _is_model_literal(value: object) -> bool:
    return isinstance(value, str) and value.lower().startswith(MODEL_LITERAL_PREFIXES)


def _name_contains_model(name: str) -> bool:
    lowered = name.lower()
    return "model" in lowered or "embedding" in lowered


class RuntimeHardcodeVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.violations: list[tuple[int, str]] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._check_function_defaults(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._check_function_defaults(node)
        self.generic_visit(node)

    def visit_keyword(self, node: ast.keyword) -> None:
        if node.arg in MODEL_KEYS and isinstance(node.value, ast.Constant) and _is_model_literal(node.value.value):
            self.violations.append((node.lineno, f"keyword {node.arg}={node.value.value!r}"))
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        if isinstance(node.value, ast.Constant) and _is_model_literal(node.value.value):
            for target in node.targets:
                if isinstance(target, ast.Name) and _name_contains_model(target.id):
                    self.violations.append((node.lineno, f"{target.id}={node.value.value!r}"))
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if (
            isinstance(node.target, ast.Name)
            and _name_contains_model(node.target.id)
            and isinstance(node.value, ast.Constant)
            and _is_model_literal(node.value.value)
        ):
            self.violations.append((node.lineno, f"{node.target.id}={node.value.value!r}"))
        self.generic_visit(node)

    def visit_Dict(self, node: ast.Dict) -> None:
        for key, value in zip(node.keys, node.values):
            if (
                isinstance(key, ast.Constant)
                and str(key.value) in MODEL_KEYS
                and isinstance(value, ast.Constant)
                and _is_model_literal(value.value)
            ):
                self.violations.append((value.lineno, f"dict {key.value!r}: {value.value!r}"))
        self.generic_visit(node)

    def _check_function_defaults(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        positional = node.args.args[-len(node.args.defaults):] if node.args.defaults else []
        defaults = list(zip(positional, node.args.defaults))
        defaults.extend(zip(node.args.kwonlyargs, node.args.kw_defaults))
        for arg, default in defaults:
            if (
                default is not None
                and _name_contains_model(arg.arg)
                and isinstance(default, ast.Constant)
                and _is_model_literal(default.value)
            ):
                self.violations.append((node.lineno, f"default {arg.arg}={default.value!r}"))


def test_runtime_model_choices_are_not_hardcoded() -> None:
    failures: list[str] = []
    for relative_path in RUNTIME_FILES:
        path = REPO_ROOT / relative_path
        tree = ast.parse(path.read_text(encoding="utf-8"))
        visitor = RuntimeHardcodeVisitor()
        visitor.visit(tree)
        failures.extend(
            f"{relative_path}:{line}: {description}"
            for line, description in visitor.violations
        )

    assert not failures, "Runtime model choices must resolve through llm_routing:\n" + "\n".join(failures)
