"""Regression guard — ``ArbiterDecision.MERGE`` must be gone.

The Phase 3 refactor (§3.4.2) eliminates the MERGE decision entirely. This
file fail-fast if any part of the Arbiter stack (enum, prompt template,
schema) still references the legacy value.
"""

from __future__ import annotations

import pytest

from arbiter import (
    ARBITER_RESPONSE_SCHEMA,
    ARBITER_USER_PROMPT_TEMPLATE,
    ArbiterDecision,
)


class TestArbiterDecisionEnum:
    """The enum must no longer carry a ``MERGE`` member."""

    def test_merge_member_is_not_defined(self):
        """Accessing ``ArbiterDecision.MERGE`` must raise ``AttributeError``."""
        with pytest.raises(AttributeError):
            _ = ArbiterDecision.MERGE  # noqa: B018 - intentional access

    def test_only_apply_and_discard_survive(self):
        """The decision enum has exactly two members after the refactor."""
        members = {member.name for member in ArbiterDecision}
        assert members == {"APPLY", "DISCARD"}

    def test_enum_values_are_lowercase_strings(self):
        """String values stay lowercase for backwards-compatible history dumps."""
        assert ArbiterDecision.APPLY.value == "apply"
        assert ArbiterDecision.DISCARD.value == "discard"


class TestArbiterUserPromptTemplate:
    """The user prompt template must no longer mention MERGE."""

    def test_template_contains_no_merge_mention(self):
        """Plain-text grep: the word ``merge`` must not appear anywhere."""
        lowered = ARBITER_USER_PROMPT_TEMPLATE.lower()
        assert "merge" not in lowered, (
            "ARBITER_USER_PROMPT_TEMPLATE still references 'merge' — the"
            " Phase 3 refactor required this wording to be removed."
        )

    def test_template_uses_uppercase_decision_enum_example(self):
        """The JSON example must show ``APPLY|DISCARD`` (uppercase)."""
        assert "APPLY|DISCARD" in ARBITER_USER_PROMPT_TEMPLATE

    def test_template_does_not_carry_merge_with_field(self):
        """The legacy ``merge_with`` field in the JSON example must be gone."""
        assert "merge_with" not in ARBITER_USER_PROMPT_TEMPLATE


class TestArbiterResponseSchema:
    """The structured-outputs schema must enforce only APPLY/DISCARD."""

    def test_schema_decision_enum_excludes_merge(self):
        """``decisions[].decision`` enum must contain exactly APPLY and DISCARD."""
        decision_schema = (
            ARBITER_RESPONSE_SCHEMA["properties"]["decisions"]["items"]
            ["properties"]["decision"]
        )
        assert decision_schema["enum"] == ["APPLY", "DISCARD"]

    def test_schema_has_additional_properties_false_at_every_level(self):
        """Strict structured outputs: every object level locks out extras."""
        root = ARBITER_RESPONSE_SCHEMA
        assert root["additionalProperties"] is False

        decisions_item = root["properties"]["decisions"]["items"]
        assert decisions_item["additionalProperties"] is False

        conflicts_item = root["properties"]["conflicts_resolved"]["items"]
        assert conflicts_item["additionalProperties"] is False

    def test_schema_required_fields_cover_all_properties(self):
        """OpenAI strict mode demands every property listed in ``required``."""
        assert set(ARBITER_RESPONSE_SCHEMA["required"]) == set(
            ARBITER_RESPONSE_SCHEMA["properties"].keys()
        )

        decisions_item = ARBITER_RESPONSE_SCHEMA["properties"]["decisions"]["items"]
        assert set(decisions_item["required"]) == set(
            decisions_item["properties"].keys()
        )

        conflicts_item = (
            ARBITER_RESPONSE_SCHEMA["properties"]["conflicts_resolved"]["items"]
        )
        assert set(conflicts_item["required"]) == set(
            conflicts_item["properties"].keys()
        )
