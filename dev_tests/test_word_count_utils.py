"""
Tests for word_count_utils.py - Word count validation utilities.

Functions tested:
- word_count_config_to_dict(): Config normalization
- is_word_count_enforcement_enabled(): Check if enabled
- extract_target_field(): JSON field extraction
- count_words(): Word counting
- validate_word_count_config(): Config validation
- calculate_word_count_range(): Range calculation
- calculate_word_count_score(): Gradual scoring
- check_word_count_compliance(): Compliance checking
- create_word_count_qa_layer(): QA layer creation
- build_word_count_instructions(): Instruction building
- prepare_qa_layers_with_word_count(): Layer preparation
"""

import pytest
from unittest.mock import Mock, MagicMock
from word_count_utils import (
    word_count_config_to_dict,
    is_word_count_enforcement_enabled,
    extract_target_field,
    count_words,
    validate_word_count_config,
    calculate_word_count_range,
    calculate_word_count_score,
    check_word_count_compliance,
    create_word_count_qa_layer,
    build_word_count_instructions,
    prepare_qa_layers_with_word_count
)
from models import QALayer, ContentRequest


class TestWordCountConfigToDict:
    """Tests for word_count_config_to_dict()."""

    def test_none_returns_none(self):
        """Given: None input, Then: Returns None"""
        assert word_count_config_to_dict(None) is None

    def test_dict_returns_same_dict(self):
        """Given: Dict input, Then: Returns same dict"""
        config = {"enabled": True, "flexibility_percent": 10}
        result = word_count_config_to_dict(config)
        assert result == config

    def test_pydantic_model_with_model_dump(self):
        """Given: Pydantic model with model_dump(), Then: Returns dict"""
        mock_model = Mock()
        mock_model.model_dump = Mock(return_value={"enabled": True})
        result = word_count_config_to_dict(mock_model)
        assert result == {"enabled": True}
        mock_model.model_dump.assert_called_once()

    def test_object_with_dict_method(self):
        """Given: Object with dict() method (no model_dump), Then: Returns dict"""
        mock_obj = Mock(spec=[])
        mock_obj.model_dump = None
        mock_obj.dict = Mock(return_value={"enabled": True})
        result = word_count_config_to_dict(mock_obj)
        assert result == {"enabled": True}

    def test_unsupported_type_returns_none(self):
        """Given: Unsupported type (string), Then: Returns None"""
        result = word_count_config_to_dict("string")
        assert result is None

    def test_unsupported_type_int_returns_none(self):
        """Given: Unsupported type (int), Then: Returns None"""
        result = word_count_config_to_dict(42)
        assert result is None

    def test_model_dump_with_mode_python_fallback(self):
        """Given: model_dump() requires mode='python', Then: Tries fallback"""
        mock_model = Mock()
        # First call raises TypeError, second with mode='python' succeeds
        mock_model.model_dump = Mock(side_effect=[TypeError(), {"enabled": True}])
        result = word_count_config_to_dict(mock_model)
        # Should have called model_dump twice
        assert mock_model.model_dump.call_count == 2

    def test_model_dump_both_fail_tries_dict(self):
        """Given: model_dump() fails both times, Then: Tries dict() method"""
        mock_model = Mock(spec=[])
        # model_dump fails with TypeError both times
        mock_model.model_dump = Mock(side_effect=[TypeError(), TypeError()])
        mock_model.dict = Mock(return_value={"enabled": True})
        result = word_count_config_to_dict(mock_model)
        assert result == {"enabled": True}
        mock_model.dict.assert_called_once()

    def test_all_methods_fail_returns_none(self):
        """Given: All methods fail, Then: Returns None"""
        mock_model = Mock(spec=[])
        # model_dump fails with TypeError
        mock_model.model_dump = Mock(side_effect=[TypeError(), TypeError()])
        # dict also fails
        mock_model.dict = Mock(side_effect=TypeError())
        result = word_count_config_to_dict(mock_model)
        assert result is None


class TestIsWordCountEnforcementEnabled:
    """Tests for is_word_count_enforcement_enabled()."""

    def test_none_returns_false(self):
        """Given: None config, Then: Returns False"""
        assert is_word_count_enforcement_enabled(None) is False

    def test_empty_dict_returns_false(self):
        """Given: Empty dict, Then: Returns False"""
        assert is_word_count_enforcement_enabled({}) is False

    def test_enabled_false_returns_false(self):
        """Given: enabled=False, Then: Returns False"""
        assert is_word_count_enforcement_enabled({"enabled": False}) is False

    def test_enabled_true_returns_true(self):
        """Given: enabled=True, Then: Returns True"""
        assert is_word_count_enforcement_enabled({"enabled": True}) is True

    def test_object_with_enabled_attribute_true(self):
        """Given: Object with enabled=True attribute, Then: Returns True"""
        mock_obj = Mock()
        mock_obj.enabled = True
        assert is_word_count_enforcement_enabled(mock_obj) is True

    def test_object_with_enabled_attribute_false(self):
        """Given: Object with enabled=False attribute, Then: Returns False"""
        mock_obj = Mock()
        mock_obj.enabled = False
        assert is_word_count_enforcement_enabled(mock_obj) is False


class TestExtractTargetField:
    """Tests for extract_target_field()."""

    def test_no_target_field_returns_full_content(self):
        """Given: No target_field, Then: Returns (content, False)"""
        content = "Hello world"
        result, is_json = extract_target_field(content, None)
        assert result == content
        assert is_json is False

    def test_empty_target_field_returns_full_content(self):
        """Given: Empty target_field, Then: Returns (content, False)"""
        content = "Hello world"
        result, is_json = extract_target_field(content, "")
        assert result == content
        assert is_json is False

    def test_extracts_top_level_json_field(self):
        """Given: JSON with target field, Then: Extracts field value"""
        content = '{"text": "extracted value", "other": "ignored"}'
        result, is_json = extract_target_field(content, "text")
        assert result == "extracted value"
        assert is_json is True

    def test_extracts_nested_json_field(self):
        """Given: Nested JSON path, Then: Extracts nested value"""
        content = '{"data": {"content": "nested value"}}'
        result, is_json = extract_target_field(content, "data.content")
        assert result == "nested value"
        assert is_json is True

    def test_missing_field_returns_fallback(self):
        """Given: Non-existent field, Then: Returns (content, False)"""
        content = '{"other": "value"}'
        result, is_json = extract_target_field(content, "missing")
        assert result == content
        assert is_json is False

    def test_invalid_json_returns_fallback(self):
        """Given: Invalid JSON, Then: Returns (content, False)"""
        content = "not json"
        result, is_json = extract_target_field(content, "field")
        assert result == content
        assert is_json is False

    def test_converts_non_string_to_string(self):
        """Given: Non-string field value, Then: Converts to string"""
        content = '{"count": 42}'
        result, is_json = extract_target_field(content, "count")
        assert result == "42"
        assert is_json is True

    def test_deeply_nested_field(self):
        """Given: Deeply nested JSON path, Then: Extracts value"""
        content = '{"level1": {"level2": {"level3": "deep value"}}}'
        result, is_json = extract_target_field(content, "level1.level2.level3")
        assert result == "deep value"
        assert is_json is True


class TestCountWords:
    """Tests for count_words()."""

    def test_empty_string_returns_zero(self):
        """Given: Empty string, Then: Returns 0"""
        assert count_words("") == 0

    def test_whitespace_only_returns_zero(self):
        """Given: Whitespace only, Then: Returns 0"""
        assert count_words("   \t\n  ") == 0

    def test_counts_simple_text(self):
        """Given: Simple text, Then: Returns word count"""
        assert count_words("Hello world") == 2
        assert count_words("one two three") == 3

    def test_handles_multiple_spaces(self):
        """Given: Multiple spaces, Then: Counts correctly"""
        assert count_words("hello    world") == 2

    def test_handles_punctuation(self):
        """Given: Punctuation, Then: Words still counted"""
        result = count_words("Hello, world! Test.")
        assert result == 3

    def test_handles_newlines_and_tabs(self):
        """Given: Text with newlines and tabs, Then: Counts correctly"""
        assert count_words("hello\nworld") == 2
        assert count_words("one\ttwo\tthree") == 3

    def test_handles_long_text(self):
        """Given: Long text, Then: Returns correct count"""
        text = "word " * 1000
        assert count_words(text) == 1000


class TestValidateWordCountConfig:
    """Tests for validate_word_count_config()."""

    def test_valid_config_passes(self):
        """Given: Valid config, Then: Returns (True, '')"""
        config = {
            "enabled": True,
            "flexibility_percent": 10,
            "direction": "both",
            "severity": "important"
        }
        is_valid, error = validate_word_count_config(config)
        assert is_valid is True
        assert error == ""

    def test_disabled_config_fails(self):
        """Given: enabled=False, Then: Returns (False, error)"""
        config = {"enabled": False, "flexibility_percent": 10}
        is_valid, error = validate_word_count_config(config)
        assert is_valid is False
        assert "enabled" in error.lower()

    def test_missing_flexibility_fails(self):
        """Given: Missing flexibility_percent, Then: Returns (False, error)"""
        config = {"enabled": True}
        is_valid, error = validate_word_count_config(config)
        assert is_valid is False
        assert "flexibility" in error.lower()

    def test_invalid_flexibility_over_100_fails(self):
        """Given: flexibility > 100, Then: Returns (False, error)"""
        config = {"enabled": True, "flexibility_percent": 150}
        is_valid, error = validate_word_count_config(config)
        assert is_valid is False

    def test_negative_flexibility_fails(self):
        """Given: flexibility < 0, Then: Returns (False, error)"""
        config = {"enabled": True, "flexibility_percent": -5}
        is_valid, error = validate_word_count_config(config)
        assert is_valid is False

    def test_invalid_direction_fails(self):
        """Given: Invalid direction, Then: Returns (False, error)"""
        config = {
            "enabled": True,
            "flexibility_percent": 10,
            "direction": "invalid"
        }
        is_valid, error = validate_word_count_config(config)
        assert is_valid is False
        assert "direction" in error.lower()

    def test_invalid_severity_fails(self):
        """Given: Invalid severity, Then: Returns (False, error)"""
        config = {
            "enabled": True,
            "flexibility_percent": 10,
            "severity": "invalid"
        }
        is_valid, error = validate_word_count_config(config)
        assert is_valid is False
        assert "severity" in error.lower()

    def test_non_dict_config_fails(self):
        """Given: Non-dict config, Then: Returns (False, error)"""
        is_valid, error = validate_word_count_config("not a dict")
        assert is_valid is False
        assert "dict" in error.lower() or "pydantic" in error.lower()


class TestCalculateWordCountRange:
    """Tests for calculate_word_count_range()."""

    def test_no_limits_returns_open_range(self):
        """Given: No min/max words, Then: Returns (0, inf)"""
        abs_min, abs_max = calculate_word_count_range(
            None, None, flexibility_percent=10, direction="both"
        )
        assert abs_min == 0
        assert abs_max == float('inf')

    def test_both_limits_with_flexibility_both(self):
        """Given: min=400, max=600, flex=10%, direction=both"""
        abs_min, abs_max = calculate_word_count_range(
            min_words=400, max_words=600,
            flexibility_percent=10, direction="both"
        )
        # 400 - 10% = 360, 600 + 10% = 660
        assert abs_min == 360
        assert abs_max == 660

    def test_direction_more_only_expands_max(self):
        """Given: direction=more, Then: Only max expands"""
        abs_min, abs_max = calculate_word_count_range(
            min_words=400, max_words=600,
            flexibility_percent=10, direction="more"
        )
        assert abs_min == 400  # Unchanged
        assert abs_max == 660  # Expanded

    def test_direction_less_only_expands_min(self):
        """Given: direction=less, Then: Only min shrinks"""
        abs_min, abs_max = calculate_word_count_range(
            min_words=400, max_words=600,
            flexibility_percent=10, direction="less"
        )
        assert abs_min == 360  # Shrunk
        assert abs_max == 600  # Unchanged

    def test_min_only_calculates_max(self):
        """Given: Only min_words, Then: Calculates reasonable max"""
        abs_min, abs_max = calculate_word_count_range(
            min_words=500, max_words=None,
            flexibility_percent=10, direction="both"
        )
        # min_words * (1 - 0.10) = 450
        assert abs_min == 450
        # max should be min * 1.5 * 1.1 = 825
        assert abs_max == 825

    def test_max_only_calculates_min(self):
        """Given: Only max_words, Then: Calculates reasonable min"""
        abs_min, abs_max = calculate_word_count_range(
            min_words=None, max_words=1000,
            flexibility_percent=10, direction="both"
        )
        # base_min = max(1, 1000 * 0.7) = 700
        # abs_min = 700 * 0.9 = 630
        assert abs_min == 630
        # abs_max = 1000 * 1.1 = 1100
        assert abs_max == 1100

    def test_zero_flexibility(self):
        """Given: 0% flexibility, Then: Range equals target"""
        abs_min, abs_max = calculate_word_count_range(
            min_words=500, max_words=1000,
            flexibility_percent=0, direction="both"
        )
        assert abs_min == 500
        assert abs_max == 1000


class TestCalculateWordCountScore:
    """Tests for calculate_word_count_score()."""

    def test_within_target_range_scores_10(self):
        """Given: Count within target range, Then: Score = 10.0"""
        score = calculate_word_count_score(
            actual_count=500,
            target_min=400,
            target_max=600,
            abs_min=350,
            abs_max=650
        )
        assert score == 10.0

    def test_at_target_min_boundary_scores_10(self):
        """Given: Count exactly at target_min, Then: Score = 10.0"""
        score = calculate_word_count_score(
            actual_count=400,
            target_min=400,
            target_max=600,
            abs_min=350,
            abs_max=650
        )
        assert score == 10.0

    def test_at_target_max_boundary_scores_10(self):
        """Given: Count exactly at target_max, Then: Score = 10.0"""
        score = calculate_word_count_score(
            actual_count=600,
            target_min=400,
            target_max=600,
            abs_min=350,
            abs_max=650
        )
        assert score == 10.0

    def test_outside_absolute_limits_scores_0(self):
        """Given: Count outside absolute limits (below), Then: Score = 0.0"""
        score = calculate_word_count_score(
            actual_count=300,  # Below abs_min=350
            target_min=400,
            target_max=600,
            abs_min=350,
            abs_max=650
        )
        assert score == 0.0

    def test_above_absolute_max_scores_0(self):
        """Given: Count above absolute max, Then: Score = 0.0"""
        score = calculate_word_count_score(
            actual_count=700,  # Above abs_max=650
            target_min=400,
            target_max=600,
            abs_min=350,
            abs_max=650
        )
        assert score == 0.0

    def test_in_lower_buffer_zone_gradual_score(self):
        """Given: Count in lower buffer zone, Then: Score between 0-10"""
        score = calculate_word_count_score(
            actual_count=375,  # Between 350 (abs_min) and 400 (target_min)
            target_min=400,
            target_max=600,
            abs_min=350,
            abs_max=650
        )
        # position_in_buffer = (375 - 350) / (400 - 350) = 25/50 = 0.5
        # score = 0.5 * 10 = 5.0
        assert score == 5.0

    def test_in_upper_buffer_zone_gradual_score(self):
        """Given: Count in upper buffer zone, Then: Score between 0-10"""
        score = calculate_word_count_score(
            actual_count=625,  # Between 600 (target_max) and 650 (abs_max)
            target_min=400,
            target_max=600,
            abs_min=350,
            abs_max=650
        )
        # position_in_buffer = (625 - 600) / (650 - 600) = 25/50 = 0.5
        # score = 10 - (0.5 * 10) = 5.0
        assert score == 5.0

    def test_only_min_words_above_threshold(self):
        """Given: Only target_min and count >= min, Then: Score = 10.0"""
        score = calculate_word_count_score(
            actual_count=600,
            target_min=400,
            target_max=None,
            abs_min=350,
            abs_max=650
        )
        assert score == 10.0

    def test_only_max_words_below_threshold(self):
        """Given: Only target_max and count <= max, Then: Score = 10.0"""
        score = calculate_word_count_score(
            actual_count=400,
            target_min=None,
            target_max=600,
            abs_min=350,
            abs_max=650
        )
        assert score == 10.0

    def test_zero_buffer_size_lower(self):
        """Given: Zero lower buffer size, Then: Returns 0.0"""
        score = calculate_word_count_score(
            actual_count=399,  # Just below target_min
            target_min=400,
            target_max=600,
            abs_min=400,  # Same as target_min = zero buffer
            abs_max=650
        )
        assert score == 0.0

    def test_zero_buffer_size_upper(self):
        """Given: Zero upper buffer size, Then: Returns 0.0"""
        score = calculate_word_count_score(
            actual_count=601,  # Just above target_max
            target_min=400,
            target_max=600,
            abs_min=350,
            abs_max=600  # Same as target_max = zero buffer
        )
        assert score == 0.0

    def test_no_targets_returns_fallback(self):
        """Given: No target_min and no target_max, Then: Returns fallback score"""
        score = calculate_word_count_score(
            actual_count=500,
            target_min=None,
            target_max=None,
            abs_min=350,
            abs_max=650
        )
        # Should return 5.0 fallback
        assert score == 5.0


class TestCheckWordCountCompliance:
    """Tests for check_word_count_compliance()."""

    def test_compliant_content_returns_true(self):
        """Given: Content within limits, Then: complies=True"""
        config = {
            "enabled": True,
            "flexibility_percent": 10,
            "direction": "both",
            "severity": "important"
        }
        result = check_word_count_compliance(
            content="word " * 500,  # 500 words
            min_words=400,
            max_words=600,
            config=config
        )
        assert result["complies"] is True
        assert result["score"] == 10.0

    def test_non_compliant_content_returns_false(self):
        """Given: Content outside limits, Then: complies=False"""
        config = {
            "enabled": True,
            "flexibility_percent": 10,
            "direction": "both",
            "severity": "important"
        }
        result = check_word_count_compliance(
            content="word " * 100,  # Only 100 words
            min_words=400,
            max_words=600,
            config=config
        )
        assert result["complies"] is False
        assert result["score"] == 0.0

    def test_returns_all_expected_fields(self):
        """Given: Any input, Then: Returns dict with all fields"""
        config = {
            "enabled": True,
            "flexibility_percent": 10,
            "direction": "both",
            "severity": "important"
        }
        result = check_word_count_compliance(
            content="word " * 500,
            min_words=400,
            max_words=600,
            config=config
        )
        assert "complies" in result
        assert "score" in result
        assert "actual_count" in result
        assert "required_min" in result
        assert "required_max" in result
        assert "target_min" in result
        assert "target_max" in result
        assert "flexibility_percent" in result
        assert "direction" in result
        assert "severity" in result
        assert "is_json_extraction" in result

    def test_json_target_field_extraction(self):
        """Given: JSON content with target_field, Then: Counts field words"""
        config = {
            "enabled": True,
            "flexibility_percent": 10,
            "direction": "both",
            "severity": "important",
            "target_field": "text"
        }
        # Create JSON with 500 words in "text" field
        text_content = " ".join(["word"] * 500)
        content = '{"text": "' + text_content + '", "meta": "ignored"}'
        result = check_word_count_compliance(
            content=content,
            min_words=400,
            max_words=600,
            config=config
        )
        assert result["is_json_extraction"] is True
        assert result["actual_count"] == 500

    def test_invalid_config_raises_error(self):
        """Given: Invalid config type, Then: Raises ValueError"""
        with pytest.raises(ValueError, match="dict-like"):
            check_word_count_compliance(
                content="test content",
                min_words=100,
                max_words=200,
                config="invalid"
            )


class TestCreateWordCountQALayer:
    """Tests for create_word_count_qa_layer()."""

    def test_creates_valid_qa_layer(self):
        """Given: Valid config, Then: Returns QALayer"""
        config = {
            "enabled": True,
            "flexibility_percent": 10,
            "direction": "both",
            "severity": "important"
        }
        layer = create_word_count_qa_layer(
            min_words=400,
            max_words=600,
            config=config
        )
        assert isinstance(layer, QALayer)
        assert layer.name == "Word Count Enforcement"
        assert layer.order == 0  # Should run first

    def test_deal_breaker_severity_adds_criteria(self):
        """Given: severity=deal_breaker, Then: deal_breaker_criteria set"""
        config = {
            "enabled": True,
            "flexibility_percent": 10,
            "direction": "both",
            "severity": "deal_breaker"
        }
        layer = create_word_count_qa_layer(
            min_words=400,
            max_words=600,
            config=config
        )
        assert layer.deal_breaker_criteria is not None
        assert "360" in layer.deal_breaker_criteria  # abs_min
        assert "660" in layer.deal_breaker_criteria  # abs_max

    def test_important_severity_no_deal_breaker(self):
        """Given: severity=important, Then: No deal_breaker_criteria"""
        config = {
            "enabled": True,
            "flexibility_percent": 10,
            "direction": "both",
            "severity": "important"
        }
        layer = create_word_count_qa_layer(
            min_words=400,
            max_words=600,
            config=config
        )
        assert layer.deal_breaker_criteria is None

    def test_min_only_creates_valid_layer(self):
        """Given: Only min_words, Then: Creates valid layer"""
        config = {
            "enabled": True,
            "flexibility_percent": 10,
            "direction": "both",
            "severity": "important"
        }
        layer = create_word_count_qa_layer(
            min_words=500,
            max_words=None,
            config=config
        )
        assert "at least 500 words" in layer.description

    def test_max_only_creates_valid_layer(self):
        """Given: Only max_words, Then: Creates valid layer"""
        config = {
            "enabled": True,
            "flexibility_percent": 10,
            "direction": "both",
            "severity": "important"
        }
        layer = create_word_count_qa_layer(
            min_words=None,
            max_words=1000,
            config=config
        )
        assert "at most 1000 words" in layer.description

    def test_invalid_config_raises_error(self):
        """Given: Invalid config, Then: Raises ValueError"""
        with pytest.raises(ValueError, match="dict-like"):
            create_word_count_qa_layer(
                min_words=400,
                max_words=600,
                config="invalid"
            )


class TestBuildWordCountInstructions:
    """Tests for build_word_count_instructions()."""

    def test_no_limits_returns_empty(self):
        """Given: No min/max words, Then: Returns empty string"""
        request = Mock()
        request.min_words = None
        request.max_words = None
        result = build_word_count_instructions(request)
        assert result == ""

    def test_both_limits_returns_range_instruction(self):
        """Given: Both limits set, Then: Returns range instruction"""
        request = Mock()
        request.min_words = 500
        request.max_words = 1000
        result = build_word_count_instructions(request)
        assert "500" in result
        assert "1000" in result
        assert "CRITICAL" in result

    def test_min_only_returns_minimum_instruction(self):
        """Given: Only min_words, Then: Returns minimum instruction"""
        request = Mock()
        request.min_words = 500
        request.max_words = None
        result = build_word_count_instructions(request)
        assert "500" in result
        assert "Minimum" in result

    def test_max_only_returns_maximum_instruction(self):
        """Given: Only max_words, Then: Returns maximum instruction"""
        request = Mock()
        request.min_words = None
        request.max_words = 1000
        result = build_word_count_instructions(request)
        assert "1000" in result
        assert "Maximum" in result


class TestPrepareQALayersWithWordCount:
    """Tests for prepare_qa_layers_with_word_count()."""

    @pytest.fixture
    def base_request(self):
        """Create a base ContentRequest for testing."""
        return ContentRequest(
            prompt="Test prompt for content generation that is long enough",
            qa_layers=[],
            min_words=None,
            max_words=None,
            word_count_enforcement=None,
            lexical_diversity=None,
            phrase_frequency=None,
            cumulative_text=None
        )

    def test_empty_layers_returns_empty(self, base_request):
        """Given: No QA layers, no enforcement, Then: Returns empty list"""
        result = prepare_qa_layers_with_word_count(base_request)
        assert result == []

    def test_preserves_existing_layers(self, base_request):
        """Given: Existing QA layers, Then: Preserves them"""
        existing_layer = QALayer(
            name="Test Layer",
            description="Test description",
            criteria="Test criteria"
        )
        base_request.qa_layers = [existing_layer]
        result = prepare_qa_layers_with_word_count(base_request)
        assert len(result) == 1
        assert result[0].name == "Test Layer"

    def test_adds_word_count_layer_when_enabled(self, base_request):
        """Given: Word count enforcement enabled, Then: Adds layer at position 0"""
        from models import WordCountEnforcement
        base_request.word_count_enforcement = WordCountEnforcement(
            enabled=True,
            flexibility_percent=10,
            direction="both",
            severity="important"
        )
        base_request.min_words = 400
        base_request.max_words = 600

        result = prepare_qa_layers_with_word_count(base_request)

        assert len(result) == 1
        assert result[0].name == "Word Count Enforcement"
        assert result[0].order == 0

    def test_word_count_layer_inserted_first(self, base_request):
        """Given: Existing layers and word count enabled, Then: Word count first"""
        from models import WordCountEnforcement
        existing_layer = QALayer(
            name="Existing Layer",
            description="Test",
            criteria="Test"
        )
        base_request.qa_layers = [existing_layer]
        base_request.word_count_enforcement = WordCountEnforcement(
            enabled=True,
            flexibility_percent=10,
            direction="both",
            severity="important"
        )
        base_request.min_words = 400
        base_request.max_words = 600

        result = prepare_qa_layers_with_word_count(base_request)

        assert len(result) == 2
        assert result[0].name == "Word Count Enforcement"
        assert result[1].name == "Existing Layer"

    def test_no_layer_without_word_limits(self, base_request):
        """Given: Enforcement enabled but no word limits, Then: No layer added"""
        from models import WordCountEnforcement
        base_request.word_count_enforcement = WordCountEnforcement(
            enabled=True,
            flexibility_percent=10,
            direction="both",
            severity="important"
        )
        # No min_words or max_words set

        result = prepare_qa_layers_with_word_count(base_request)

        assert len(result) == 0

    def test_invalid_config_logs_warning_returns_original(self, base_request):
        """Given: Invalid word count config, Then: Returns original layers"""
        existing_layer = QALayer(
            name="Existing",
            description="Test",
            criteria="Test"
        )
        base_request.qa_layers = [existing_layer]
        # Create a mock that looks enabled but has invalid config
        mock_config = Mock()
        mock_config.enabled = True
        mock_config.model_dump = Mock(return_value={
            "enabled": True,
            "flexibility_percent": -10,  # Invalid
            "direction": "both",
            "severity": "important"
        })
        base_request.word_count_enforcement = mock_config
        base_request.min_words = 400
        base_request.max_words = 600

        result = prepare_qa_layers_with_word_count(base_request)

        # Should return original layers without word count layer
        assert len(result) == 1
        assert result[0].name == "Existing"

    def test_preflight_removes_duplicate_layers(self, base_request):
        """Given: Preflight identifies duplicates, Then: Removes them"""
        duplicate_layer = QALayer(
            name="Duplicate Word Count",
            description="Test",
            criteria="Test"
        )
        base_request.qa_layers = [duplicate_layer]

        preflight_result = Mock()
        preflight_result.duplicate_word_count_layers_to_remove = ["Duplicate Word Count"]
        preflight_result.enable_algorithmic_word_count = False

        result = prepare_qa_layers_with_word_count(base_request, preflight_result)

        assert len(result) == 0

    def test_config_not_normalizable_returns_original(self, base_request):
        """Given: Config cannot be normalized to dict, Then: Returns original layers"""
        existing_layer = QALayer(
            name="Existing",
            description="Test",
            criteria="Test"
        )
        base_request.qa_layers = [existing_layer]
        # Create a mock that looks enabled but returns None when normalized
        mock_config = Mock(spec=[])
        mock_config.enabled = True
        mock_config.model_dump = None
        mock_config.dict = None
        base_request.word_count_enforcement = mock_config
        base_request.min_words = 400
        base_request.max_words = 600

        result = prepare_qa_layers_with_word_count(base_request)

        # Should return original layers without word count layer
        assert len(result) == 1
        assert result[0].name == "Existing"

    def test_lexical_diversity_layer_added(self, base_request):
        """Given: Lexical diversity enabled, Then: Adds layer"""
        from models import LexicalDiversityConfig
        base_request.lexical_diversity = LexicalDiversityConfig(enabled=True)

        result = prepare_qa_layers_with_word_count(base_request)

        assert len(result) == 1
        assert result[0].name == "Lexical Diversity Guard"

    def test_lexical_diversity_after_word_count(self, base_request):
        """Given: Both word count and lexical diversity, Then: Lexical after word count"""
        from models import WordCountEnforcement, LexicalDiversityConfig
        base_request.word_count_enforcement = WordCountEnforcement(
            enabled=True,
            flexibility_percent=10,
            direction="both",
            severity="important"
        )
        base_request.min_words = 400
        base_request.max_words = 600
        base_request.lexical_diversity = LexicalDiversityConfig(enabled=True)

        result = prepare_qa_layers_with_word_count(base_request)

        assert len(result) == 2
        assert result[0].name == "Word Count Enforcement"
        assert result[1].name == "Lexical Diversity Guard"

    def test_phrase_frequency_layer_added(self, base_request):
        """Given: Phrase frequency enabled, Then: Adds layer"""
        from models import PhraseFrequencyConfig, PhraseFrequencyRule
        base_request.phrase_frequency = PhraseFrequencyConfig(
            enabled=True,
            rules=[
                PhraseFrequencyRule(
                    name="test_rule",
                    min_length=3,
                    max_ratio_tokens=0.1
                )
            ]
        )

        result = prepare_qa_layers_with_word_count(base_request)

        assert len(result) == 1
        assert result[0].name == "Phrase Frequency Guard"

    def test_all_layers_combined_ordering(self, base_request):
        """Given: All layer types enabled, Then: Correct ordering"""
        from models import WordCountEnforcement, LexicalDiversityConfig, PhraseFrequencyConfig, PhraseFrequencyRule
        base_request.word_count_enforcement = WordCountEnforcement(
            enabled=True,
            flexibility_percent=10,
            direction="both",
            severity="important"
        )
        base_request.min_words = 400
        base_request.max_words = 600
        base_request.lexical_diversity = LexicalDiversityConfig(enabled=True)
        base_request.phrase_frequency = PhraseFrequencyConfig(
            enabled=True,
            rules=[
                PhraseFrequencyRule(
                    name="test_rule",
                    min_length=3,
                    max_ratio_tokens=0.1
                )
            ]
        )

        result = prepare_qa_layers_with_word_count(base_request)

        assert len(result) == 3
        assert result[0].name == "Word Count Enforcement"
        assert result[1].name == "Lexical Diversity Guard"
        assert result[2].name == "Phrase Frequency Guard"

    def test_cumulative_repetition_layer_added(self, base_request):
        """Given: Cumulative text and phrase frequency enabled, Then: Adds cumulative layer"""
        from models import PhraseFrequencyConfig, PhraseFrequencyRule
        base_request.cumulative_text = "Previous chapter content here with some words."
        base_request.phrase_frequency = PhraseFrequencyConfig(
            enabled=True,
            rules=[
                PhraseFrequencyRule(
                    name="test_rule",
                    min_length=3,
                    max_ratio_tokens=0.1
                )
            ]
        )

        result = prepare_qa_layers_with_word_count(base_request)

        layer_names = [layer.name for layer in result]
        assert "Cumulative Repetition Guard" in layer_names
        assert "Phrase Frequency Guard" in layer_names

    def test_duplicate_lexical_diversity_not_added(self, base_request):
        """Given: Lexical Diversity Guard already exists, Then: Not added again"""
        from models import LexicalDiversityConfig
        existing_layer = QALayer(
            name="Lexical Diversity Guard",
            description="Existing",
            criteria="Test"
        )
        base_request.qa_layers = [existing_layer]
        base_request.lexical_diversity = LexicalDiversityConfig(enabled=True)

        result = prepare_qa_layers_with_word_count(base_request)

        # Should only have the original layer, not a duplicate
        assert len(result) == 1
        assert result[0].description == "Existing"

    def test_duplicate_phrase_frequency_not_added(self, base_request):
        """Given: Phrase Frequency Guard already exists, Then: Not added again"""
        from models import PhraseFrequencyConfig, PhraseFrequencyRule
        existing_layer = QALayer(
            name="Phrase Frequency Guard",
            description="Existing",
            criteria="Test"
        )
        base_request.qa_layers = [existing_layer]
        base_request.phrase_frequency = PhraseFrequencyConfig(
            enabled=True,
            rules=[
                PhraseFrequencyRule(
                    name="test_rule",
                    min_length=3,
                    max_ratio_tokens=0.1
                )
            ]
        )

        result = prepare_qa_layers_with_word_count(base_request)

        # Should only have the original layer, not a duplicate
        assert len(result) == 1
        assert result[0].description == "Existing"

    def test_preflight_algorithmic_word_count_logging(self, base_request):
        """Given: Preflight enables algorithmic word count, Then: Logs info"""
        preflight_result = Mock()
        preflight_result.duplicate_word_count_layers_to_remove = []
        preflight_result.enable_algorithmic_word_count = True

        # Should not raise error
        result = prepare_qa_layers_with_word_count(base_request, preflight_result)
        assert result == []
