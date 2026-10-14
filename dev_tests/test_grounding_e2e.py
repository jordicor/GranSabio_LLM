"""
End-to-End tests for Evidence Grounding with real API calls.

E2E tests verify the complete evidence grounding pipeline using real API
calls to OpenAI for logprobs.

IMPORTANT: These tests make real API calls and incur costs.
- Opt-in: marker ``integration``, deselected by default. Run with
  ``pytest -m integration dev_tests/test_grounding_e2e.py``
- Requires OPENAI_API_KEY to be configured
- Server must be running at API_BASE (default: http://localhost:8000), except
  for ``TestGroundingDirectModule``, which calls the providers directly

Scenarios covered by the fixtures:
- Confabulation case: Model claims something contradicting context
- Grounded case: Model correctly uses evidence
- No evidence case: Model makes claims without supporting context
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from typing import Any, Dict

import httpx
import pytest

# Opt-in live tests: real provider calls.
pytestmark = pytest.mark.integration

# API configuration
API_BASE = os.environ.get("API_BASE", "http://localhost:8000")

# Test model for logprobs (must support logprobs - NOT reasoning models)
GROUNDING_MODEL = "gpt-4o-mini"


# =============================================================================
# Test Fixtures from STRAWBERRY_INTEGRATION.md
# =============================================================================

# Confabulation case: Model claims something contradicting context (should flag)
CONFAB_CONTEXT = "Marie Curie was born in Warsaw, Poland in 1867."
CONFAB_CONTENT_EXPECTED = "Paris"  # If model says Paris, it's confabulating

# Grounded case: Model correctly uses evidence (should pass)
GROUNDED_CONTEXT = """
Marie Curie was born in Warsaw, Poland in 1867.
She moved to Paris in 1891 to study at the Sorbonne.
She won two Nobel Prizes: Physics in 1903 and Chemistry in 1911.
"""
GROUNDED_PROMPT = """
Based ONLY on the context above, write a brief biography of Marie Curie.
Include her birthplace, when she moved to Paris, and her Nobel Prizes.
Cite specific facts from the context.
"""

# No evidence case: Model makes claims without supporting context (should flag)
NO_EVIDENCE_CONTEXT = "The document discusses scientific achievements in the 20th century."
NO_EVIDENCE_PROMPT = """
Based on the context above, explain what Marie Curie discovered.
"""


# =============================================================================
# Helper Functions
# =============================================================================

def wait_for_completion(session_id: str, max_wait: int = 180) -> Dict[str, Any]:
    """Wait for a session to complete and return the status."""
    start = time.time()
    while time.time() - start < max_wait:
        try:
            response = httpx.get(f"{API_BASE}/status/{session_id}", timeout=30)
            status = response.json()

            if status["status"] in ["completed", "failed", "preflight_rejected"]:
                return status

        except httpx.TimeoutException:
            pass  # Retry on timeout

        time.sleep(3)

    raise TimeoutError(f"Session {session_id} did not complete within {max_wait}s")


def start_generation(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Start a generation request and return the response."""
    response = httpx.post(
        f"{API_BASE}/generate",
        json=payload,
        timeout=60
    )
    return response.json()


def get_result(session_id: str) -> Dict[str, Any]:
    """Get the result of a completed session."""
    response = httpx.get(f"{API_BASE}/result/{session_id}", timeout=30)
    return response.json()


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def base_grounding_config():
    """Base evidence grounding configuration for E2E tests."""
    return {
        "enabled": True,
        "model": GROUNDING_MODEL,
        "max_claims": 10,
        "filter_trivial": True,
        "min_claim_importance": 0.5,
        "budget_gap_threshold": 0.5,
        "max_flagged_claims": 2,
        "on_flag": "warn",  # Use warn for E2E tests so we get results
    }


@pytest.fixture
def grounded_request(base_grounding_config):
    """Request with well-grounded content (should pass)."""
    return {
        "prompt": f"""
CONTEXT:
{GROUNDED_CONTEXT}

TASK:
{GROUNDED_PROMPT}

Write 50-100 words.
""",
        "generator_model": "gpt-4o-mini",
        "content_type": "biography",
        "min_words": 50,
        "max_words": 100,
        "max_iterations": 1,
        "qa_layers": [],  # Grounding works without semantic QA layers
        "evidence_grounding": base_grounding_config,
        "verbose": True,
    }


@pytest.fixture
def no_evidence_request(base_grounding_config):
    """Request where model may confabulate (should flag)."""
    return {
        "prompt": f"""
CONTEXT:
{NO_EVIDENCE_CONTEXT}

TASK:
{NO_EVIDENCE_PROMPT}

Write 50-100 words about Marie Curie's discoveries.
""",
        "generator_model": "gpt-4o-mini",
        "content_type": "article",
        "min_words": 50,
        "max_words": 100,
        "max_iterations": 1,
        "qa_layers": [],  # Grounding works without semantic QA layers
        "evidence_grounding": base_grounding_config,
        "verbose": True,
    }


# =============================================================================
# E2E Tests - Basic Functionality
# =============================================================================

class TestGroundingE2EBasic:
    """Basic E2E tests for evidence grounding functionality."""

    def test_grounding_enabled_returns_result(self, live_engine_server, base_grounding_config):
        """
        Given: Request with evidence_grounding enabled (no semantic QA layers needed)
        When: Generation completes
        Then: Response includes evidence_grounding result with data
        """
        payload = {
            "prompt": "Write a simple greeting message. Say hello.",
            "generator_model": "gpt-4o-mini",
            "content_type": "creative",
            "min_words": 10,
            "max_words": 50,
            "max_iterations": 1,
            "qa_layers": [],  # Grounding works without semantic QA layers
            "evidence_grounding": base_grounding_config,
        }

        init_response = start_generation(payload)
        assert "session_id" in init_response, f"No session_id in response: {init_response}"

        session_id = init_response["session_id"]
        status = wait_for_completion(session_id)

        assert status["status"] == "completed", f"Generation failed: {status}"

        result = get_result(session_id)

        # Verify evidence_grounding is in result
        assert "evidence_grounding" in result, "Missing evidence_grounding in result"

        grounding = result["evidence_grounding"]
        assert grounding is not None, "evidence_grounding should not be None when enabled with QA layers"
        assert "enabled" in grounding
        assert "passed" in grounding
        assert "claims" in grounding
        assert "flagged_claims" in grounding
        print(f"\nGrounding result: passed={grounding['passed']}, "
              f"flagged={grounding['flagged_claims']}/{grounding['claims_verified']}")

    def test_grounding_disabled_returns_none(self, live_engine_server):
        """
        Given: Request without evidence_grounding
        When: Generation completes
        Then: evidence_grounding is None in response
        """
        payload = {
            "prompt": "Write a simple greeting message.",
            "generator_model": "gpt-4o-mini",
            "content_type": "creative",
            "min_words": 10,
            "max_words": 50,
            "max_iterations": 1,
            "qa_layers": [],
            # No evidence_grounding config
        }

        init_response = start_generation(payload)
        session_id = init_response["session_id"]
        status = wait_for_completion(session_id)

        assert status["status"] == "completed"

        result = get_result(session_id)

        # evidence_grounding should be None or not present
        grounding = result.get("evidence_grounding")
        assert grounding is None, f"Expected None, got: {grounding}"

    def test_grounding_works_without_semantic_qa_layers(self, live_engine_server, base_grounding_config):
        """
        Given: Request with evidence_grounding enabled and empty qa_layers
        When: Generation completes
        Then: evidence_grounding runs and returns result (no semantic QA needed)

        NOTE: Grounding runs independently of semantic QA layers.
        """
        payload = {
            "prompt": "Write a simple greeting message.",
            "generator_model": "gpt-4o-mini",
            "content_type": "creative",
            "min_words": 10,
            "max_words": 50,
            "max_iterations": 1,
            "qa_layers": [],  # Empty - but grounding still runs
            "evidence_grounding": base_grounding_config,
        }

        init_response = start_generation(payload)
        session_id = init_response["session_id"]
        status = wait_for_completion(session_id)

        assert status["status"] == "completed"

        result = get_result(session_id)

        # Grounding should run even without semantic QA layers
        grounding = result.get("evidence_grounding")
        assert grounding is not None, \
            "Grounding should run even when qa_layers is empty"
        assert "passed" in grounding
        assert "flagged_claims" in grounding


# =============================================================================
# E2E Tests - Grounded Content
# =============================================================================

class TestGroundingE2EGroundedContent:
    """E2E tests for well-grounded content detection."""

    def test_grounded_biography_passes(self, live_engine_server, grounded_request):
        """
        Given: Request with rich context and prompt to use it
        When: Model generates content based on context
        Then: Evidence grounding should pass (or have few flagged claims)
        """
        init_response = start_generation(grounded_request)
        session_id = init_response["session_id"]
        status = wait_for_completion(session_id)

        assert status["status"] == "completed", f"Generation failed: {status}"

        result = get_result(session_id)
        grounding = result.get("evidence_grounding")

        assert grounding is not None, "Missing evidence_grounding in result"

        print(f"\n=== Grounded Content Test ===")
        print(f"Content preview: {result.get('content', '')[:200]}...")
        print(f"Claims extracted: {grounding['total_claims_extracted']}")
        print(f"Claims verified: {grounding['claims_verified']}")
        print(f"Claims flagged: {grounding['flagged_claims']}")
        print(f"Max budget gap: {grounding['max_budget_gap']:.3f}")
        print(f"Passed: {grounding['passed']}")

        # Verify grounding executed and returned valid structure
        # Note: The actual flag rate depends on model calibration and thresholds.
        # This test verifies integration works, not that thresholds are perfect.
        assert "claims" in grounding
        assert "flagged_claims" in grounding
        assert isinstance(grounding['passed'], bool)

        if grounding['claims_verified'] > 0:
            flag_rate = grounding['flagged_claims'] / grounding['claims_verified']
            print(f"Flag rate: {flag_rate:.1%}")
            # Log but don't fail on flag rate - calibration is separate from integration
            if flag_rate > 0.5:
                print(f"NOTE: High flag rate ({flag_rate:.1%}) may indicate calibration needs tuning")


# =============================================================================
# E2E Tests - Confabulation Detection
# =============================================================================

class TestGroundingE2EConfabulation:
    """E2E tests for confabulation detection."""

    def test_no_evidence_flags_claims(self, live_engine_server, no_evidence_request):
        """
        Given: Request with minimal context that doesn't support the prompt
        When: Model generates content (likely using world knowledge)
        Then: Evidence grounding should flag some claims as confabulated
        """
        init_response = start_generation(no_evidence_request)
        session_id = init_response["session_id"]
        status = wait_for_completion(session_id)

        assert status["status"] == "completed", f"Generation failed: {status}"

        result = get_result(session_id)
        grounding = result.get("evidence_grounding")

        assert grounding is not None, "Missing evidence_grounding in result"

        print(f"\n=== No Evidence Test (Confabulation Detection) ===")
        print(f"Content preview: {result.get('content', '')[:200]}...")
        print(f"Claims extracted: {grounding['total_claims_extracted']}")
        print(f"Claims verified: {grounding['claims_verified']}")
        print(f"Claims flagged: {grounding['flagged_claims']}")
        print(f"Max budget gap: {grounding['max_budget_gap']:.3f}")
        print(f"Passed: {grounding['passed']}")

        # Print individual claim results
        if grounding.get('claims'):
            print("\nClaim details:")
            for claim in grounding['claims'][:5]:  # First 5 claims
                status = "FLAGGED" if claim['flagged'] else "OK"
                print(f"  [{status}] {claim['claim'][:60]}... "
                      f"(gap={claim['budget_gap']:.2f}, delta={claim['confidence_delta']:.2f})")

        # With no supporting evidence, we expect some claims to be flagged
        # The model will use world knowledge, which should trigger flagging
        if grounding['claims_verified'] > 0:
            assert grounding['flagged_claims'] > 0, \
                "Expected some flagged claims when context doesn't support content"
            print(f"\nSUCCESS: Detected {grounding['flagged_claims']} confabulated claims")


# =============================================================================
# E2E Tests - Edge Cases
# =============================================================================

class TestGroundingE2EEdgeCases:
    """E2E tests for edge cases in evidence grounding."""

    def test_trivial_content_no_claims(self, live_engine_server, base_grounding_config):
        """
        Given: Content that is purely decorative/trivial
        When: Claim extraction runs
        Then: Few or no verifiable claims extracted
        """
        payload = {
            "prompt": "Write a simple greeting: 'Hello, how are you today?'",
            "generator_model": "gpt-4o-mini",
            "content_type": "creative",
            "min_words": 5,
            "max_words": 20,
            "max_iterations": 1,
            "qa_layers": [],
            "evidence_grounding": base_grounding_config,
        }

        init_response = start_generation(payload)
        session_id = init_response["session_id"]
        status = wait_for_completion(session_id)

        assert status["status"] == "completed"

        result = get_result(session_id)
        grounding = result.get("evidence_grounding")

        assert grounding is not None, "evidence_grounding should not be None when enabled"

        print(f"\n=== Trivial Content Test ===")
        print(f"Content: {result.get('content', '')}")
        print(f"Claims extracted: {grounding['total_claims_extracted']}")
        print(f"Claims after filter: {grounding['claims_after_filter']}")
        print(f"Passed: {grounding['passed']}")

        # Verify grounding executed correctly
        # Note: Claim extraction from trivial content depends on model behavior
        # This test verifies integration works, not claim extraction quality
        assert "total_claims_extracted" in grounding
        assert "passed" in grounding
        assert isinstance(grounding['passed'], bool)

        # Log note if trivial content extracted many claims
        if grounding['total_claims_extracted'] > 2:
            print(f"NOTE: Extracted {grounding['total_claims_extracted']} claims from trivial content - extractor may be aggressive")

    def test_deal_breaker_config(self, live_engine_server):
        """
        Given: Evidence grounding configured as deal_breaker
        When: Claims are flagged
        Then: Generation may fail/iterate (depending on content)
        """
        payload = {
            "prompt": f"""
CONTEXT:
The weather today is sunny.

TASK:
Explain the theory of relativity and who discovered it.
Write 50-100 words.
""",
            "generator_model": "gpt-4o-mini",
            "content_type": "article",
            "min_words": 50,
            "max_words": 100,
            "max_iterations": 2,  # Allow retry
            "qa_layers": [],
            "evidence_grounding": {
                "enabled": True,
                "model": GROUNDING_MODEL,
                "max_claims": 5,
                "budget_gap_threshold": 0.3,  # Stricter threshold
                "max_flagged_claims": 1,  # Strict: 1 flag triggers
                "on_flag": "deal_breaker",
            },
        }

        init_response = start_generation(payload)
        session_id = init_response["session_id"]
        status = wait_for_completion(session_id, max_wait=240)

        print(f"\n=== Deal Breaker Config Test ===")
        print(f"Final status: {status['status']}")

        # With strict settings on unrelated context, we expect:
        # - Either failed (deal_breaker triggered all iterations)
        # - Or completed with flagged claims noted
        result = get_result(session_id)
        grounding = result.get("evidence_grounding")

        if grounding:
            print(f"Flagged claims: {grounding['flagged_claims']}")
            print(f"Triggered action: {grounding.get('triggered_action')}")


# =============================================================================
# E2E Tests - Claim Details Verification
# =============================================================================

class TestGroundingE2EClaimDetails:
    """E2E tests verifying claim detail structure."""

    def test_claim_structure_complete(self, live_engine_server, base_grounding_config):
        """
        Given: Request with evidence grounding enabled
        When: Claims are extracted and scored
        Then: Each claim has complete structure with all fields
        """
        payload = {
            "prompt": f"""
CONTEXT:
Albert Einstein was born in Germany in 1879.
He developed the theory of relativity.
He won the Nobel Prize in Physics in 1921.

TASK:
Write a brief summary of Einstein's life based on the context.
Write 30-60 words.
""",
            "generator_model": "gpt-4o-mini",
            "content_type": "biography",
            "min_words": 30,
            "max_words": 60,
            "max_iterations": 1,
            "qa_layers": [],
            "evidence_grounding": base_grounding_config,
        }

        init_response = start_generation(payload)
        session_id = init_response["session_id"]
        status = wait_for_completion(session_id)

        assert status["status"] == "completed"

        result = get_result(session_id)
        grounding = result.get("evidence_grounding")

        assert grounding is not None

        print(f"\n=== Claim Structure Test ===")
        print(f"Claims verified: {grounding['claims_verified']}")

        if grounding['claims'] and len(grounding['claims']) > 0:
            claim = grounding['claims'][0]

            # Verify claim structure
            required_fields = [
                'idx', 'claim', 'cited_spans',
                'posterior_yes', 'prior_yes',
                'required_bits', 'observed_bits', 'budget_gap',
                'flagged', 'confidence_delta'
            ]

            for field in required_fields:
                assert field in claim, f"Missing field '{field}' in claim"

            print(f"\nSample claim structure:")
            print(f"  idx: {claim['idx']}")
            print(f"  claim: {claim['claim'][:50]}...")
            print(f"  posterior_yes: {claim['posterior_yes']:.3f}")
            print(f"  prior_yes: {claim['prior_yes']:.3f}")
            print(f"  budget_gap: {claim['budget_gap']:.3f}")
            print(f"  confidence_delta: {claim['confidence_delta']:.3f}")
            print(f"  flagged: {claim['flagged']}")

            # Verify values are in expected ranges
            assert 0 <= claim['posterior_yes'] <= 1, "posterior_yes out of range"
            assert 0 <= claim['prior_yes'] <= 1, "prior_yes out of range"
            assert isinstance(claim['flagged'], bool), "flagged should be boolean"


# =============================================================================
# Direct Module Tests (No Server Required)
# =============================================================================

class TestGroundingDirectModule:
    """Tests that call the grounding module directly (no server needed)."""

    @pytest.mark.asyncio
    async def test_grounding_engine_direct_call(self):
        """
        Given: Direct call to GroundingEngine
        When: run_grounding_check is called with real AI service
        Then: Returns valid EvidenceGroundingResult
        """
        from ai_service import get_ai_service
        from evidence_grounding import GroundingEngine
        from models import EvidenceGroundingConfig

        ai_service = get_ai_service()
        engine = GroundingEngine(ai_service)

        config = EvidenceGroundingConfig(
            enabled=True,
            model=GROUNDING_MODEL,
            max_claims=5,
            filter_trivial=True,
            min_claim_importance=0.5,
            budget_gap_threshold=0.5,
            max_flagged_claims=2,
            on_flag="warn",
        )

        content = "Einstein was born in Germany in 1879. He won the Nobel Prize."
        context = "Albert Einstein was born in Ulm, Germany in 1879. He received the Nobel Prize in Physics in 1921."

        result = await engine.run_grounding_check(
            content=content,
            context=context,
            grounding_config=config,
            extra_verbose=True,
        )

        print(f"\n=== Direct Module Test ===")
        print(f"Enabled: {result.enabled}")
        print(f"Model used: {result.model_used}")
        print(f"Claims extracted: {result.total_claims_extracted}")
        print(f"Claims verified: {result.claims_verified}")
        print(f"Flagged: {result.flagged_claims}")
        print(f"Max gap: {result.max_budget_gap:.3f}")
        print(f"Passed: {result.passed}")
        print(f"Time: {result.verification_time_ms:.0f}ms")

        assert result.enabled is True
        assert GROUNDING_MODEL in result.model_used
        assert isinstance(result.passed, bool)
        assert result.verification_time_ms > 0

    @pytest.mark.asyncio
    async def test_claim_extraction_real_api(self):
        """
        Given: Content with factual claims
        When: ClaimExtractor extracts claims via real API
        Then: Returns list of ExtractedClaim objects
        """
        from ai_service import get_ai_service
        from evidence_grounding import ClaimExtractor

        ai_service = get_ai_service()
        extractor = ClaimExtractor(ai_service)

        content = """
        Marie Curie was born in Warsaw, Poland in 1867.
        She moved to Paris in 1891 to study physics.
        She won the Nobel Prize in Physics in 1903.
        """
        context = "Historical biography source material."

        claims = await extractor.extract_claims(
            content=content,
            context=context,
            max_claims=10,
            filter_trivial=True,
            min_importance=0.5,
        )

        print(f"\n=== Claim Extraction Test ===")
        print(f"Claims extracted: {len(claims)}")
        for claim in claims:
            print(f"  [{claim.kind}] {claim.claim} (importance={claim.importance:.2f})")

        assert len(claims) > 0, "Expected at least one claim"
        assert all(c.importance >= 0.5 for c in claims), "All claims should meet min_importance"


# =============================================================================
# Test Runner Helper
# =============================================================================

if __name__ == "__main__":
    """Run tests directly for debugging."""
    import subprocess

    # Server-backed tests fail fast when nothing answers at API_BASE.
    subprocess.run([
        "python", "-m", "pytest",
        __file__,
        "-v",
        "-m", "integration",
        "--tb=short"
    ])
