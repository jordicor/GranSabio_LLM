"""
Integration tests for attachment workflows.

Tests the complete attachment lifecycle:
- File upload (multipart, base64, URL)
- Metadata retrieval
- Attachment usage in generation
- Rate limiting
- Error scenarios
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import base64
import io


# Note: Fixtures inherited from conftest.py:
# - bypass_ip_filter (autouse)
# - base_test_client
# - valid_generation_request


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_text_file():
    """Create a sample text file for upload testing."""
    content = b"This is a sample text file for testing attachments."
    return io.BytesIO(content)


@pytest.fixture
def sample_base64_content():
    """Sample base64 encoded content."""
    content = b"This is base64 encoded test content."
    return base64.b64encode(content).decode('utf-8')


@pytest.fixture
def mock_attachment_record():
    """Create a real AttachmentRecord for testing."""
    from services.attachment_manager import AttachmentRecord

    return AttachmentRecord(
        upload_id="test-upload-123",
        origin="multipart",
        intended_usage="context",
        original_filename="test.txt",
        stored_filename="abc123.txt",
        mime_type="text/plain",
        size_bytes=1024,
        declared_size=None,
        declared_mime=None,
        detected_mime=None,
        sha256="abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
        metadata_signature="valid-signature-hash",
        original_url=None,
        created_at="2025-01-07T12:00:00Z",
        hash_prefix1="abc",
        hash_prefix2="1234",
        user_hash="a" * 40,
        storage_path="uploads/testuser/test.txt",
        metadata_path="uploads/testuser/test.json",
    )


@pytest.fixture
def mock_attachment_manager(mock_attachment_record):
    """Mock attachment manager for testing."""
    manager = MagicMock()

    async def mock_store_upload(*args, **kwargs):
        return mock_attachment_record

    async def mock_store_bytes(*args, **kwargs):
        return mock_attachment_record

    async def mock_store_from_url(*args, **kwargs):
        return mock_attachment_record

    manager.store_upload = AsyncMock(side_effect=mock_store_upload)
    manager.store_bytes = AsyncMock(side_effect=mock_store_bytes)
    manager.store_from_url = AsyncMock(side_effect=mock_store_from_url)
    manager.get_metadata = MagicMock(return_value=mock_attachment_record)
    manager.settings = MagicMock()
    manager.settings.max_size_bytes = 10 * 1024 * 1024  # 10MB

    @staticmethod
    def decode_base64_payload(content, max_decoded_size=None):
        return base64.b64decode(content)

    manager.decode_base64_payload = decode_base64_payload

    return manager


@pytest.fixture
def mock_rate_limiter():
    """Mock rate limiter that always passes."""
    limiter = MagicMock()
    limiter.check = AsyncMock()
    limiter.check_dual = AsyncMock()
    return limiter


@pytest.fixture
def attachment_test_client(mock_attachment_manager, mock_rate_limiter, mock_ai_service, mock_preflight_proceed):
    """Test client with mocked attachment services."""
    with patch('attachments_router.get_attachment_manager', return_value=mock_attachment_manager), \
         patch('attachments_router.get_rate_limiter', return_value=mock_rate_limiter), \
         patch('attachments_router._attachment_manager', mock_attachment_manager), \
         patch('config.config.PEPPER', 'test-pepper-secret'), \
         patch('ai_service.get_ai_service', return_value=mock_ai_service), \
         patch('core.app_state._ensure_services'), \
         patch('preflight_validator.run_preflight_validation', AsyncMock(return_value=mock_preflight_proceed)):
        from main import app
        from fastapi.testclient import TestClient
        client = TestClient(app, headers={"X-Forwarded-For": "127.0.0.1"})
        yield client


# ============================================================================
# Multipart Upload Tests
# ============================================================================

class TestMultipartUpload:
    """Tests for multipart/form-data file uploads."""

    def test_upload_attachment_success(self, attachment_test_client, sample_text_file):
        """
        Given: Valid file and username
        When: POST /attachments
        Then: Returns upload_id and metadata
        """
        response = attachment_test_client.post(
            "/attachments",
            data={"username": "testuser", "intended_usage": "context"},
            files={"file": ("test.txt", sample_text_file, "text/plain")}
        )
        assert response.status_code == 200
        data = response.json()
        assert "upload_id" in data
        assert "metadata" in data

    def test_upload_attachment_missing_username_returns_422(self, attachment_test_client, sample_text_file):
        """
        Given: File without username
        When: POST /attachments
        Then: Returns 422 validation error
        """
        response = attachment_test_client.post(
            "/attachments",
            files={"file": ("test.txt", sample_text_file, "text/plain")}
        )
        assert response.status_code == 422

    def test_upload_attachment_missing_file_returns_422(self, attachment_test_client):
        """
        Given: Username without file
        When: POST /attachments
        Then: Returns 422 validation error
        """
        response = attachment_test_client.post(
            "/attachments",
            data={"username": "testuser"}
        )
        assert response.status_code == 422

    def test_upload_attachment_with_intended_usage(self, attachment_test_client, sample_text_file):
        """
        Given: File with custom intended_usage
        When: POST /attachments
        Then: Processes successfully
        """
        response = attachment_test_client.post(
            "/attachments",
            data={"username": "testuser", "intended_usage": "reference"},
            files={"file": ("test.txt", sample_text_file, "text/plain")}
        )
        assert response.status_code == 200


# ============================================================================
# Base64 Upload Tests
# ============================================================================

class TestBase64Upload:
    """Tests for base64 encoded file uploads."""

    def test_upload_base64_attachment_success(self, attachment_test_client, sample_base64_content):
        """
        Given: Valid base64 content
        When: POST /attachments/base64
        Then: Returns upload_id and metadata
        """
        response = attachment_test_client.post(
            "/attachments/base64",
            json={
                "username": "testuser",
                "filename": "test.txt",
                "content_base64": sample_base64_content,
                "intended_usage": "context"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "upload_id" in data
        assert "metadata" in data

    def test_upload_base64_missing_username_returns_422(self, attachment_test_client, sample_base64_content):
        """
        Given: Base64 content without username
        When: POST /attachments/base64
        Then: Returns 422 validation error
        """
        response = attachment_test_client.post(
            "/attachments/base64",
            json={
                "filename": "test.txt",
                "content_base64": sample_base64_content
            }
        )
        assert response.status_code == 422

    def test_upload_base64_missing_filename_returns_422(self, attachment_test_client, sample_base64_content):
        """
        Given: Base64 content without filename
        When: POST /attachments/base64
        Then: Returns 422 validation error
        """
        response = attachment_test_client.post(
            "/attachments/base64",
            json={
                "username": "testuser",
                "content_base64": sample_base64_content
            }
        )
        assert response.status_code == 422

    def test_upload_base64_missing_content_returns_422(self, attachment_test_client):
        """
        Given: Request without base64 content
        When: POST /attachments/base64
        Then: Returns 422 validation error
        """
        response = attachment_test_client.post(
            "/attachments/base64",
            json={
                "username": "testuser",
                "filename": "test.txt"
            }
        )
        assert response.status_code == 422


# ============================================================================
# URL Upload Tests
# ============================================================================

class TestURLUpload:
    """Tests for URL-based file uploads."""

    def test_upload_url_attachment_success(self, attachment_test_client):
        """
        Given: Valid URL
        When: POST /attachments/url
        Then: Returns upload_id and metadata
        """
        response = attachment_test_client.post(
            "/attachments/url",
            json={
                "username": "testuser",
                "url": "https://example.com/test.txt",
                "intended_usage": "context"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "upload_id" in data
        assert "metadata" in data

    def test_upload_url_missing_username_returns_422(self, attachment_test_client):
        """
        Given: URL without username
        When: POST /attachments/url
        Then: Returns 422 validation error
        """
        response = attachment_test_client.post(
            "/attachments/url",
            json={
                "url": "https://example.com/test.txt"
            }
        )
        assert response.status_code == 422

    def test_upload_url_missing_url_returns_422(self, attachment_test_client):
        """
        Given: Request without URL
        When: POST /attachments/url
        Then: Returns 422 validation error
        """
        response = attachment_test_client.post(
            "/attachments/url",
            json={
                "username": "testuser"
            }
        )
        assert response.status_code == 422


# ============================================================================
# Metadata Retrieval Tests
# ============================================================================

class TestMetadataRetrieval:
    """Tests for attachment metadata retrieval."""

    def test_get_metadata_success(self, attachment_test_client):
        """
        Given: Valid upload_id and username
        When: GET /attachments/{upload_id}
        Then: Returns attachment metadata
        """
        response = attachment_test_client.get(
            "/attachments/test-upload-123",
            params={"username": "testuser"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "upload_id" in data

    def test_get_metadata_missing_username_returns_422(self, attachment_test_client):
        """
        Given: upload_id without username
        When: GET /attachments/{upload_id}
        Then: Returns 422 validation error
        """
        response = attachment_test_client.get("/attachments/test-upload-123")
        assert response.status_code == 422


# ============================================================================
# Attachment + Generation Integration Tests
# ============================================================================

class TestAttachmentGeneration:
    """Tests for using attachments in generation requests."""

    def test_generate_with_context_documents_requires_username(self, base_test_client):
        """
        Given: context_documents without username
        When: POST /generate
        Then: Returns 400
        """
        response = base_test_client.post(
            "/generate",
            json={
                "prompt": "Test prompt for generation with context documents.",
                "generator_model": "gpt-4o",
                "context_documents": [
                    {"username": "testuser", "upload_id": "test-upload-123"}
                ]
                # Missing username at request level
            }
        )
        assert response.status_code == 400

    def test_generate_context_document_wrong_user_returns_403(self, base_test_client):
        """
        Given: context_document belonging to different user
        When: POST /generate
        Then: Returns 403
        """
        response = base_test_client.post(
            "/generate",
            json={
                "prompt": "Test prompt for generation with wrong user context.",
                "generator_model": "gpt-4o",
                "username": "user1",
                "context_documents": [
                    {"username": "user2", "upload_id": "test-upload-123"}  # Different user
                ]
            }
        )
        assert response.status_code == 403
