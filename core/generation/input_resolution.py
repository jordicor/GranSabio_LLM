"""Input-resolution helpers for generation requests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from fastapi import HTTPException

from models import ContentRequest, ImageData
from services.attachment_manager import (
    AttachmentError,
    AttachmentManager,
    AttachmentNotFoundError,
    AttachmentValidationError,
    ResolvedAttachment,
)


@dataclass
class GenerationInputResolution:
    attachment_manager: Optional[AttachmentManager]
    resolved_attachments: List[ResolvedAttachment]
    resolved_images: List[ImageData]
    preflight_context: List[Dict[str, Any]]
    preflight_image_info: Optional[Dict[str, Any]]


async def resolve_generation_inputs(
    request: ContentRequest,
    *,
    attachment_manager_factory: Any,
    config_obj: Any,
    resolve_images_for_generation_fn: Any,
    logger: Any,
) -> GenerationInputResolution:
    """Resolve context documents and image references for a generation request."""

    attachment_manager: Optional[AttachmentManager] = None
    resolved_attachments: List[ResolvedAttachment] = []
    preflight_context: List[Dict[str, Any]] = []

    if request.context_documents:
        if not request.username:
            raise HTTPException(status_code=400, detail="username is required when providing context_documents")
        attachment_manager = attachment_manager_factory()
        max_allowed = config_obj.ATTACHMENTS.max_files_per_request
        if len(request.context_documents) > max_allowed:
            raise HTTPException(status_code=400, detail=f"Maximum of {max_allowed} context documents are allowed per request")
        seen_upload_ids = set()
        for ref in request.context_documents:
            if ref.username != request.username:
                raise HTTPException(status_code=403, detail="Context document does not belong to the requesting user")
            if ref.upload_id in seen_upload_ids:
                raise HTTPException(status_code=400, detail=f"Duplicate context document: {ref.upload_id}")
            seen_upload_ids.add(ref.upload_id)
            try:
                resolved = attachment_manager.resolve_attachment(
                    username=request.username,
                    upload_id=ref.upload_id,
                )
            except AttachmentValidationError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except AttachmentNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except AttachmentError as exc:
                logger.exception("Unexpected attachment resolution error", exc_info=exc)
                raise HTTPException(status_code=500, detail="Unable to access attachment content") from exc

            resolved_attachments.append(resolved)
            preflight_context.append(attachment_manager.build_preflight_summary(resolved))

        logger.info(
            "Resolved %d context documents for user %s",
            len(resolved_attachments),
            request.username,
        )

    resolved_images: List[ImageData] = []
    if request.images:
        if not request.username:
            raise HTTPException(status_code=400, detail="username is required when providing images")

        if not attachment_manager:
            attachment_manager = attachment_manager_factory()

        max_images = config_obj.IMAGE.max_images_per_request
        if len(request.images) > max_images:
            raise HTTPException(
                status_code=400,
                detail=f"Maximum of {max_images} images allowed per request",
            )

        seen_image_ids = set()
        for img_ref in request.images:
            if img_ref.username != request.username:
                raise HTTPException(
                    status_code=403,
                    detail="Image does not belong to the requesting user",
                )
            if img_ref.upload_id in seen_image_ids:
                raise HTTPException(
                    status_code=400,
                    detail=f"Duplicate image reference: {img_ref.upload_id}",
                )
            seen_image_ids.add(img_ref.upload_id)

            try:
                resolved = attachment_manager.resolve_attachment(
                    username=request.username,
                    upload_id=img_ref.upload_id,
                )
                if not attachment_manager.is_image(resolved.record):
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"Attachment {img_ref.upload_id} is not an image "
                            f"(type: {resolved.record.mime_type})"
                        ),
                    )
            except AttachmentValidationError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except AttachmentNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except AttachmentError as exc:
                logger.exception("Unexpected image resolution error", exc_info=exc)
                raise HTTPException(status_code=500, detail="Unable to access image content") from exc

        try:
            resolved_images = await resolve_images_for_generation_fn(request, attachment_manager)
            logger.info(
                "Resolved %d images for vision-enabled generation, user %s",
                len(resolved_images),
                request.username,
            )
        except (AttachmentError, AttachmentNotFoundError, AttachmentValidationError) as exc:
            raise HTTPException(status_code=400, detail=f"Image processing failed: {exc}") from exc

    preflight_image_info: Optional[Dict[str, Any]] = None
    if resolved_images:
        try:
            generator_model_info = config_obj.get_model_info(request.generator_model)
            generator_capabilities = generator_model_info.get("capabilities", [])
            generator_supports_vision = "vision" in [
                capability.lower()
                for capability in generator_capabilities
                if isinstance(capability, str)
            ]
        except Exception:
            generator_supports_vision = False

        if not generator_supports_vision:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Model '{request.generator_model}' does not support vision/images. "
                    f"Request includes {len(resolved_images)} image(s). "
                    "Please use a vision-capable model (e.g., gpt-4o, claude-sonnet-4, "
                    "gemini-2.5-flash) or remove the images from the request."
                ),
            )

        preflight_image_info = {
            "count": len(resolved_images),
            "total_estimated_tokens": sum(
                image.estimated_tokens or 0 for image in resolved_images
            ),
            "filenames": [image.original_filename for image in resolved_images],
            "total_size_bytes": sum(image.size_bytes for image in resolved_images),
            "generator_supports_vision": generator_supports_vision,
            "detail_levels": list(set(
                image.detail for image in resolved_images if image.detail
            )) or ["auto"],
        }

    return GenerationInputResolution(
        attachment_manager=attachment_manager,
        resolved_attachments=resolved_attachments,
        resolved_images=resolved_images,
        preflight_context=preflight_context,
        preflight_image_info=preflight_image_info,
    )
