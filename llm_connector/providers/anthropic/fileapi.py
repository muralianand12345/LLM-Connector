from __future__ import annotations

from typing import Union, Optional, BinaryIO, List, TYPE_CHECKING

from ...base import FileAPI, AsyncFileAPI, FileObject, PurposeType
from ...exceptions import NotImplementedError

if TYPE_CHECKING:
    from anthropic import Anthropic
    from anthropic import AsyncAnthropic


class AnthropicFileAPI(FileAPI):
    """
    Anthropic File API implementation.

    Note: Anthropic does not support a traditional File API.

    For batch processing:
        Use connector.batch() which handles the Message Batches API directly.

    For document/image processing:
        Include content directly in messages using base64 encoding:

        ```python
        from llm_connector import UserMessage, ImageBlock, TextBlock, Role
        import base64

        with open("document.pdf", "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")

        message = UserMessage(
            role=Role.USER,
            content=[
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": data,
                    },
                },
                TextBlock(text="Please summarize this document."),
            ],
        )
        ```
    """

    def __init__(self, client: "Anthropic") -> None:
        self._client = client

    def upload(self, *, file: Union[str, bytes, BinaryIO], purpose: PurposeType) -> str:
        """
        Upload a file to Anthropic.

        Note: Anthropic does not support file uploads in the same way as OpenAI.

        For batch processing:
            Use connector.batch().create() which accepts requests directly.

        For documents/images:
            Include them as base64-encoded content directly in messages.

        Raises:
            FileError: Always raised as this operation is not supported.
        """
        raise NotImplementedError(
            "Anthropic does not support file uploads. "
            "For batch processing, use connector.batch().create() with requests directly. "
            "For documents/images, include them as base64-encoded content in messages."
        )

    def retrieve(self, *, file_id: str) -> FileObject:
        """
        Retrieve file metadata.

        Note: Anthropic does not support file storage/retrieval.

        Raises:
            FileError: Always raised as this operation is not supported.
        """
        raise NotImplementedError(
            "Anthropic does not support file retrieval. "
            "Anthropic does not store files - content is passed directly in API requests."
        )

    def download(self, *, file_id: str) -> bytes:
        """
        Download file content.

        Note: Anthropic does not support file storage/download.

        For batch results:
            Use connector.batch().result() to get batch processing results.

        Raises:
            FileError: Always raised as this operation is not supported.
        """
        raise NotImplementedError(
            "Anthropic does not support file downloads. "
            "For batch results, use connector.batch().result() instead."
        )

    def delete(self, *, file_id: str) -> None:
        """
        Delete a file.

        Note: Anthropic does not support file storage/deletion.

        Raises:
            FileError: Always raised as this operation is not supported.
        """
        raise NotImplementedError(
            "Anthropic does not support file deletion. "
            "Anthropic does not store files - content is passed directly in API requests."
        )

    def list(self, *, purpose: Optional[PurposeType] = None) -> List[FileObject]:
        """
        List files.

        Note: Anthropic does not support file storage/listing.

        For listing batches:
            Use connector.batch().list() to list batch jobs.

        Raises:
            FileError: Always raised as this operation is not supported.
        """
        raise NotImplementedError(
            "Anthropic does not support file listing. "
            "For batch jobs, use connector.batch().list() instead."
        )


class AnthropicAsyncFileAPI(AsyncFileAPI):
    """
    Anthropic Async File API implementation.

    Note: Anthropic does not support a traditional File API.
    See AnthropicFileAPI for details and alternatives.
    """

    def __init__(self, client: "AsyncAnthropic") -> None:
        self._client = client

    async def upload(
        self, *, file: Union[str, bytes, BinaryIO], purpose: PurposeType
    ) -> str:
        """
        Upload a file to Anthropic asynchronously.

        Note: Anthropic does not support file uploads.

        Raises:
            FileError: Always raised as this operation is not supported.
        """
        raise NotImplementedError(
            "Anthropic does not support file uploads. "
            "For batch processing, use connector.async_batch().create() with requests directly. "
            "For documents/images, include them as base64-encoded content in messages."
        )

    async def retrieve(self, *, file_id: str) -> FileObject:
        """
        Retrieve file metadata asynchronously.

        Note: Anthropic does not support file storage/retrieval.

        Raises:
            FileError: Always raised as this operation is not supported.
        """
        raise NotImplementedError(
            "Anthropic does not support file retrieval. "
            "Anthropic does not store files - content is passed directly in API requests."
        )

    async def download(self, *, file_id: str) -> bytes:
        """
        Download file content asynchronously.

        Note: Anthropic does not support file storage/download.

        Raises:
            FileError: Always raised as this operation is not supported.
        """
        raise NotImplementedError(
            "Anthropic does not support file downloads. "
            "For batch results, use connector.async_batch().result() instead."
        )

    async def delete(self, *, file_id: str) -> None:
        """
        Delete a file asynchronously.

        Note: Anthropic does not support file storage/deletion.

        Raises:
            FileError: Always raised as this operation is not supported.
        """
        raise NotImplementedError(
            "Anthropic does not support file deletion. "
            "Anthropic does not store files - content is passed directly in API requests."
        )

    async def list(self, *, purpose: Optional[PurposeType] = None) -> List[FileObject]:
        """
        List files asynchronously.

        Note: Anthropic does not support file storage/listing.

        Raises:
            FileError: Always raised as this operation is not supported.
        """
        raise NotImplementedError(
            "Anthropic does not support file listing. "
            "For batch jobs, use connector.async_batch().list() instead."
        )
