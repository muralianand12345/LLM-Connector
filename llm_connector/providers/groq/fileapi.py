"""
Groq File API Implementation

Groq's File API is OpenAI-compatible and supports:
- JSONL file format
- Max 50,000 lines per file
- Max 200MB file size
- Purpose: "batch" for batch processing
"""

from __future__ import annotations

from typing import Union, Optional, BinaryIO, List, TYPE_CHECKING

from ...base import FileAPI, AsyncFileAPI, FileObject, PurposeType
from ...exceptions import FileError, AuthenticationError, APIError

if TYPE_CHECKING:
    from groq import Groq
    from groq import AsyncGroq


class GroqFileAPI(FileAPI):
    """Groq File API implementation."""

    def __init__(self, client: "Groq") -> None:
        self._client = client

    def upload(self, *, file: Union[str, bytes, BinaryIO], purpose: PurposeType) -> str:
        """
        Upload a file to Groq.

        Args:
            file: File path, bytes, or file-like object
            purpose: Purpose of the file ('batch' for batch processing)

        Returns:
            The file ID

        Note:
            Groq currently supports JSONL files with:
            - Max 50,000 lines
            - Max 200MB size
        """
        try:
            if isinstance(file, str):
                with open(file, "rb") as f:
                    response = self._client.files.create(file=f, purpose=purpose)
            elif isinstance(file, bytes):
                response = self._client.files.create(
                    file=("file.jsonl", file), purpose=purpose
                )
            else:
                response = self._client.files.create(file=file, purpose=purpose)

            return response.id

        except Exception as e:
            raise self._handle_exception(e)

    def retrieve(self, *, file_id: str) -> FileObject:
        """
        Retrieve file metadata.

        Args:
            file_id: The ID of the file

        Returns:
            FileObject with file metadata
        """
        try:
            response = self._client.files.retrieve(file_id)
            return FileObject(
                id=response.id,
                filename=response.filename,
                purpose=response.purpose,
                bytes=response.bytes,
                created_at=response.created_at,
                status=response.status if hasattr(response, "status") else None,
            )
        except Exception as e:
            raise self._handle_exception(e)

    def download(self, *, file_id: str) -> bytes:
        """
        Download file content.

        Args:
            file_id: The ID of the file

        Returns:
            File content as bytes
        """
        try:
            response = self._client.files.content(file_id)
            return response.content
        except Exception as e:
            raise self._handle_exception(e)

    def delete(self, *, file_id: str) -> None:
        """
        Delete a file.

        Args:
            file_id: The ID of the file to delete
        """
        try:
            self._client.files.delete(file_id)
        except Exception as e:
            raise self._handle_exception(e)

    def list(self, *, purpose: Optional[PurposeType] = None) -> List[FileObject]:
        """
        List files.

        Args:
            purpose: Optional filter by purpose

        Returns:
            List of FileObject instances
        """
        try:
            kwargs = {}
            if purpose:
                kwargs["purpose"] = purpose

            response = self._client.files.list(**kwargs)

            return [
                FileObject(
                    id=f.id,
                    filename=f.filename,
                    purpose=f.purpose,
                    bytes=f.bytes,
                    created_at=f.created_at,
                    status=f.status if hasattr(f, "status") else None,
                )
                for f in response.data
            ]
        except Exception as e:
            raise self._handle_exception(e)

    def _handle_exception(self, e: Exception) -> Exception:
        """Convert Groq exceptions to our custom exceptions."""
        try:
            import groq
        except ImportError:
            return FileError(str(e))

        if isinstance(e, groq.AuthenticationError):
            return AuthenticationError(str(e))
        elif isinstance(e, groq.NotFoundError):
            return FileError(f"File not found: {e}")
        elif isinstance(e, groq.APIError):
            return APIError(str(e), status_code=getattr(e, "status_code", None))
        else:
            return FileError(str(e))


class GroqAsyncFileAPI(AsyncFileAPI):
    """Groq Async File API implementation."""

    def __init__(self, client: "AsyncGroq") -> None:
        self._client = client

    async def upload(
        self, *, file: Union[str, bytes, BinaryIO], purpose: PurposeType
    ) -> str:
        """
        Upload a file to Groq asynchronously.

        Args:
            file: File path, bytes, or file-like object
            purpose: Purpose of the file ('batch' for batch processing)

        Returns:
            The file ID

        Note:
            Groq currently supports JSONL files with:
            - Max 50,000 lines
            - Max 200MB size
        """
        try:
            if isinstance(file, str):
                with open(file, "rb") as f:
                    response = await self._client.files.create(file=f, purpose=purpose)
            elif isinstance(file, bytes):
                response = await self._client.files.create(
                    file=("file.jsonl", file), purpose=purpose
                )
            else:
                response = await self._client.files.create(file=file, purpose=purpose)

            return response.id

        except Exception as e:
            raise self._handle_exception(e)

    async def retrieve(self, *, file_id: str) -> FileObject:
        """
        Retrieve file metadata asynchronously.

        Args:
            file_id: The ID of the file

        Returns:
            FileObject with file metadata
        """
        try:
            response = await self._client.files.retrieve(file_id)
            return FileObject(
                id=response.id,
                filename=response.filename,
                purpose=response.purpose,
                bytes=response.bytes,
                created_at=response.created_at,
                status=response.status if hasattr(response, "status") else None,
            )
        except Exception as e:
            raise self._handle_exception(e)

    async def download(self, *, file_id: str) -> bytes:
        """
        Download file content asynchronously.

        Args:
            file_id: The ID of the file

        Returns:
            File content as bytes
        """
        try:
            response = await self._client.files.content(file_id)
            return response.content
        except Exception as e:
            raise self._handle_exception(e)

    async def delete(self, *, file_id: str) -> None:
        """
        Delete a file asynchronously.

        Args:
            file_id: The ID of the file to delete
        """
        try:
            await self._client.files.delete(file_id)
        except Exception as e:
            raise self._handle_exception(e)

    async def list(self, *, purpose: Optional[PurposeType] = None) -> List[FileObject]:
        """
        List files asynchronously.

        Args:
            purpose: Optional filter by purpose

        Returns:
            List of FileObject instances
        """
        try:
            kwargs = {}
            if purpose:
                kwargs["purpose"] = purpose

            response = await self._client.files.list(**kwargs)

            return [
                FileObject(
                    id=f.id,
                    filename=f.filename,
                    purpose=f.purpose,
                    bytes=f.bytes,
                    created_at=f.created_at,
                    status=f.status if hasattr(f, "status") else None,
                )
                for f in response.data
            ]
        except Exception as e:
            raise self._handle_exception(e)

    def _handle_exception(self, e: Exception) -> Exception:
        """Convert Groq exceptions to our custom exceptions."""
        try:
            import groq
        except ImportError:
            return FileError(str(e))

        if isinstance(e, groq.AuthenticationError):
            return AuthenticationError(str(e))
        elif isinstance(e, groq.NotFoundError):
            return FileError(f"File not found: {e}")
        elif isinstance(e, groq.APIError):
            return APIError(str(e), status_code=getattr(e, "status_code", None))
        else:
            return FileError(str(e))
