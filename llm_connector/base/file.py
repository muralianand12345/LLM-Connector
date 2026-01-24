from pydantic import BaseModel
from abc import ABC, abstractmethod
from typing import Literal, Union, Optional, BinaryIO, List

PurposeType = Literal["fine-tune", "batch"]


class FileObject(BaseModel):
    """Metadata for a file object."""

    id: str
    filename: str
    purpose: PurposeType
    bytes: int
    created_at: int
    status: Optional[str] = None


class FileAPI(ABC):
    """Abstract base class for file operations API."""

    @abstractmethod
    def upload(self, *, file: Union[str, bytes, BinaryIO], purpose: PurposeType) -> str:
        """Upload a file and return the file ID."""
        pass

    @abstractmethod
    def retrieve(self, *, file_id: str) -> FileObject:
        """Retrieve file metadata by ID."""
        pass

    @abstractmethod
    def download(self, *, file_id: str) -> bytes:
        """Download file content by ID."""
        pass

    @abstractmethod
    def delete(self, *, file_id: str) -> None:
        """Delete a file by ID."""
        pass

    @abstractmethod
    def list(self, *, purpose: Optional[PurposeType] = None) -> List[FileObject]:
        """List all files, optionally filtered by purpose."""
        pass


class AsyncFileAPI(ABC):
    """Abstract base class for async file operations API."""

    @abstractmethod
    async def upload(
        self, *, file: Union[str, bytes, BinaryIO], purpose: PurposeType
    ) -> str:
        """Upload a file asynchronously and return the file ID."""
        pass

    @abstractmethod
    async def retrieve(self, *, file_id: str) -> FileObject:
        """Retrieve file metadata by ID asynchronously."""
        pass

    @abstractmethod
    async def download(self, *, file_id: str) -> bytes:
        """Download file content by ID asynchronously."""
        pass

    @abstractmethod
    async def delete(self, *, file_id: str) -> None:
        """Delete a file by ID asynchronously."""
        pass

    @abstractmethod
    async def list(self, *, purpose: Optional[PurposeType] = None) -> List[FileObject]:
        """List all files asynchronously, optionally filtered by purpose."""
        pass
