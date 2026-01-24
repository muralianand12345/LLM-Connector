from enum import Enum
from pydantic import BaseModel
from abc import ABC, abstractmethod
from typing import Union, Any, Optional, Literal, List, BinaryIO


class BatchStatus(str, Enum):
    """Status of a batch job."""

    VALIDATING = "validating"
    FAILED = "failed"
    IN_PROGRESS = "in_progress"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class BatchTimestamp(BaseModel):
    """Timestamps for various stages of a batch job."""

    created_at: str
    in_progress_at: Optional[str] = None
    cancelled_at: Optional[str] = None
    completed_at: Optional[str] = None
    expired_at: Optional[str] = None
    failed_at: Optional[str] = None
    finalized_at: Optional[str] = None


class BatchRequest(BaseModel):
    """Details of a batch job request."""

    id: str
    status: BatchStatus
    timestamps: BatchTimestamp
    completion_window: Literal["24h"]
    input_file_id: str
    output_file_id: Optional[str] = None
    error_file_id: Optional[str] = None
    endpoint: Optional[str] = "/v1/chat/completions"
    request_counts: Optional[dict] = None


class BatchResult(BaseModel):
    """Results of a completed batch job."""

    job_id: str
    output_file_id: Optional[str]
    records: List[dict]


class BatchProcess(ABC):
    """Abstract base class for batch processing API."""

    @abstractmethod
    def create(
        self,
        *,
        file: Union[str, bytes, BinaryIO],
        completion_window: Literal["24h"] = "24h",
        **kwargs: Any,
    ) -> BatchRequest:
        """Create a new batch job."""
        pass

    @abstractmethod
    def status(self, job_id: str, **kwargs: Any) -> BatchRequest:
        """Get the status of a batch job."""
        pass

    @abstractmethod
    def result(self, job_id: str, **kwargs: Any) -> BatchResult:
        """Get the results of a completed batch job."""
        pass

    @abstractmethod
    def cancel(self, job_id: str, **kwargs: Any) -> BatchRequest:
        """Cancel a batch job."""
        pass

    @abstractmethod
    def list(
        self, *, limit: int = 20, after: Optional[str] = None, **kwargs: Any
    ) -> List[BatchRequest]:
        """List batch jobs."""
        pass


class AsyncBatchProcess(ABC):
    """Abstract base class for async batch processing API."""

    @abstractmethod
    async def create(
        self,
        *,
        file: Union[str, bytes, BinaryIO],
        completion_window: Literal["24h"] = "24h",
        **kwargs: Any,
    ) -> BatchRequest:
        """Create a new batch job asynchronously."""
        pass

    @abstractmethod
    async def status(self, job_id: str, **kwargs: Any) -> BatchRequest:
        """Get the status of a batch job asynchronously."""
        pass

    @abstractmethod
    async def result(self, job_id: str, **kwargs: Any) -> BatchResult:
        """Get the results of a completed batch job asynchronously."""
        pass

    @abstractmethod
    async def cancel(self, job_id: str, **kwargs: Any) -> BatchRequest:
        """Cancel a batch job asynchronously."""
        pass

    @abstractmethod
    async def list(
        self, *, limit: int = 20, after: Optional[str] = None, **kwargs: Any
    ) -> List[BatchRequest]:
        """List batch jobs asynchronously."""
        pass
