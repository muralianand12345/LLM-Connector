from __future__ import annotations

import json
from typing import Union, Any, Optional, Literal, List, BinaryIO, TYPE_CHECKING

from ...exceptions import BatchError, AuthenticationError, APIError
from ...base import (
    BatchProcess,
    AsyncBatchProcess,
    BatchRequest,
    BatchResult,
    BatchStatus,
    BatchTimestamp,
)

if TYPE_CHECKING:
    from groq import Groq
    from groq import AsyncGroq


class GroqBatchProcess(BatchProcess):
    """Groq Batch API implementation."""

    def __init__(self, client: "Groq") -> None:
        self._client = client

    def create(
        self,
        *,
        file: Union[str, bytes, BinaryIO],
        completion_window: Literal["24h"] = "24h",
        endpoint: str = "/v1/chat/completions",
        metadata: Optional[dict] = None,
        **kwargs: Any,
    ) -> BatchRequest:
        """
        Create a new batch job.

        Args:
            file: JSONL file (path, bytes, or file-like object) with batch requests
            completion_window: Time window for batch completion ('24h')
            endpoint: API endpoint for the batch requests
            metadata: Optional metadata for the batch
            **kwargs: Additional arguments

        Returns:
            BatchRequest with job details
        """
        try:
            # Upload file first
            if isinstance(file, str):
                with open(file, "rb") as f:
                    file_response = self._client.files.create(file=f, purpose="batch")
            elif isinstance(file, bytes):
                file_response = self._client.files.create(
                    file=("batch.jsonl", file), purpose="batch"
                )
            else:
                file_response = self._client.files.create(file=file, purpose="batch")

            # Create batch job
            batch_kwargs = {
                "input_file_id": file_response.id,
                "endpoint": endpoint,
                "completion_window": completion_window,
            }
            if metadata:
                batch_kwargs["metadata"] = metadata

            response = self._client.batches.create(**batch_kwargs)
            return self._to_batch_request(response)

        except Exception as e:
            raise self._handle_exception(e)

    def status(self, job_id: str, **kwargs: Any) -> BatchRequest:
        """
        Get the status of a batch job.

        Args:
            job_id: The batch job ID

        Returns:
            BatchRequest with current status
        """
        try:
            response = self._client.batches.retrieve(job_id)
            return self._to_batch_request(response)
        except Exception as e:
            raise self._handle_exception(e)

    def result(self, job_id: str, **kwargs: Any) -> BatchResult:
        """
        Get the results of a completed batch job.

        Args:
            job_id: The batch job ID

        Returns:
            BatchResult with output records
        """
        try:
            batch = self._client.batches.retrieve(job_id)
            if batch.status != "completed":
                raise BatchError(
                    f"Batch job is not completed. Current status: {batch.status}"
                )

            if not batch.output_file_id:
                raise BatchError("Batch job has no output file")

            # Download output file
            content = self._client.files.content(batch.output_file_id)
            lines = content.text.strip().split("\n")

            records = []
            for line in lines:
                if line:
                    records.append(json.loads(line))

            return BatchResult(
                job_id=job_id,
                output_file_id=batch.output_file_id,
                records=records,
            )

        except BatchError:
            raise
        except Exception as e:
            raise self._handle_exception(e)

    def cancel(self, job_id: str, **kwargs: Any) -> BatchRequest:
        """
        Cancel a batch job.

        Args:
            job_id: The batch job ID

        Returns:
            BatchRequest with updated status
        """
        try:
            response = self._client.batches.cancel(job_id)
            return self._to_batch_request(response)
        except Exception as e:
            raise self._handle_exception(e)

    def list(
        self, *, limit: int = 20, after: Optional[str] = None, **kwargs: Any
    ) -> List[BatchRequest]:
        """
        List batch jobs.

        Args:
            limit: Maximum number of jobs to return
            after: Cursor for pagination

        Returns:
            List of BatchRequest objects
        """
        try:
            list_kwargs = {"limit": limit}
            if after:
                list_kwargs["after"] = after

            response = self._client.batches.list(**list_kwargs)

            return [self._to_batch_request(batch) for batch in response.data]

        except Exception as e:
            raise self._handle_exception(e)

    def _to_batch_request(self, response) -> BatchRequest:
        """Convert Groq batch response to BatchRequest."""
        status_map = {
            "validating": BatchStatus.VALIDATING,
            "failed": BatchStatus.FAILED,
            "in_progress": BatchStatus.IN_PROGRESS,
            "finalizing": BatchStatus.FINALIZING,
            "completed": BatchStatus.COMPLETED,
            "expired": BatchStatus.EXPIRED,
            "cancelled": BatchStatus.CANCELLED,
            "cancelling": BatchStatus.CANCELLED,
        }

        timestamps = BatchTimestamp(
            created_at=self._timestamp_to_str(response.created_at),
            in_progress_at=self._timestamp_to_str(response.in_progress_at),
            cancelled_at=self._timestamp_to_str(response.cancelled_at),
            completed_at=self._timestamp_to_str(response.completed_at),
            expired_at=self._timestamp_to_str(response.expired_at),
            failed_at=self._timestamp_to_str(response.failed_at),
            finalized_at=self._timestamp_to_str(response.finalizing_at),
        )

        return BatchRequest(
            id=response.id,
            status=status_map.get(response.status, BatchStatus.FAILED),
            timestamps=timestamps,
            completion_window=response.completion_window,
            input_file_id=response.input_file_id,
            output_file_id=response.output_file_id,
            error_file_id=response.error_file_id,
            endpoint=response.endpoint,
            request_counts=(
                response.request_counts.model_dump()
                if response.request_counts
                else None
            ),
        )

    def _timestamp_to_str(self, timestamp: Optional[int]) -> Optional[str]:
        """Convert Unix timestamp to ISO string."""
        if timestamp is None:
            return None
        from datetime import datetime, timezone

        return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()

    def _handle_exception(self, e: Exception) -> Exception:
        """Convert Groq exceptions to our custom exceptions."""
        try:
            import groq
        except ImportError:
            return BatchError(str(e))

        if isinstance(e, groq.AuthenticationError):
            return AuthenticationError(str(e))
        elif isinstance(e, groq.NotFoundError):
            return BatchError(f"Batch job not found: {e}")
        elif isinstance(e, groq.APIError):
            return APIError(str(e), status_code=getattr(e, "status_code", None))
        else:
            return BatchError(str(e))


class GroqAsyncBatchProcess(AsyncBatchProcess):
    """Groq Async Batch API implementation."""

    def __init__(self, client: "AsyncGroq") -> None:
        self._client = client

    async def create(
        self,
        *,
        file: Union[str, bytes, BinaryIO],
        completion_window: Literal["24h"] = "24h",
        endpoint: str = "/v1/chat/completions",
        metadata: Optional[dict] = None,
        **kwargs: Any,
    ) -> BatchRequest:
        """
        Create a new batch job asynchronously.

        Args:
            file: JSONL file (path, bytes, or file-like object) with batch requests
            completion_window: Time window for batch completion ('24h')
            endpoint: API endpoint for the batch requests
            metadata: Optional metadata for the batch
            **kwargs: Additional arguments

        Returns:
            BatchRequest with job details
        """
        try:
            # Upload file first
            if isinstance(file, str):
                with open(file, "rb") as f:
                    file_response = await self._client.files.create(
                        file=f, purpose="batch"
                    )
            elif isinstance(file, bytes):
                file_response = await self._client.files.create(
                    file=("batch.jsonl", file), purpose="batch"
                )
            else:
                file_response = await self._client.files.create(
                    file=file, purpose="batch"
                )

            # Create batch job
            batch_kwargs = {
                "input_file_id": file_response.id,
                "endpoint": endpoint,
                "completion_window": completion_window,
            }
            if metadata:
                batch_kwargs["metadata"] = metadata

            response = await self._client.batches.create(**batch_kwargs)
            return self._to_batch_request(response)

        except Exception as e:
            raise self._handle_exception(e)

    async def status(self, job_id: str, **kwargs: Any) -> BatchRequest:
        """
        Get the status of a batch job asynchronously.

        Args:
            job_id: The batch job ID

        Returns:
            BatchRequest with current status
        """
        try:
            response = await self._client.batches.retrieve(job_id)
            return self._to_batch_request(response)
        except Exception as e:
            raise self._handle_exception(e)

    async def result(self, job_id: str, **kwargs: Any) -> BatchResult:
        """
        Get the results of a completed batch job asynchronously.

        Args:
            job_id: The batch job ID

        Returns:
            BatchResult with output records
        """
        try:
            batch = await self._client.batches.retrieve(job_id)
            if batch.status != "completed":
                raise BatchError(
                    f"Batch job is not completed. Current status: {batch.status}"
                )

            if not batch.output_file_id:
                raise BatchError("Batch job has no output file")

            # Download output file
            content = await self._client.files.content(batch.output_file_id)
            lines = content.text.strip().split("\n")

            records = []
            for line in lines:
                if line:
                    records.append(json.loads(line))

            return BatchResult(
                job_id=job_id,
                output_file_id=batch.output_file_id,
                records=records,
            )

        except BatchError:
            raise
        except Exception as e:
            raise self._handle_exception(e)

    async def cancel(self, job_id: str, **kwargs: Any) -> BatchRequest:
        """
        Cancel a batch job asynchronously.

        Args:
            job_id: The batch job ID

        Returns:
            BatchRequest with updated status
        """
        try:
            response = await self._client.batches.cancel(job_id)
            return self._to_batch_request(response)
        except Exception as e:
            raise self._handle_exception(e)

    async def list(
        self, *, limit: int = 20, after: Optional[str] = None, **kwargs: Any
    ) -> List[BatchRequest]:
        """
        List batch jobs asynchronously.

        Args:
            limit: Maximum number of jobs to return
            after: Cursor for pagination

        Returns:
            List of BatchRequest objects
        """
        try:
            list_kwargs = {"limit": limit}
            if after:
                list_kwargs["after"] = after

            response = await self._client.batches.list(**list_kwargs)

            return [self._to_batch_request(batch) for batch in response.data]

        except Exception as e:
            raise self._handle_exception(e)

    def _to_batch_request(self, response) -> BatchRequest:
        """Convert Groq batch response to BatchRequest."""
        status_map = {
            "validating": BatchStatus.VALIDATING,
            "failed": BatchStatus.FAILED,
            "in_progress": BatchStatus.IN_PROGRESS,
            "finalizing": BatchStatus.FINALIZING,
            "completed": BatchStatus.COMPLETED,
            "expired": BatchStatus.EXPIRED,
            "cancelled": BatchStatus.CANCELLED,
            "cancelling": BatchStatus.CANCELLED,
        }

        timestamps = BatchTimestamp(
            created_at=self._timestamp_to_str(response.created_at),
            in_progress_at=self._timestamp_to_str(response.in_progress_at),
            cancelled_at=self._timestamp_to_str(response.cancelled_at),
            completed_at=self._timestamp_to_str(response.completed_at),
            expired_at=self._timestamp_to_str(response.expired_at),
            failed_at=self._timestamp_to_str(response.failed_at),
            finalized_at=self._timestamp_to_str(response.finalizing_at),
        )

        return BatchRequest(
            id=response.id,
            status=status_map.get(response.status, BatchStatus.FAILED),
            timestamps=timestamps,
            completion_window=response.completion_window,
            input_file_id=response.input_file_id,
            output_file_id=response.output_file_id,
            error_file_id=response.error_file_id,
            endpoint=response.endpoint,
            request_counts=(
                response.request_counts.model_dump()
                if response.request_counts
                else None
            ),
        )

    def _timestamp_to_str(self, timestamp: Optional[int]) -> Optional[str]:
        """Convert Unix timestamp to ISO string."""
        if timestamp is None:
            return None
        from datetime import datetime, timezone

        return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()

    def _handle_exception(self, e: Exception) -> Exception:
        """Convert Groq exceptions to our custom exceptions."""
        try:
            import groq
        except ImportError:
            return BatchError(str(e))

        if isinstance(e, groq.AuthenticationError):
            return AuthenticationError(str(e))
        elif isinstance(e, groq.NotFoundError):
            return BatchError(f"Batch job not found: {e}")
        elif isinstance(e, groq.APIError):
            return APIError(str(e), status_code=getattr(e, "status_code", None))
        else:
            return BatchError(str(e))
