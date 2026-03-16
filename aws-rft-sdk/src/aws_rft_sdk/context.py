"""Context for RFT rollout metadata using contextvars for thread propagation.

The rft_handler decorator populates this context from the payload metadata.
The Strands model wrapper reads it to inject per-request headers.

Uses contextvars.ContextVar (not threading.local) because Strands Agent.__call__()
runs in a worker thread via ThreadPoolExecutor with contextvars.copy_context(),
which propagates ContextVar but not threading.local state.
"""

import contextvars
import uuid
from typing import Optional

_metadata_var: contextvars.ContextVar[Optional[dict]] = contextvars.ContextVar(
    'rft_metadata', default=None
)


class RFTContext:
    """Access the current RFT rollout context.

    Set by @rft_handler, read by wrap_model adapters to inject headers.

    The injected headers match the AgenticRFTRuntimeService API:
      - ``X-Rft-Job-Arn``: job ARN that identifies the Lego session
      - ``X-Trajectory-Id``: groups turns into a single trajectory
      - ``X-Span-Id``: unique ID for each turn within the trajectory
    """

    @staticmethod
    def get_headers() -> dict:
        """Return HTTP headers for the current rollout context.

        A new ``X-Span-Id`` is generated on every call so each inference
        turn gets a unique span within the trajectory.
        """
        metadata = _metadata_var.get()
        if metadata is None:
            return {}
        headers = {}
        # Support both snake_case (SDK convention) and camelCase (TLM convention)
        job_arn = metadata.get('job_arn') or metadata.get('jobId')
        trajectory_id = metadata.get('trajectory_id') or metadata.get('trajectoryId') or metadata.get('rolloutId')
        if job_arn:
            headers['X-Rft-Job-Arn'] = job_arn
        if trajectory_id:
            headers['X-Trajectory-Id'] = trajectory_id
            headers['X-Span-Id'] = str(uuid.uuid4())
        return headers

    @staticmethod
    def get_metadata() -> Optional[dict]:
        """Return the raw metadata dict, or None if not in an RFT context."""
        return _metadata_var.get()


def _set_metadata(metadata: dict):
    _metadata_var.set(metadata)


def _clear_metadata():
    _metadata_var.set(None)
