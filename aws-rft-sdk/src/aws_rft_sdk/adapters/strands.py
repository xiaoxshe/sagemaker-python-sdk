"""Strands model adapter -- wraps a Strands model to inject RFT headers.

Usage::

    from aws_rft_sdk.adapters.strands import wrap_model
    from strands.models.openai import OpenAIModel

    model = OpenAIModel(
        client_args={"api_key": key, "base_url": endpoint},
        model_id="my-model",
    )
    model = wrap_model(model)  # Now injects X-RFT-* headers on every call

Injects headers via client_args["default_headers"] because Strands
OpenAIModel.stream() does not support extra_headers kwarg. OpenAIModel
creates a new client from client_args per request.
"""

import logging
from typing import Any

from aws_rft_sdk.context import RFTContext

logger = logging.getLogger(__name__)


def wrap_model(model: Any) -> Any:
    """Wrap a Strands model to automatically inject RFT training headers.

    The wrapper reads the current rollout context (populated by ``@rft_handler``)
    and adds ``X-RFT-*`` headers to every inference request so the training
    inference endpoint can correlate requests with rollouts.

    Args:
        model: A Strands model instance (e.g., ``OpenAIModel``).

    Returns:
        A wrapped model that transparently injects RFT headers.
    """
    return _RFTModelWrapper(model)


class _RFTModelWrapper:
    """Transparent proxy that injects RFT headers into Strands model calls.

    Delegates all attribute access to the inner model so it quacks like
    the original. Intercepts ``stream()`` to inject headers via
    ``client_args["default_headers"]``.
    """

    def __init__(self, inner_model: Any):
        object.__setattr__(self, '_inner', inner_model)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def __setattr__(self, name: str, value: Any):
        if name == '_inner':
            object.__setattr__(self, name, value)
            return
        setattr(self._inner, name, value)

    async def stream(self, *args: Any, **kwargs: Any) -> Any:
        """Intercept stream() to inject RFT headers via client_args default_headers."""
        rft_headers = RFTContext.get_headers()
        logger.info('wrap_model.stream() called, rft_headers=%s, has_client_args=%s',
                     rft_headers, hasattr(self._inner, 'client_args'))
        if rft_headers:
            # Inject via client_args["default_headers"] since OpenAIModel
            # creates a new OpenAI client from client_args on each request
            if hasattr(self._inner, 'client_args'):
                existing = self._inner.client_args.get('default_headers') or {}
                existing.update(rft_headers)
                self._inner.client_args['default_headers'] = existing
                logger.info('Injected RFT headers into client_args: %s', list(rft_headers.keys()))
            else:
                logger.warning('No client_args on inner model, cannot inject headers')
        else:
            logger.warning('No RFT headers available (empty context)')
        async for event in self._inner.stream(*args, **kwargs):
            yield event

    def update_config(self, **model_config: Any) -> None:
        self._inner.update_config(**model_config)

    def get_config(self) -> Any:
        return self._inner.get_config()

    def structured_output(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner.structured_output(*args, **kwargs)
