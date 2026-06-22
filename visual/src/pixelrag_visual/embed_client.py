"""Embed text via LM Studio's OpenAI-compatible /v1/embeddings endpoint."""

import numpy as np
from openai import OpenAI


DEFAULT_LM_STUDIO_URL = "http://172.16.0.203:1234/v1"


def get_lm_client(base_url: str | None = None) -> OpenAI:
    """Create an OpenAI client configured for LM Studio.

    Args:
        base_url: LM Studio API base URL. Defaults to DEFAULT_LM_STUDIO_URL.

    Returns:
        Configured OpenAI client instance.
    """
    url = base_url or DEFAULT_LM_STUDIO_URL
    return OpenAI(base_url=url, api_key="not-needed")


def embed_text(
    client: OpenAI, text: str, model: str = "qwen3-vl-embedding-2b"
) -> np.ndarray:
    """Embed a single text string via LM Studio.

    Args:
        client: OpenAI-compatible client configured for LM Studio.
        text: Text to embed.
        model: Embedding model name (e.g. 'qwen3-vl-embedding-2b').

    Returns:
        L2-normalized float32 embedding vector.
    """
    response = client.embeddings.create(model=model, input=text)
    vec = np.array(response.data[0].embedding, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def embed_batch(
    client: OpenAI,
    texts: list[str],
    model: str = "qwen3-vl-embedding-2b",
    batch_size: int = 16,
) -> np.ndarray:
    """Embed multiple text strings via LM Studio.

    Args:
        client: OpenAI-compatible client configured for LM Studio.
        texts: List of text strings to embed.
        model: Embedding model name.
        batch_size: Number of texts per API call (LM Studio may have limits).

    Returns:
        Stacked float32 array of shape (N, D), L2-normalized.
    """
    all_vectors: list[np.ndarray] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(model=model, input=batch)

        for item in response.data:
            vec = np.array(item.embedding, dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            all_vectors.append(vec)

    return np.stack(all_vectors, axis=0)
