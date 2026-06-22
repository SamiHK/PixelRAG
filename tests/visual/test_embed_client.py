import numpy as np
from unittest.mock import MagicMock

from pixelrag_visual.embed_client import get_lm_client, embed_text, embed_batch


def test_get_lm_client_default_url():
    client = get_lm_client()
    # OpenAI SDK normalizes URLs with a trailing slash
    assert str(client.base_url).rstrip("/") == "http://172.16.0.203:1234/v1"


def test_get_lm_client_custom_url():
    client = get_lm_client("http://localhost:1234/v1")
    assert str(client.base_url).rstrip("/") == "http://localhost:1234/v1"


def test_embed_text_returns_normalized_vector():
    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=[1.0, 2.0, 3.0])]
    )

    result = embed_text(mock_client, "test query", model="dummy-model")

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    expected_norm = np.sqrt(np.sum(np.array([1.0, 2.0, 3.0]) ** 2))
    expected_normalized = np.array([1.0, 2.0, 3.0]) / expected_norm
    np.testing.assert_allclose(result, expected_normalized, atol=1e-6)


def test_embed_text_calls_correct_api():
    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=[0.1, 0.2])]
    )

    embed_text(mock_client, "hello world", model="test-model")

    mock_client.embeddings.create.assert_called_once()
    call_kwargs = mock_client.embeddings.create.call_args[1]
    assert call_kwargs["model"] == "test-model"
    assert call_kwargs["input"] == "hello world"


def test_embed_batch_stacks_vectors():
    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=[1.0, 0.0]), MagicMock(embedding=[0.0, 1.0])]
    )

    result = embed_batch(mock_client, ["a", "b"], model="test-model")

    assert result.shape == (2, 2)
    np.testing.assert_array_equal(result[0], [1.0, 0.0])
    np.testing.assert_array_equal(result[1], [0.0, 1.0])


def test_embed_batch_normalizes_each_vector():
    mock_client = MagicMock()
    # Return unnormalized vectors [3, 4] and [5, 12]
    mock_client.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=[3.0, 4.0]), MagicMock(embedding=[5.0, 12.0])]
    )

    result = embed_batch(mock_client, ["a", "b"], model="test-model")

    # [3, 4] has norm 5 → normalized = [0.6, 0.8]
    # [5, 12] has norm 13 → normalized = [5/13, 12/13]
    np.testing.assert_allclose(result[0], [0.6, 0.8], atol=1e-6)
    np.testing.assert_allclose(result[1], [5 / 13, 12 / 13], atol=1e-6)


def test_embed_batch_empty_list():
    mock_client = MagicMock()

    result = embed_batch(mock_client, [], model="test-model")

    assert isinstance(result, np.ndarray)
    assert result.shape == (0, 0)
    assert result.dtype == np.float32
    # Ensure no API call was made for empty input
    mock_client.embeddings.create.assert_not_called()
