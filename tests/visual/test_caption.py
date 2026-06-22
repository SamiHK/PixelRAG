import base64
from unittest.mock import MagicMock, patch

from pixelrag_visual.caption import (
    encode_image_b64,
    generate_caption,
    generate_captions,
)


def test_encode_image_b64_returns_data_url():
    mock_file = MagicMock()
    mock_file.read.return_value = b"\x89PNG\r\n\x1a\n"
    mock_ctx = MagicMock()
    mock_ctx.__enter__.return_value = mock_file
    with patch("builtins.open", return_value=mock_ctx):
        result = encode_image_b64("/fake/photo.png")

    assert result.startswith("data:image/png;base64,")
    decoded = base64.b64decode(result.split(",", 1)[1])
    assert decoded == b"\x89PNG\r\n\x1a\n"


def test_encode_image_b64_jpeg_mime():
    mock_file = MagicMock()
    mock_file.read.return_value = b"\xff\xd8\xff\xe0fake"
    mock_ctx = MagicMock()
    mock_ctx.__enter__.return_value = mock_file
    with patch("builtins.open", return_value=mock_ctx):
        result = encode_image_b64("/fake/photo.jpg")

    assert result.startswith("data:image/jpeg;base64,")


def test_generate_caption_extracts_text_from_response():
    mock_client = MagicMock()
    mock_message = MagicMock()
    mock_message.content = "A group of people smiling at the camera in a park."

    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=mock_message)]
    mock_client.chat.completions.create.return_value = mock_response

    with patch("pixelrag_visual.caption.encode_image_b64") as mock_encode:
        mock_encode.return_value = "data:image/png;base64,SGVsbG8="
        result = generate_caption(
            mock_client, "/fake/photo.png", model="test-vl-model"
        )

    assert result == "A group of people smiling at the camera in a park."
    mock_client.chat.completions.create.assert_called_once()


def test_generate_caption_uses_base64_image_format():
    mock_client = MagicMock()
    mock_message = MagicMock()
    mock_message.content = "A red car."

    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=mock_message)]
    mock_client.chat.completions.create.return_value = mock_response

    with patch("pixelrag_visual.caption.encode_image_b64") as mock_encode:
        mock_encode.return_value = "data:image/png;base64,SGVsbG8="
        generate_caption(mock_client, "/fake/photo.png", model="test-vl-model")

    call_args = mock_client.chat.completions.create.call_args[1]
    messages = call_args["messages"]

    user_msg = [m for m in messages if m["role"] == "user"][0]
    content = user_msg["content"]

    assert isinstance(content, list)
    assert any(part.get("type") == "text" for part in content)
    assert any(part.get("type") == "image_url" for part in content)


def test_generate_caption_default_system_prompt():
    mock_client = MagicMock()
    mock_message = MagicMock()
    mock_message.content = "A sunset."

    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=mock_message)]
    mock_client.chat.completions.create.return_value = mock_response

    with patch("pixelrag_visual.caption.encode_image_b64"):
        generate_caption(mock_client, "/fake/photo.png")

    call_args = mock_client.chat.completions.create.call_args[1]
    system_msg = [m for m in call_args["messages"] if m["role"] == "system"][0]

    assert "visual" in system_msg["content"].lower()


def test_generate_captions_returns_path_to_caption_map():
    mock_client = MagicMock()

    captions = [
        "A red car.",
        "A sunset over the ocean.",
        "A group of people smiling at the camera in a park.",
    ]

    def make_response(content):
        mock_message = MagicMock()
        mock_message.content = content
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=mock_message)]
        return mock_response

    mock_client.chat.completions.create.side_effect = (
        make_response(c) for c in captions
    )

    with patch("pixelrag_visual.caption.encode_image_b64"):
        results = generate_captions(
            mock_client,
            ["/fake/a.png", "/fake/b.jpg", "/fake/c.png"],
            model="test-vl-model",
        )

    assert isinstance(results, dict)
    assert results["/fake/a.png"] == "A red car."
    assert results["/fake/b.jpg"] == "A sunset over the ocean."
    assert results["/fake/c.png"] == (
        "A group of people smiling at the camera in a park."
    )
    assert mock_client.chat.completions.create.call_count == 3


def test_generate_captions_handles_errors():
    mock_client = MagicMock()

    def side_effect(*args, **kwargs):
        raise RuntimeError("model unavailable")

    mock_client.chat.completions.create.side_effect = side_effect

    with patch("pixelrag_visual.caption.encode_image_b64"):
        results = generate_captions(
            mock_client, ["/fake/broken.png"], model="test-vl-model"
        )

    assert "/fake/broken.png" in results
    assert "[ERROR:" in results["/fake/broken.png"]


def test_generate_captions_empty_list():
    mock_client = MagicMock()

    with patch("pixelrag_visual.caption.encode_image_b64"):
        results = generate_captions(mock_client, [], model="test-vl-model")

    assert results == {}
