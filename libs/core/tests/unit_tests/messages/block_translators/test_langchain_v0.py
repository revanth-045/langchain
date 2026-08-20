from langchain_core.messages import HumanMessage
from langchain_core.messages import content as types
from langchain_core.messages.block_translators.langchain_v0 import (
    _convert_legacy_v0_content_block_to_v1,
)
from tests.unit_tests.language_models.chat_models.test_base import (
    _content_blocks_equal_ignore_id,
)


def test_convert_to_v1_from_openai_input() -> None:
    message = HumanMessage(
        content=[
            {"type": "text", "text": "Hello"},
            {
                "type": "image",
                "source_type": "url",
                "url": "https://example.com/image.png",
            },
            {
                "type": "image",
                "source_type": "base64",
                "data": "<base64 data>",
                "mime_type": "image/png",
            },
            {
                "type": "file",
                "source_type": "url",
                "url": "<document url>",
            },
            {
                "type": "file",
                "source_type": "base64",
                "data": "<base64 data>",
                "mime_type": "application/pdf",
            },
            {
                "type": "audio",
                "source_type": "base64",
                "data": "<base64 data>",
                "mime_type": "audio/mpeg",
            },
            {
                "type": "file",
                "source_type": "id",
                "id": "<file id>",
            },
        ]
    )

    expected: list[types.ContentBlock] = [
        {"type": "text", "text": "Hello"},
        {
            "type": "image",
            "url": "https://example.com/image.png",
        },
        {
            "type": "image",
            "base64": "<base64 data>",
            "mime_type": "image/png",
        },
        {
            "type": "file",
            "url": "<document url>",
        },
        {
            "type": "file",
            "base64": "<base64 data>",
            "mime_type": "application/pdf",
        },
        {
            "type": "audio",
            "base64": "<base64 data>",
            "mime_type": "audio/mpeg",
        },
        {
            "type": "file",
            "file_id": "<file id>",
        },
    ]

    assert _content_blocks_equal_ignore_id(message.content_blocks, expected)


def test_convert_with_extras_on_v0_block() -> None:
    """Test that extras on old-style blocks are preserved in conversion.

    Refer to `_extract_v0_extras` for details.
    """
    block = {
        "type": "image",
        "source_type": "url",
        "url": "https://example.com/image.png",
        # extras follow
        "alt_text": "An example image",
        "caption": "Example caption",
        "name": "example_image",
        "description": None,
        "attribution": None,
    }
    expected_output = {
        "type": "image",
        "url": "https://example.com/image.png",
        "extras": {
            "alt_text": "An example image",
            "caption": "Example caption",
            "name": "example_image",
            # "description": None,  # These are filtered out
            # "attribution": None,
        },
    }

    assert _convert_legacy_v0_content_block_to_v1(block) == expected_output


def test_v0_id_block_keeps_mime_type_top_level() -> None:
    """`mime_type` is a standard field, so it must not be demoted into `extras`.

    v0 `source_type="id"` blocks used to route `mime_type` into `extras` for images
    and audio, while `file` and the `url`/`base64` source types kept it top level.
    Genuine unknown keys still belong in `extras`.
    """
    image_block = {
        "type": "image",
        "source_type": "id",
        "id": "<file id>",
        "mime_type": "image/png",
        "alt_text": "An example image",
    }
    assert _convert_legacy_v0_content_block_to_v1(image_block) == {
        "type": "image",
        "file_id": "<file id>",
        "mime_type": "image/png",
        "extras": {"alt_text": "An example image"},
    }

    audio_block = {
        "type": "audio",
        "source_type": "id",
        "id": "<file id>",
        "mime_type": "audio/mpeg",
        "alt_text": "An example clip",
    }
    assert _convert_legacy_v0_content_block_to_v1(audio_block) == {
        "type": "audio",
        "file_id": "<file id>",
        "mime_type": "audio/mpeg",
        "extras": {"alt_text": "An example clip"},
    }


def test_v0_id_block_without_mime_type_omits_it() -> None:
    """A v0 block carrying no `mime_type` must not gain an empty one."""
    assert _convert_legacy_v0_content_block_to_v1(
        {"type": "image", "source_type": "id", "id": "<file id>"}
    ) == {"type": "image", "file_id": "<file id>"}

    assert _convert_legacy_v0_content_block_to_v1(
        {"type": "audio", "source_type": "id", "id": "<file id>"}
    ) == {"type": "audio", "file_id": "<file id>"}
