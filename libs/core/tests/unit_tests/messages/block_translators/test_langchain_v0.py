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


def test_v0_index_is_kept_at_the_top_level() -> None:
    """`index` is a declared field on the data block types, not an extra.

    Only `file` + `source_type: "id"` used to place it correctly, because that branch
    forwards extras into a constructor that accepts `index`.
    """
    cases: list[dict[str, object]] = [
        {"type": "image", "source_type": "url", "url": "https://example.com/i.png"},
        {"type": "image", "source_type": "base64", "data": "QUJD"},
        {"type": "image", "source_type": "id", "id": "file-1"},
        {"type": "audio", "source_type": "url", "url": "https://example.com/a.wav"},
        {"type": "audio", "source_type": "base64", "data": "QUJD"},
        {"type": "audio", "source_type": "id", "id": "file-1"},
        {"type": "file", "source_type": "url", "url": "https://example.com/f.pdf"},
        {"type": "file", "source_type": "base64", "data": "QUJD"},
        {"type": "file", "source_type": "id", "id": "file-1"},
        {"type": "file", "source_type": "text", "url": "hello"},
    ]

    for case in cases:
        converted = _convert_legacy_v0_content_block_to_v1({**case, "index": 3})

        assert converted["index"] == 3, case
        assert "index" not in converted.get("extras", {}), case


def test_v0_index_does_not_displace_genuine_extras() -> None:
    block = {
        "type": "image",
        "source_type": "url",
        "url": "https://example.com/image.png",
        "index": 7,
        "alt_text": "An example image",
    }

    assert _convert_legacy_v0_content_block_to_v1(block) == {
        "type": "image",
        "url": "https://example.com/image.png",
        "index": 7,
        "extras": {"alt_text": "An example image"},
    }


def test_v0_block_without_index_is_unchanged() -> None:
    block = {
        "type": "image",
        "source_type": "url",
        "url": "https://example.com/image.png",
    }

    assert "index" not in _convert_legacy_v0_content_block_to_v1(block)
