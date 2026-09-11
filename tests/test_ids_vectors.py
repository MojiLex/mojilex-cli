import unicodedata

import pytest

from mojilex_cli.domain.ids import IdentityError, collection_id, emoji_id, membership_id


def test_ids_match_normative_spec_vectors() -> None:
    collection = collection_id("telegram", "sticker_set.name", "global", "SuspiciousCats")
    emoji = emoji_id("telegram", "custom_emoji.id", "global", "5368324170671202286")
    assert collection == "mxc_5f3fda56-0066-518c-88d9-1fc62da96d38"
    assert emoji == "mxe_9f8f6af4-72dc-5fef-9e71-f787a06ac68d"
    assert membership_id(collection, emoji) == "mxm_e3fbd0c6-b5d2-5151-b4b1-c820891028d0"


def test_ids_normalize_every_component_to_nfc() -> None:
    decomposed = "Cafe\u0301"
    composed = unicodedata.normalize("NFC", decomposed)
    assert collection_id("telegram", "ns", "global", decomposed) == collection_id(
        "telegram", "ns", "global", composed
    )


@pytest.mark.parametrize("bad", ["bad\0id", "\0", "x\0"])
def test_ids_reject_nul_in_components(bad: str) -> None:
    with pytest.raises(IdentityError, match=r"U\+0000"):
        emoji_id("telegram", "custom_emoji.id", "global", bad)


def test_identity_epoch_is_decimal_integer_and_changes_identity() -> None:
    epoch_zero = collection_id("telegram", "ns", "global", "id", 0)
    assert collection_id("telegram", "ns", "global", "id", 1) != epoch_zero
    with pytest.raises(IdentityError):
        collection_id("telegram", "ns", "global", "id", True)
    with pytest.raises(IdentityError):
        collection_id("telegram", "ns", "global", "id", -1)
