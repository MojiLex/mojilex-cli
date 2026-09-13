"""Bind semantic view references to the single source media shown in this pipeline."""

from collections.abc import Sequence

from .base import AIOutputError, DescriptionItem, SemanticMediaReference


def bind_primary_media_references(
    description: DescriptionItem, *, background_variants: Sequence[str]
) -> DescriptionItem:
    """Return a copy with source references, preserving the immutable AI response.

    Light/dark PNGs are views of one primary source, not separate dataset media.
    Only views actually supplied to the model can be bound; arbitrary variants
    and alternate media remain invalid rather than being silently discarded.
    """
    content = description.facets.text_content
    changed = False
    items = []
    primary = SemanticMediaReference(role="primary")
    for item in content.items:
        for reference in item.media_refs:
            if reference.variant_id is not None or (
                reference.role != "primary"
                and (
                    reference.role not in {"light", "dark"}
                    or reference.role not in background_variants
                )
            ):
                raise AIOutputError(
                    "AI text media_refs do not match the supplied source media/views"
                )
        if item.media_refs == (primary,):
            items.append(item)
        else:
            changed = True
            items.append(item.model_copy(update={"media_refs": (primary,)}))
    if not changed:
        return description
    return description.model_copy(
        update={
            "facets": description.facets.model_copy(
                update={"text_content": content.model_copy(update={"items": tuple(items)})}
            )
        }
    )
