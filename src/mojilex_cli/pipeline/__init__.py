"""Application pipeline joining adapters to the platform-neutral domain."""

from .transform import CollectionPlan, plan_collection_merge

__all__ = ["CollectionPlan", "plan_collection_merge"]
