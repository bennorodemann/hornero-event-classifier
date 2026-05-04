"""Enum types used across the core module."""

from enum import Enum, StrEnum, auto


class ItemType(StrEnum):
    """String-valued item types used to label tracked objects and events."""

    BIRD = auto()
    """Tracked bird instance."""
    RING_METAL = auto()
    """Metal leg ring."""
    RING_PLASTIC = auto()
    """Plastic leg ring."""
    MUD = auto()
    """Mud present on the bird. Value aliases: :code:`"beak_mud"`."""
    BEAK_MUD = MUD
    """Alias for :py:attr:`ItemType.MUD`."""
    EVENT = auto()
    """An event detected by the classifier."""
    OTHER = auto()
    """Miscellaneous or unclassified item. Value aliases: :code:`"cowbird"`."""
    COWBIRD = OTHER
    """Alias for :py:attr:`ItemType.OTHER` when item is a cowbird."""


ItemType.MUD._add_value_alias_("beak_mud")
ItemType.OTHER._add_value_alias_("cowbird")
ItemType.OTHER._add_value_alias_("another_bird")


class Subject(Enum):
    """Subject labels for classification output."""

    NOT_CLASSIFIED = "not_classified"
    """Subject has not been classified yet."""
    RING = "ring"
    """Ringed bird. Value aliases: :code:`True`."""
    NO_RING = "no_ring"
    """Unringed bird. Value aliases: :code:`False`."""


Subject.RING._add_value_alias_(True)
Subject.NO_RING._add_value_alias_(False)
