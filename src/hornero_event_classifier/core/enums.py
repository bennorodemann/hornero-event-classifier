"""Enum types used across the core module."""

from enum import Enum, auto

try:
    from enum import StrEnum
except ImportError:  # pragma: no cover - Python < 3.11
    class StrEnum(str, Enum):
        """Python 3.10-compatible fallback for :class:`enum.StrEnum`."""

        @staticmethod
        def _generate_next_value_(name, start, count, last_values):
            del start, count, last_values
            return name.lower()


class ItemType(StrEnum):
    """String-valued item types used to label tracked objects and events."""

    BIRD = auto()
    """Tracked bird instance."""
    RING_METAL = auto()
    """Metal leg ring."""
    RING_PLASTIC = auto()
    """Plastic leg ring."""
    RING = auto()
    """Generic ring (unspecified type)."""
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

    @classmethod
    def _missing_(cls, value):
        alias_map = {
            "beak_mud": cls.MUD,
            "cowbird": cls.OTHER,
            "another_bird": cls.OTHER,
        }
        return alias_map.get(value)


class Subject(Enum):
    """Subject labels for classification output."""

    NOT_CLASSIFIED = "not_classified"
    """Subject has not been classified yet."""
    RING = "ring"
    """Ringed bird. Value aliases: :code:`True`."""
    NO_RING = "no_ring"
    """Unringed bird. Value aliases: :code:`False`."""

    @classmethod
    def _missing_(cls, value):
        alias_map = {
            True: cls.RING,
            False: cls.NO_RING,
        }
        return alias_map.get(value)
