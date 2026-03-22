from enum import Enum, StrEnum, auto


class ItemType(StrEnum):
    BIRD = auto()
    RING_METAL = auto()
    RING_PLASTIC = auto()
    MUD = auto()
    BEAK_MUD = MUD
    EVENT = auto()
    OTHER = auto()
    COWBIRD = OTHER


ItemType.MUD._add_value_alias_("beak_mud")
ItemType.OTHER._add_value_alias_("cowbird")


class Subject(Enum):
    NOT_CLASSIFIED = "not_classified"
    RING = "ring"
    NO_RING = "no_ring"


Subject.RING._add_value_alias_(True)
Subject.NO_RING._add_value_alias_(False)
