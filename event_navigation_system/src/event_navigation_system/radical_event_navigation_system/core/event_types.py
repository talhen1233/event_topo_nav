#!/usr/bin/env python3
"""Event, mode, decision, and junction enumerations."""

from enum import Enum, auto


class EventType(Enum):
    """Navigation events that can be stored and matched."""
    JUNCTION = auto()
    GEOMETRY = auto()
    START = auto()


class NavigationMode(Enum):
    """Navigation state-machine modes."""
    WAIT_START = auto()
    EXPLORE = auto()
    BACKTRACK = auto()
    RETURN_TO_BASE = auto()
    IDLE = auto()


class Decision(Enum):
    HOVER = auto()


class JunctionType(Enum):
    """Junction topology labels."""
    T_JUNCTION = auto()
    CROSS_JUNCTION = auto()
    DEAD_END = auto()
