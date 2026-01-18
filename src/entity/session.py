"""Session and Event models."""

from dataclasses import dataclass
from typing import List


@dataclass
class Event:
    """Event model."""
    # Article Id (Product Code) of the associated Event
    aid: int
    # Unix timestamp of the Event
    ts: int
    # Type of the Event (e.g., clicks, carts, orders)
    type: str


@dataclass
class Session:
    """Session model."""
    # Unique Session Id
    session: int
    # Ordered sequence of events in the session
    events: List[Event]

