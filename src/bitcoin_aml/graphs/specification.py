from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class GraphViewName(StrEnum):
    NONE = "none"
    DIRECTED = "directed"
    BIDIRECTIONAL = "bidirectional"
    INCOMING = "incoming"
    REVERSED = "reversed"

class GraphAccessPolicy(StrEnum):
    STRICT_ROLLING = "strict_rolling"
    FULL_TRANSDUCTIVE = "full_transductive"

@dataclass(frozen=True, slots=True)
class GraphSpec:
    view: GraphViewName
    hops: int | None = None

    def __post_init__(self) -> None:
        if self.view == GraphViewName.INCOMING:
            if self.hops is None:
                raise ValueError("incoming graph view requires a positive number of hops")

            if self.hops < 1:
                raise ValueError("hops must be >= 1")

        elif self.hops is not None:
            raise ValueError("hops can only be specified for incoming graph views")