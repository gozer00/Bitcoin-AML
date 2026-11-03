from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class FeatureSetName(StrEnum):
    LOCAL_21 = "local_21"
    LOCAL_93 = "local_93"
    AGGREGATE_72 = "aggregate_72"
    ALL_165 = "all_165"

class ScalingStrategy(StrEnum):
    NONE = "none"
    STANDARD = "standard"
    YEO_JOHNSON_SKEWED = "yeo_johnson_skewed"

@dataclass(frozen=True, slots=True)
class FeatureSpec:
    feature_set: FeatureSetName
    scaling: ScalingStrategy