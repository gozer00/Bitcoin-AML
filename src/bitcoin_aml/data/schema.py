from enum import IntEnum

# ---------------------------------------------
# contract for the canonical representation of the Elliptic data set
#   * defines invariants
#   * e.g. for validation of raw columns and shapes
# ---------------------------------------------

ELLIPTIC_NUM_TIMESTEPS = 49
ELLIPTIC_TIMESTEPS = tuple(range(1, ELLIPTIC_NUM_TIMESTEPS + 1))

ELLIPTIC_NUM_FEATURES = 165
ELLIPTIC_NUM_LOCAL21_FEATURES = 21
ELLIPTIC_NUM_LOCAL_FEATURES = 93
ELLIPTIC_NUM_AGGREGATE_FEATURES = 72

class TransactionLabel(IntEnum):
    UNKNOWN = -1
    LICIT = 0
    ILLICIT = 1


RAW_CLASS_MAPPING = {
    "1": TransactionLabel.ILLICIT,
    "2": TransactionLabel.LICIT,
    "unknown": TransactionLabel.UNKNOWN,
}
