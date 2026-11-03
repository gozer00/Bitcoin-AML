import numpy as np
import pytest

from bitcoin_aml.data.dataset import EllipticDataset
from bitcoin_aml.features.specification import (
    FeatureSetName,
    FeatureSpec,
    ScalingStrategy,
)
from bitcoin_aml.graphs.specification import (
    GraphSpec,
    GraphViewName,
)
from bitcoin_aml.models.base import ModelAdapter
from bitcoin_aml.splits.specification import (
    SplitSpec,
    TimestepRange,
)

def test_valid_thesis_split() -> None:
    split = SplitSpec(
        train=TimestepRange(1, 29),
        validation=TimestepRange(30, 34),
        test=TimestepRange(35, 49),
    )

    assert split.train.start == 1
    assert split.train.end == 29
    assert split.validation.start == 30
    assert split.validation.end == 34
    assert split.test.start == 35
    assert split.test.end == 49

def test_overlapping_splits_is_rejected() -> None:
    with pytest.raises(ValueError):
        SplitSpec(
            train=TimestepRange(1, 29),
            validation=TimestepRange(29, 34),
            test=TimestepRange(35, 49),
        )

def test_feature_spec() -> None:
    spec = FeatureSpec(
        feature_set=FeatureSetName.LOCAL_93,
        scaling=ScalingStrategy.STANDARD,
    )

    assert spec.feature_set == FeatureSetName.LOCAL_93

def test_incoming_graph_requires_hops() -> None:
    with pytest.raises(ValueError):
        GraphSpec(
            view=GraphViewName.INCOMING,
        )

def test_valid_incoming_graph() -> None:
    spec = GraphSpec(
        view=GraphViewName.INCOMING,
        hops=2
    )

    assert spec.hops == 2

def test_reversed_graph_rejects_hops() -> None:
    with pytest.raises(ValueError):
        GraphSpec(
            view=GraphViewName.REVERSED,
            hops=1,
        )

def test_dataset_requires_aligned_node_arrays() -> None:
    with pytest.raises(ValueError):
        EllipticDataset(
            tx_id=np.array([1, 2, 3]),
            timestep=np.array([1, 1]),
            features=np.zeros((3, 165)),
            labels=np.array([0, 1, -1]),
            edge_index=np.empty((2, 0), dtype=int),
        )


def test_model_adapter_is_abstract() -> None:
    with pytest.raises(TypeError):
        ModelAdapter()
