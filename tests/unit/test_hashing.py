from bitcoin_aml.utils.hashing import sha256_json


def test_sha256_json_is_deterministic_regardless_of_key_order() -> None:
    first = sha256_json({"a": 1, "b": 2})
    second = sha256_json({"b": 2, "a": 1})

    assert first == second


def test_sha256_json_differs_for_different_payloads() -> None:
    first = sha256_json({"a": 1})
    second = sha256_json({"a": 2})

    assert first != second


def test_sha256_json_handles_nested_structures() -> None:
    payload = {"list": [1, 2, {"x": None, "y": True}], "value": 3.5}

    assert sha256_json(payload) == sha256_json(payload)
