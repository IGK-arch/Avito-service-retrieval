"""Small independent checks for ranking, ID handling and archive integrity."""

import hashlib
import json
import zipfile

import numpy as np
import pytest

from common import FEATURES, select_legal_candidates
from package_reproduction import CACHE_FILES, MANIFEST, extract
from ranking_features import augment_features, standardize_scores
from validate_answer import validate_rows


def test_legal_topk_excludes_validation_items_before_selecting():
    item_ids = np.array(["0000000000000000", "0000000000000001", "0000000000000002"])
    result = select_legal_candidates(
        np.array([0, 0, 0, 1, 1]),
        np.array([2, 1, 0, 1, 0]),
        np.array([100, 5, 5, 1, 2]),
        item_ids,
        set(item_ids[:2]),
        k=2,
    )
    assert result == [item_ids[:2].tolist(), item_ids[:2].tolist()]


def test_duplicate_candidates_fail_instead_of_wasting_submission_slots():
    with pytest.raises(ValueError, match="Duplicate"):
        select_legal_candidates(
            np.array([0, 0]),
            np.array([0, 0]),
            np.array([1, 2]),
            np.array(["0000000000000000"]),
            {"0000000000000000"},
        )


def test_relative_features_handle_ties_and_short_groups():
    X = np.zeros((4, len(FEATURES)), dtype=np.float32)
    X[:, 0] = [3, 3, 1, 10]
    augmented = augment_features(X, np.array([0, 0, 0, 1]))
    # Both tied leaders have percentile 1; the final item has percentile 0.
    np.testing.assert_array_equal(augmented[:, 41], [1, 1, 0, 1])
    np.testing.assert_array_equal(augmented[:, 42], [0, 0, -2, 0])
    # For fewer than 50 candidates, the last score is the gap reference.
    np.testing.assert_array_equal(augmented[:, 43], [2, 2, 0, 0])
    with pytest.raises(ValueError, match="sorted"):
        augment_features(X, np.array([0, 1, 0, 1]))


def test_ensemble_calibration_does_not_mix_query_groups():
    values = standardize_scores(np.array([2, 4, 50, 50]), np.array([0, 0, 1, 1]))
    np.testing.assert_allclose(values, [-1, 1, 0, 0])


def test_submission_preserves_literal_identifiers():
    query = "aBcD0123456789Xy"
    item = "0000000000000001"
    valid = validate_rows(["query_id", "answer"], [[query, item]], [query], [item])
    assert valid["valid"]
    invalid = validate_rows(
        ["query_id", "answer"], [[query.lower(), item + " " + item]], [query], [item]
    )
    assert invalid["error_counts"]["unknown_query_id"] == 1
    assert invalid["error_counts"]["duplicate_item_id"] == 1
    assert invalid["error_counts"]["missing_query_id"] == 1


def test_archive_integrity_is_checked_after_extraction(tmp_path):
    contents = {name: b"example artifact" for name in CACHE_FILES}
    metadata = {
        "files": {
            name: {"sha256": hashlib.sha256(data).hexdigest()}
            for name, data in contents.items()
        }
    }
    contents[CACHE_FILES[0]] = b"corrupted artifact"
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for name, data in contents.items():
            bundle.writestr("artifacts/" + name, data)
        bundle.writestr("artifacts/" + MANIFEST, json.dumps(metadata))
    with pytest.raises(ValueError, match="Checksum mismatch"):
        extract(archive, tmp_path / "restored")


def test_archive_rejects_unexpected_paths(tmp_path):
    archive = tmp_path / "unexpected.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../outside.txt", "unexpected")
    with pytest.raises(ValueError, match="Unexpected"):
        extract(archive, tmp_path / "restored")
    assert not (tmp_path / "outside.txt").exists()
