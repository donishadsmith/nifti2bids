import io
from pathlib import Path

import numpy as np, pandas as pd, pytest

from nifti2bids.qc import (
    compute_n_dummy_scans,
    compute_framewise_displacement,
    create_censor_mask,
    merge_censor_masks,
    compute_consecutive_censor_stats,
)


def _create_data(tmp_dir, data, input_type, has_header=True):
    data = io.StringIO("".join(data), newline=None)

    df = pd.read_csv(data, sep="\t")
    if input_type == "file":
        input_data = Path(tmp_dir.name) / "confounds.tsv"
        df.to_csv(input_data, sep="\t", index=None, header=has_header)
    else:
        input_data = df

    return input_data


@pytest.mark.parametrize("input_type", ["file", "df"])
def test_compute_n_dummy_scans(tmp_dir, input_type):
    "Test for ``compute_n_dummy_scans``."
    data = ["non_steady_state_outlier_01\tnon_steady_state_outlier_02\n", "0\t0\n"]
    assert compute_n_dummy_scans(_create_data(tmp_dir, data, input_type)) == 2


@pytest.mark.parametrize("input_type", ["file", "df"])
def test_compute_framewise_displacement(tmp_dir, input_type):
    "Test ``compute_framewise_displacement``."
    header = (
        "\t".join(["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]) + "\n"
    )
    data = [header, "1\t1\t1\t1\t1\t1\n", "1\t1\t1\t1\t1\t1\n"]

    input_data = _create_data(tmp_dir, data, input_type)
    assert np.array_equal(compute_framewise_displacement(input_data), np.array([0, 0]))

    if input_type == "file":
        input_data = _create_data(
            tmp_dir,
            data,
            input_type,
            has_header=False,
        )
        assert np.array_equal(
            compute_framewise_displacement(
                input_data,
                has_header=False,
                verbose=True,
            ),
            np.array([0, 0]),
        )
    else:
        input_data = _create_data(tmp_dir, data, input_type)
        assert np.array_equal(
            compute_framewise_displacement(
                input_data.to_numpy(copy=True),
                verbose=True,
            ),
            np.array([0, 0]),
        )


@pytest.mark.parametrize("input_type", ["file", "df"])
def test_create_censor_mask(tmp_dir, input_type):
    data = ["fd\tplaceholder\n", "0\n0.1\n0.5\n0.6\n0.7\n0.1\n"]

    censor_mask = create_censor_mask(
        _create_data(tmp_dir, data, input_type), n_dummy_scans=2
    )
    assert np.array_equal(censor_mask, np.array([0, 0, 1, 1, 1, 1]))

    censor_mask = create_censor_mask(
        _create_data(tmp_dir, data, input_type), n_dummy_scans=2
    )
    assert np.array_equal(censor_mask, np.array([0, 0, 1, 1, 1, 1]))

    if input_type == "file":
        censor_mask = create_censor_mask(
            _create_data(tmp_dir, data, input_type),
            n_dummy_scans=2,
            column_name="fd",
            threshold=0.5,
            verbose=True,
        )
        assert np.array_equal(censor_mask, np.array([0, 0, 1, 0, 0, 1]))
    else:
        data = _create_data(tmp_dir, data, input_type)
        censor_mask = create_censor_mask(
            data["fd"].to_numpy(copy=True),
            n_dummy_scans=2,
            threshold=0.5,
            verbose=True,
        )
        assert np.array_equal(censor_mask, np.array([0, 0, 1, 0, 0, 1]))


def test_merge_censor_mask():
    "Test ``merge_censor_mask``."
    censor_masks = [np.array([0, 0, 1]), np.array([1, 0, 1])]

    assert np.array_equal(merge_censor_masks(censor_masks), np.array([0, 0, 1]))


def test_compute_consecutive_censor_stats():
    "Test ``compute_consecutive_censor_stats``."
    censor_mask = np.array([0, 0, 1, 1])
    mean, std = compute_consecutive_censor_stats(censor_mask)
    assert (mean, std) == (2.0, 0.0)

    mean, std = compute_consecutive_censor_stats(censor_mask, n_dummy_scans=2)
    assert (mean, std) == (0.0, np.nan)
