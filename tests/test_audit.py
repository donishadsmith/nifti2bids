from pathlib import Path

import pandas as pd, pytest, numpy as np
from pandas.testing import assert_frame_equal

from nifti2bids.audit import BIDSAuditor
from nifti2bids.simulate import simulate_bids_dataset


@pytest.mark.parametrize("n_sessions", [1, None])
def test_BIDSAuditor(tmp_dir, n_sessions):
    """Test for ``BIDSAuditor``."""
    bids_root = simulate_bids_dataset(
        output_dir=Path(tmp_dir.name) / "BIDS", n_sessions=n_sessions
    )

    BIDSAuditor.clear_caches()

    # Ensure no error for ``derivatives_dir``
    auditor = BIDSAuditor(bids_root, derivatives_dir=True)
    auditor = BIDSAuditor(
        bids_root, derivatives_dir=bids_root / "derivatives" / "fmriprep"
    )

    base_dict = {"subject": ["1"], "session": (["1" if n_sessions else np.nan])}

    expected_nifti_df = pd.DataFrame(
        base_dict
        | {
            "T1w": ["No"],
            "rest": ["Yes"],
        }
    )

    assert_frame_equal(auditor.check_raw_nifti_availability(), expected_nifti_df)

    expected_event_df = pd.DataFrame(
        base_dict
        | {
            "rest": ["No"],
        }
    )
    assert_frame_equal(auditor.check_events_availability(), expected_event_df)

    expected_json_df = pd.DataFrame(
        base_dict
        | {
            "T1w": ["No"],
            "rest": ["No"],
        }
    )
    assert_frame_equal(auditor.check_raw_sidecar_availability(), expected_json_df)

    expected_preprocessed_df = pd.DataFrame(
        base_dict
        | {
            "rest": ["No"],
        }
    )
    assert_frame_equal(
        auditor.check_preprocessed_nifti_availability(
            template_space="MNI152NLin2009cAsym"
        ),
        expected_preprocessed_df,
    )

    kwargs = {"analysis_dir": bids_root / "derivatives" / "firstlevel"}
    if n_sessions is None:
        filename = (
            bids_root
            / "derivatives"
            / "firstlevel"
            / "sub-1"
            / "func"
            / "sub-1_task-rest_desc-stats.nii"
        )
    else:
        filename = (
            bids_root
            / "derivatives"
            / "firstlevel"
            / "sub-1"
            / "ses-1"
            / "func"
            / "sub-1_ses-1_task-rest_run-1_space-MNI152_desc-stats_something.nii"
        )
        kwargs.update(
            {"template_space": "MNI152", "run_id": "1", "desc": "stats_something"}
        )

    filename.parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "w") as foo:
        pass

    expected_preprocessed_df = pd.DataFrame(
        base_dict
        | {
            "rest": ["Yes"],
        }
    )

    assert_frame_equal(
        auditor.check_first_level_availability(**kwargs),
        expected_preprocessed_df,
    )

    if n_sessions:
        kwargs["template_space"] = None
        kwargs["run_id"] = None
        assert_frame_equal(
            auditor.check_first_level_availability(**kwargs),
            expected_preprocessed_df,
        )
