from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from nifti2bids.audit import BIDSAuditor
from nifti2bids.simulate import simulate_bids_dataset


def test_BIDSAuditor(tmp_dir):
    """Test for ``BIDSAuditor``."""
    bids_root = simulate_bids_dataset(output_dir=Path(tmp_dir.name) / "BIDS")

    BIDSAuditor.clear_caches()

    auditor = BIDSAuditor(bids_root)

    expected_nifti_df = pd.DataFrame(
        {
            "subject": ["0"],
            "session": ["0"],
            "T1w": ["No"],
            "rest": ["Yes"],
        }
    )
    assert_frame_equal(auditor.check_raw_nifti_availability(), expected_nifti_df)

    expected_event_df = pd.DataFrame(
        {
            "subject": ["0"],
            "session": ["0"],
            "rest": ["No"],
        }
    )
    assert_frame_equal(auditor.check_events_availability(), expected_event_df)

    expected_json_df = pd.DataFrame(
        {
            "subject": ["0"],
            "session": ["0"],
            "T1w": ["No"],
            "rest": ["No"],
        }
    )
    assert_frame_equal(auditor.check_raw_sidecar_availability(), expected_json_df)

    expected_preprocessed_df = pd.DataFrame(
        {
            "subject": ["0"],
            "session": ["0"],
            "rest": ["No"],
        }
    )
    assert_frame_equal(
        auditor.check_preprocessed_nifti_availability(
            template_space="MNI152NLin2009cAsym"
        ),
        expected_preprocessed_df,
    )
