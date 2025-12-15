from pathlib import Path

import pandas as pd, pytest

from nifti2bids.bids import (
    create_bids_file,
    create_participant_tsv,
    create_dataset_description,
    save_dataset_description,
    PresentationBlockExtractor,
    PresentationEventExtractor,
    EPrimeBlockExtractor,
    EPrimeEventExtractor,
)
from ._constants import BLOCK_PRESENTATION_DATA, EVENT_PRESENTATION_DATA, EPRIME_DATA


@pytest.mark.parametrize("dst_dir, remove_src_file", ([None, True], [True, False]))
def test_create_bids_file(nifti_img_and_path, dst_dir, remove_src_file):
    """Test for ``create_bids_file``."""
    _, img_path = nifti_img_and_path
    dst_dir = None if not dst_dir else img_path.parent / "test"
    if dst_dir:
        dst_dir.mkdir()

    bids_filename = create_bids_file(
        img_path,
        subj_id="01",
        desc="bold",
        remove_src_file=remove_src_file,
        dst_dir=dst_dir,
        return_bids_filename=True,
    )
    assert bids_filename
    assert Path(bids_filename).name == "sub-01_bold.nii"

    if dst_dir:
        dst_file = list(dst_dir.glob("*.nii"))[0]
        assert Path(dst_file).name == "sub-01_bold.nii"

        src_file = list(img_path.parent.glob("*.nii"))[0]
        assert Path(src_file).name == "img.nii"
    else:
        files = list(img_path.parent.glob("*.nii"))
        assert len(files) == 1
        assert files[0].name == "sub-01_bold.nii"


def test_create_dataset_description():
    """Test for ``create_dataset_description``."""
    dataset_desc = create_dataset_description(dataset_name="test", bids_version="1.2.0")
    assert dataset_desc.get("Name") == "test"
    assert dataset_desc.get("BIDSVersion") == "1.2.0"


def test_save_dataset_description(tmp_dir):
    """Test for ``save_dataset_description``."""
    dataset_desc = create_dataset_description(dataset_name="test", bids_version="1.2.0")
    save_dataset_description(dataset_desc, tmp_dir.name)
    files = list(Path(tmp_dir.name).glob("*.json"))
    assert len(files) == 1
    assert Path(files[0]).name == "dataset_description.json"


def test_create_participant_tsv(tmp_dir):
    """Test for ``create_participant_tsv``."""
    path = Path(tmp_dir.name)
    extended_path = path / "sub-01"
    extended_path.mkdir()

    df = create_participant_tsv(path, save_df=True, return_df=True)
    assert isinstance(df, pd.DataFrame)

    filename = path / "participants.tsv"
    assert filename.is_file()

    df = pd.read_csv(filename, sep="\t")
    assert df["participant_id"].values[0] == "sub-01"


def _create_presentation_logfile(dst_dir, design):
    dst_dir = Path(dst_dir)
    data = BLOCK_PRESENTATION_DATA if design == "block" else EVENT_PRESENTATION_DATA

    filename = dst_dir / f"{design}.txt"
    with open(filename, mode="w") as file:
        for line in data:
            file.write("\t".join(line) + "\n")


@pytest.mark.parametrize("scanner_start_time", [None, 10000])
def test_PresentationBlockExtractor(tmp_dir, scanner_start_time):
    """Test for ``PresentationBlockExtractor``."""
    from pandas.testing import assert_frame_equal

    filename = Path(tmp_dir.name) / "block.txt"
    _create_presentation_logfile(tmp_dir.name, "block")

    expected_df = pd.DataFrame(
        {
            "onset": [0.0, 34.0] if scanner_start_time is None else [1.0, 35.0],
            "duration": [14.0, 14.0],
            "trial_type": ["indoor", "indoor"],
        }
    )

    extractor = PresentationBlockExtractor(
        log_or_df=filename,
        block_cue_codes=["indoor"],
        convert_to_seconds=["Time"],
        scanner_event_type="Pulse",
        scanner_trigger_code="99",
        n_discarded_volumes=1 if scanner_start_time is None else 0,
        tr=1 if scanner_start_time is None else None,
        rest_block_code="rest",
        quit_code="quit",
    )

    scanner_start_time = scanner_start_time / 10000 if scanner_start_time else None
    onsets = extractor.extract_onsets(scanner_start_time=scanner_start_time)
    durations = extractor.extract_durations()
    trial_types = extractor.extract_trial_types()

    df = pd.DataFrame(
        {"onset": onsets, "duration": durations, "trial_type": trial_types}
    )
    assert_frame_equal(df, expected_df)


def test_PresentationBlockExtractor_mean_rt_and_accuracy(tmp_dir):
    """Test for ``PresentationBlockExtractor.extract_mean_reaction_times`` and ``extract_mean_accuracies``."""
    filename = Path(tmp_dir.name) / "block.txt"
    _create_presentation_logfile(tmp_dir.name, "block")

    extractor = PresentationBlockExtractor(
        log_or_df=filename,
        block_cue_codes=["indoor"],
        convert_to_seconds=["Time"],
        scanner_event_type="Pulse",
        scanner_trigger_code="99",
        rest_block_code="rest",
        quit_code="quit",
    )

    mean_rts = extractor.extract_mean_reaction_times()
    assert len(mean_rts) == 2
    # Block 1: Mean = (1.5 + 1.2 + 1.6 + 1.1) / 4 = 1.35s
    assert mean_rts[0] == pytest.approx(1.35, rel=1e-2)
    # Block 2: Mean = (1.5 + 1.1 + 1.6) / 3 = 1.400s
    assert mean_rts[1] == pytest.approx(1.400, rel=1e-2)

    # Test mean accuracies
    response_map = {"hit": 1, "miss": 0}
    mean_accs = extractor.extract_mean_accuracies(response_map=response_map)
    assert len(mean_accs) == 2
    # Block 1
    assert mean_accs[0] == 1.0
    # Block 2 is missing one response (1 + 1 + 1 + 0) = 0.75
    assert mean_accs[1] == 0.75


@pytest.mark.parametrize("scanner_start_time", [None, 94621])
def test_PresentationEventExtractor(tmp_dir, scanner_start_time):
    """Test for ``PresentationEventExtractor``."""
    from pandas.testing import assert_frame_equal

    filename = Path(tmp_dir.name) / "event.txt"
    _create_presentation_logfile(tmp_dir.name, "event")

    expected_df = pd.DataFrame(
        {
            "onset": [7.9107, 10.8965],
            "duration": [0.5, 0.5],
            "trial_type": ["incongruentright", "congruentleft"],
            "reaction_time": [0.7058, float("nan")],
            "response": ["hit", float("nan")],
            "accuracy": [1, float("nan")],
        }
    )

    extractor = PresentationEventExtractor(
        log_or_df=filename,
        trial_types=["incongruentright", "congruentleft"],
        convert_to_seconds=["Time", "Duration"],
        scanner_event_type="Pulse",
        scanner_trigger_code="99",
    )

    scanner_start_time = scanner_start_time / 10000 if scanner_start_time else None
    onsets = extractor.extract_onsets(scanner_start_time=scanner_start_time)
    durations = extractor.extract_durations()
    trial_types = extractor.extract_trial_types()
    reaction_times = extractor.extract_reaction_times()
    responses = extractor.extract_responses()
    accuracies = extractor.extract_accuracies({"hit": 1})

    df = pd.DataFrame(
        {
            "onset": onsets,
            "duration": durations,
            "trial_type": trial_types,
            "reaction_time": reaction_times,
            "response": responses,
            "accuracy": accuracies,
        }
    )
    assert_frame_equal(df, expected_df)


def _create_eprime_logfile(dst_dir, design):
    dst_dir = Path(dst_dir)

    filename = dst_dir / f"{design}.txt"
    with open(filename, mode="w") as file:
        for line in EPRIME_DATA:
            file.write("\t".join(line) + "\n")


def test_EPrimeBlockExtractor(tmp_dir):
    """Test for ``EPrimeBlockExtractor``."""
    from pandas.testing import assert_frame_equal

    filename = Path(tmp_dir.name) / "block.txt"

    _create_eprime_logfile(tmp_dir.name, "block")

    for column, scanner_start_time in [(None, 10.0), ("EndTime", None)]:
        expected_df = pd.DataFrame(
            {
                "onset": [-1.0, 19.0] if column else [0.0, 20.0],
                "duration": [20.0, 20.0],
                "trial_type": ["A", "B"],
            }
        )
        extractor = EPrimeBlockExtractor(
            log_or_df=filename,
            block_cue_codes=["A", "B"],
            onset_column_name="Data.OnsetTime",
            procedure_column_name="Procedure",
            trigger_column_name=column,
            convert_to_seconds=["Data.OnsetTime", "EndTime"],
            n_discarded_volumes=1 if column else 0,
            tr=1 if column else None,
            rest_block_code="Rest",
            rest_code_frequency="variable",
        )

        onsets = extractor.extract_onsets(scanner_start_time=scanner_start_time)
        durations = extractor.extract_durations()
        trial_types = extractor.extract_trial_types()
        df = pd.DataFrame(
            {"onset": onsets, "duration": durations, "trial_type": trial_types}
        )
        assert_frame_equal(df, expected_df)


def test_EPrimeBlockExtractor_mean_rt_and_accuracy(tmp_dir):
    """
    Test for ``extract_mean_reaction_times`` and ``extract_mean_accuracies`` in
    ``EPrimeBlockExtractor``."""
    filename = Path(tmp_dir.name) / "block.txt"
    _create_eprime_logfile(tmp_dir.name, "block")

    extractor = EPrimeBlockExtractor(
        log_or_df=filename,
        block_cue_codes=["A", "B"],
        onset_column_name="Data.OnsetTime",
        procedure_column_name="Procedure",
        trigger_column_name="EndTime",
        convert_to_seconds=["Data.OnsetTime", "Data.RT", "EndTime"],
        rest_block_code="Rest",
        rest_code_frequency="variable",
    )

    mean_rts = extractor.extract_mean_reaction_times(
        reaction_time_column_name="Data.RT",
        subject_response_column="Data.RESP",
        correct_response_column="Data.CRESP",
        response_type="correct",
        response_trial_codes=("A", "B"),
    )
    assert len(mean_rts) == 2
    # Block A
    assert mean_rts[0] == 0.5
    # Block B
    assert mean_rts[1] == 0.5

    mean_accs = extractor.extract_mean_accuracies(
        subject_response_column="Data.RESP",
        correct_response_column="Data.CRESP",
        response_trial_codes=("A", "B"),
    )
    assert len(mean_accs) == 2
    # Block A
    assert mean_accs[0] == 0.5
    # Block B
    assert mean_accs[1] == 1.0


def test_EPrimeEventExtractor(tmp_dir):
    """Test for ``EPrimeEventExtractor``."""
    from pandas.testing import assert_frame_equal

    filename = Path(tmp_dir.name) / "event.txt"

    _create_eprime_logfile(tmp_dir.name, "event")

    expected_df = pd.DataFrame(
        {
            "onset": [0.0, 10.0, 20.0, 30.0, 40.0],
            "duration": [1.0, 1.0, 1.0, 1.0, 1.0],
            "trial_type": ["A", "A", "B", "B", "Rest"],
            "reaction_time": [0.5, 0.5, 0.5, 0.5, float("nan")],
            "response": [1, 0, 1, 1, 1],
        }
    )

    for column, scanner_start_time in [(None, 10.0), ("EndTime", None)]:
        extractor = EPrimeEventExtractor(
            log_or_df=filename,
            trial_types=["A", "B", "Rest"],
            onset_column_name="Data.OnsetTime",
            procedure_column_name="Procedure",
            trigger_column_name=column,
            convert_to_seconds=[
                "Data.OnsetTime",
                "Data.OffsetTime",
                "Data.RT",
                "EndTime",
            ],
        )

        onsets = extractor.extract_onsets(scanner_start_time=scanner_start_time)
        durations = extractor.extract_durations(offset_column_name="Data.OffsetTime")
        trial_types = extractor.extract_trial_types()
        reaction_times = extractor.extract_reaction_times(
            reaction_time_column_name="Data.RT"
        )
        responses = extractor.extract_accuracies(
            subject_response_column="Data.RESP",
            correct_response_column="Data.CRESP",
        )

        df = pd.DataFrame(
            {
                "onset": onsets,
                "duration": durations,
                "trial_type": trial_types,
                "reaction_time": reaction_times,
                "response": responses,
            }
        )
        assert_frame_equal(df, expected_df)
