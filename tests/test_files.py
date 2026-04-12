from pathlib import Path

import pytest, pandas as pd

from bidsaid.files import (
    create_bids_file,
    create_participant_tsv,
    create_sessions_tsv,
    create_dataset_description,
    save_dataset_description,
    get_entity_value,
)


@pytest.mark.parametrize("dst_dir, remove_src_file", ([None, True], [True, False]))
def test_create_bids_file(nifti_img_and_path, dst_dir, remove_src_file):
    """Test for ``create_bids_file``."""
    _, img_path = nifti_img_and_path
    dst_dir = None if not dst_dir else img_path.parent / "subdir" / "test"

    bids_filename = create_bids_file(
        img_path,
        sub_id="01",
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
    dataset_desc = create_dataset_description(
        dataset_name="test", bids_version="1.2.0", derivative=True
    )
    assert dataset_desc.get("Name") == "test"
    assert dataset_desc.get("BIDSVersion") == "1.2.0"
    assert dataset_desc.get("GeneratedBy") == [{"Name": "test"}]


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


def test_create_sessions_tsv(tmp_dir):
    """Test for ``create_sessions_tsv``."""
    from bidsaid.simulate import simulate_bids_dataset

    path = simulate_bids_dataset(n_sessions=3, output_dir=Path(tmp_dir.name) / "BIDS")

    df = create_sessions_tsv(path, sub_id=1, save_df=True, return_df=True)
    assert isinstance(df, pd.DataFrame)

    filename = path / "sub-1_sessions.tsv"
    assert filename.is_file()

    df = pd.read_csv(filename, sep="\t")
    assert df["session_id"].values.tolist() == ["ses-1", "ses-2", "ses-3"]


def test_get_entity_value():
    """Test for ``get_entity_value``."""
    filename = "sub-01_task-test_bold.nii.gz"
    assert get_entity_value(filename, "task") == "test"
    assert not get_entity_value(filename, "ses")
