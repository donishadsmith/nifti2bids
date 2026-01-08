from pathlib import Path

import nibabel as nib, numpy as np, pytest

from nifti2bids.simulate import (
    create_affine,
    simulate_nifti_image,
    simulate_bids_dataset,
)


def test_create_affine():
    """Test for ``create_affine``."""
    affine = create_affine(xyz_diagonal_value=2, translation_vector=(1, 1, 1, 1))
    assert all(np.diagonal(affine) == np.array([2, 2, 2, 1]))
    assert all(affine[:, 3] == np.array([1, 1, 1, 1]))


@pytest.mark.parametrize(
    "affine",
    [
        None,
        create_affine(xyz_diagonal_value=1, translation_vector=np.array([1, 1, 1, 1])),
    ],
)
def test_simulate_nifti_image(affine):
    """Test for ``simulate_nifti_image``."""
    img = simulate_nifti_image(img_shape=(20, 20, 20, 20), affine=affine)
    assert isinstance(img, nib.Nifti1Image)
    if affine is not None:
        assert all(np.diagonal(img.affine) == np.array([1, 1, 1, 1]))


@pytest.mark.parametrize("n_sessions", [1, None])
def test_simulate_bids_dataset(tmp_dir, n_sessions):
    """Test for ``simulate_bids_dataset``."""
    import bids

    bids_root = simulate_bids_dataset(
        output_dir=Path(tmp_dir.name) / "BIDS", n_sessions=n_sessions
    )

    layout = bids.BIDSLayout(bids_root, derivatives=True)
    files = layout.get(return_type="file", extension="nii.gz")
    assert len(files) == 2

    if n_sessions:
        assert layout.get_sessions() == ["1"]
    else:
        assert not layout.get_sessions()
