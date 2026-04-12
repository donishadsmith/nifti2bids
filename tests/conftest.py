import tempfile
from pathlib import Path

import nibabel as nib, pytest

from bidsaid.simulate import simulate_nifti_image


@pytest.fixture(autouse=False, scope="function")
def tmp_dir():
    """Create temporary directory for each test module."""
    temp_dir = tempfile.TemporaryDirectory()

    yield temp_dir

    temp_dir.cleanup()


@pytest.fixture(autouse=False, scope="function")
def nifti_img_and_path(tmp_dir):
    img = simulate_nifti_image((20, 20, 10, 5))
    img_path = Path(tmp_dir.name) / "img.nii"
    nib.save(img, img_path)

    yield img, img_path
