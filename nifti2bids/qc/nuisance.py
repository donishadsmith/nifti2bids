from pathlib import Path
from typing import Optional

import nibabel as nib, numpy as np
from nilearn.maskers import NiftiMasker

from nifti2bids.io import load_nifti


def compute_global_signal(
    func_file_or_img: str | Path | nib.nifti1.Nifti1Image,
    mask_img_or_file: Optional[str | Path | nib.nifti1.Nifti1Image] = None,
) -> dict[str, np.ndarray]:
    """
    Compute global signal and percent signal change from functional NIfTI image.

    Parameters
    ----------
    func_file_or_img : :obj:`str`, :obj:`Path`, or :obj:`Nifti1Image`
        Path to the functional NIfTI file or a functional NIfTI image.

    mask_img_or_file : :obj:`str`, :obj:`Path`, :obj:`Nifti1Image`, or :obj:`None`, default=None
        Path to the mask NIfTI file or a mask NIfTI image. If None, a mask
        is computed automatically.

    Returns
    -------
    dict[str, np.ndarray]
        A dictionary with the following keys:

        - "global_signal": Raw global signal time series.
        - "global_signal_pct": Percent signal change time series.
    """
    func_file_or_img = load_nifti(func_file_or_img)

    masker = NiftiMasker(
        mask_img=(load_nifti(mask_img_or_file) if mask_img_or_file else None),
        standardize=False,
    )
    masked_data = masker.fit_transform(func_file_or_img)

    global_signal = np.mean(masked_data, axis=1)
    mean_gs = np.mean(global_signal)
    global_signal_pct = ((global_signal - mean_gs) / mean_gs) * 100

    return {
        "global_signal": global_signal,
        "global_signal_pct": global_signal_pct,
    }
