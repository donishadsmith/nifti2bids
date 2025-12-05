API
===

:mod:`nifti2bids.bids`
----------------------
Module for initializing and creating BIDS compliant files.

.. currentmodule:: nifti2bids.bids

.. autosummary::
   :template: function.rst
   :nosignatures:
   :toctree: generated/

   create_bids_file
   create_dataset_description
   save_dataset_description
   create_participant_tsv

.. autosummary::
   :template: class.rst
   :nosignatures:
   :toctree: generated/

   PresentationBlockExtractor
   PresentationEventExtractor
   EPrimeBlockExtractor
   EPrimeEventExtractor

:mod:`nifti2bids.logging`
-------------------------
Module setting up a logger.

.. currentmodule:: nifti2bids.logging

.. autosummary::
   :template: function.rst
   :nosignatures:
   :toctree: generated/

   setup_logger


:mod:`nifti2bids.io`
--------------------
Module for input/output operations on NIfTI files and images.

.. currentmodule:: nifti2bids.io

.. autosummary::
   :template: function.rst
   :nosignatures:
   :toctree: generated/

   load_nifti
   compress_image
   glob_contents
   get_nifti_header
   get_nifti_affine

:mod:`nifti2bids.metadata`
--------------------------
Module containing functions to extract metadata information
from NIfTIs.

.. currentmodule:: nifti2bids.metadata

.. autosummary::
   :template: function.rst
   :nosignatures:
   :toctree: generated/

   determine_slice_axis
   get_hdr_metadata
   get_n_volumes
   get_image_orientation
   get_n_slices
   get_tr
   create_slice_timing
   is_3d_img
   get_scanner_info
   is_valid_date
   parse_date_from_path
   get_entity_value
   infer_task_from_image
   get_recon_matrix_pe
   compute_effective_echo_spacing
   compute_total_readout_time

:mod:`nifti2bids.parsers`
--------------------------
Module containing functions to parse raw logs from stimulus
Presentation and E-Prime 3 software.

.. currentmodule:: nifti2bids.parsers

.. autosummary::
   :template: function.rst
   :nosignatures:
   :toctree: generated/

   convert_edat3_to_tsv
   load_eprime_log
   load_presentation_log
