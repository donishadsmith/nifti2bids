"""
Post-hoc BIDS conversion toolkit for NIfTI datasets without original DICOMs.
----------------------------------------------------------------------------
Documentation can be found at https://nifti2bids.readthedocs.io.

Submodules
----------
bids -- Operations related to initializing and creating BIDS compliant files

io -- Generic operations related to loading NIfTI data

logging -- Set up a logger using ``RichHandler`` as the default handler if a root or
module specific handler is not available

metadata -- Operations related to extracting or creating metadata information from NIfTI images

parsers -- Operations related to standardizing and parsing information logs created by stimulus
presentation software such as Presentation and EPrime

simulate -- Simulate a basic NIfTI image for testing purposes
"""

__version__ = "0.6.2"
