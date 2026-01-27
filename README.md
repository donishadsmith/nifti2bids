# NIfTI2BIDS

[![Latest Version](https://img.shields.io/pypi/v/nifti2bids.svg)](https://pypi.python.org/pypi/nifti2bids/)
[![Python Versions](https://img.shields.io/pypi/pyversions/nifti2bids.svg)](https://pypi.python.org/pypi/nifti2bids/)
[![Source Code](https://img.shields.io/badge/Source%20Code-nifti2bids-purple)](https://github.com/donishadsmith/nifti2bids)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Test Status](https://github.com/donishadsmith/nifti2bids/actions/workflows/testing.yaml/badge.svg)](https://github.com/donishadsmith/nifti2bids/actions/workflows/testing.yaml)
[![codecov](https://codecov.io/gh/donishadsmith/nifti2bids/graph/badge.svg?token=PCJ17NA627)](https://codecov.io/gh/donishadsmith/nifti2bids)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Documentation Status](https://readthedocs.org/projects/nifti2bids/badge/?version=stable)](http://nifti2bids.readthedocs.io/en/stable/?badge=stable)


A toolkit for post-hoc BIDS conversion of legacy or unstructured NIfTI datasets. Intended for cases that require custom code and flexibility, such as when NIfTI source files lack consistent naming conventions, organized folder hierarchies, or sidecar metadata. Includes utilities for metadata reconstruction from NIfTI headers, file renaming, neurobehavioral log parsing (for E-Prime and Presentation), and JSON sidecar generation.

## Installation

### Standard Installation
```bash
pip install nifti2bids[all]
```

### Development Version
```bash
git clone --depth 1 https://github.com/donishadsmith/nifti2bids/
cd nifti2bids
pip install -e .[all]
```

## Features

- **File renaming**: Convert arbitrary filenames to BIDS-compliant naming
- **File creation**: Generate `dataset_description.json` and `participants.tsv`
- **Metadata utilities**: Extract header metadata (e.g., TR, orientation, scanner info) and generate slice timing for singleband and multiband acquisitions
- **Log parsing**: Load Presentation (e.g., `.log`) and E-Prime 3 (e.g, `.edat3`, `.txt`) files as DataFrames, or use extractor classes to generate BIDS events for block and event designs:

    | Class | Software | Design | Description |
    |-------|----------|--------|-------------|
    | `PresentationBlockExtractor` | Presentation | Block | Extracts block-level timing with mean RT and accuracy |
    | `PresentationEventExtractor` | Presentation | Event | Extracts trial-level timing with individual responses |
    | `EPrimeBlockExtractor` | E-Prime 3 | Block | Extracts block-level timing with mean RT and accuracy |
    | `EPrimeEventExtractor` | E-Prime 3 | Event | Extracts trial-level timing with individual responses |

- **Auditing**: Generate a table of showing the presence or abscence of certain files for each subject and session

## Quick Start

### Creating BIDS-Compliant Filenames
```python
from nifti2bids.bids import create_bids_file

create_bids_file(
    src_file="101_mprage.nii.gz",
    subj_id="101",
    ses_id="01",
    desc="T1w",
    dst_dir="/data/bids/sub-101/ses-01/anat",
)
```

### Extracting Metadata from NIfTI Headers
```python
from nifti2bids.metadata import get_tr, create_slice_timing, get_image_orientation

tr = get_tr("sub-01_bold.nii.gz")
slice_timing = create_slice_timing(
    "sub-01_bold.nii.gz",
    slice_acquisition_method="interleaved",
    multiband_factor=4,
)
orientation_map, orientation = get_image_orientation("sub-01_bold.nii.gz")
```

### Loading Raw Log Files
```python
from nifti2bids.parsers import (
    load_presentation_log,
    load_eprime_log,
    convert_edat3_to_txt,
)

presentation_df = load_presentation_log("sub-01_task.log", convert_to_seconds=["Time"])

# E-Prime 3: convert .edat3 to text first, or load .txt directly
eprime_txt_path = convert_edat3_to_txt("sub-01_task.edat3")
eprime_df = load_eprime_log(eprime_txt_path, convert_to_seconds=["Stimulus.OnsetTime"])
```

### Creating BIDS Events from Presentation Logs
```python
from nifti2bids.bids import PresentationBlockExtractor
import pandas as pd

extractor = PresentationBlockExtractor(
    "sub-01_task-faces.log",
    block_cue_names=("Face", "Place"),  # Can use regex ("Fa.*", "Pla.*")
    scanner_event_type="Pulse",
    scanner_trigger_code="99",
    convert_to_seconds=["Time"],
    rest_block_codes="crosshair",
    rest_code_frequency="fixed",
    split_cue_as_instruction=True,
)

events_df = pd.DataFrame(
    {
        "onset": extractor.extract_onsets(),
        "duration": extractor.extract_durations(),
        "trial_type": extractor.extract_trial_types(),
        "mean_rt": extractor.extract_mean_reaction_times(),
    }
)
```

### Creating BIDS Events from E-Prime Logs
```python
from nifti2bids.bids import EPrimeEventExtractor
import pandas as pd

extractor = EPrimeEventExtractor(
    "sub-01_task-gonogo.txt",
    trial_types="Go|NoGo",  # Can also use ("Go", "NoGo")
    onset_column_name="Stimulus.OnsetTime",
    procedure_column_name="Procedure",
    trigger_column_name="ScannerTrigger.RTTime",
    convert_to_seconds=[
        "Stimulus.OnsetTime",
        "Stimulus.OffsetTime",
        "ScannerTrigger.RTTime",
    ],
)

events_df = pd.DataFrame(
    {
        "onset": extractor.extract_onsets(),
        "duration": extractor.extract_durations(
            offset_column_name="Stimulus.OffsetTime"
        ),
        "trial_type": extractor.extract_trial_types(),
        "reaction_time": extractor.extract_reaction_times(
            reaction_time_column_name="Stimulus.RT"
        ),
    }
)
```

### Audit BIDS Dataset
```python
from nifti2bids.audit import BIDSAuditor
from nifti2bids.simulate import simulate_bids_dataset

bids_root = simulate_bids_dataset()

auditor = BIDSAuditor(bids_root)
auditor.check_raw_nifti_availability()
auditor.check_raw_sidecar_availability()
auditor.check_events_availability()
auditor.check_preprocessed_nifti_availability()

analysis_dir = bids_root / "first_level"
analysis_sub_dir = analysis_dir / "sub-1" / "ses-1"
analysis_sub_dir.mkdir(parents=True, exist_ok=True)

with open(analysis_sub_dir / "sub-1_task-rest_desc-betas.nii.gz", "w") as f:
    pass

auditor.check_first_level_availability(analysis_dir=analysis_dir, desc="betas")
```

See the [API documentation](https://nifti2bids.readthedocs.io/en/latest/api.html) for full parameter details and additional utilities.
