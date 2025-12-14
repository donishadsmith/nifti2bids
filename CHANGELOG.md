# Changelog

Documentation of changes for each version of ``Nifti2Bids``.

**Currently doesn't follow semantic versioning.**

## [0.6.2.post2] - 2025-12-14
- Clean doc string

## [0.6.2.post1] - 2025-12-14
- Add doc string resource

## [0.6.2] - 2025-12-14
- Datetime fix for older Python versions

## [0.6.1] - 2025-12-14
- New function to get file creation date

## [0.6.0] - 2025-12-13
- For PresentationEventExtractor, make duration the "Duration" column
- For event extractors, create reaction time functions and improve logic
- ``duration_column_name`` changed to ``offset_column_name``

## [0.5.0.post1] - 2025-12-12
- Doc string fix

## [0.5.0] - 2025-12-12
- Add parameter for trigger column for E-Prime classes

## [0.4.0] - 2025-12-10
- Change ``convert_edat3_to_tsv`` to ``convert_edat3_to_text``
- Add a ``format`` parameter to ``convert_edat3_to_text`` to allow tsv and csv
- In ``convert_edat3_to_text``, default to csv

## [0.3.6.post1] - 2025-12-05
- Improve docs

## [0.3.6] - 2025-12-04
- Add parameters for rest code frequency and quit code to block extractors for better duration
computation

## [0.3.5] - 2025-12-03
- Iterate over filtered indices instead of entire dataframe
- In the block extractor for E-Prime, put onset_column_name in init for
consistency

## [0.3.4] - 2025-12-03
- Replace use_first_pulse with scanner_trigger_code and scanner_event_code
- Improve series filtering code

## [0.3.3] - 2025-12-03
- Add new trial column parameter to Presentation classes

## [0.3.2] - 2025-12-03
- More flexibility for log parsing

## [0.3.1.post1] - 2025-12-03
- Add directive to doc

## [0.3.1] - 2025-12-02
- Fix default value issue

## [0.3.0] - 2025-12-02
- Add new parameters to the Presentation Extractor classes

## [0.2.11] - 2025-12-01
- Ensure starting block indices are sorted for EPrimeBlockExtractor

## [0.2.10] - 2025-12-01
- Add new classes and functions for getting
onsets, durations, trial types and responses from
EPrime and Presentation logs

## [0.2.9] - 2025-11-28
- Change `get_date_from_filename` to `parse_date_from_path`

## [0.2.8] - 2025-11-28
- Added assertions

## [0.2.7] - 2025-11-28
- Add function to convert edat3 to tsv

## [0.2.6] - 2025-11-27
- Add new parameter for parser functions

## [0.2.5] - 2025-11-27
- Add E-Prime 3 parser
- Convert the `convert_to_seconds` from bool to a iterable
- Add `convert_to_seconds` to the presentation to events function

## [0.2.4] - 2025-11-25
- Add new function to convert presentation logs to bids event files
- Simplify some function parameters

## [0.2.3] - 2025-11-21
- Add new slice acquisition methods
- Remove ability to use indivisible multiband factor since ordering
may depend on software version for GE.

## [0.2.2] - 2025-11-21
- Add fallback trt parameter

## [0.2.1] - 2025-11-20
- Fix potential edge case in parsing function

## [0.2.0] - 2025-11-20
- Add parser module

## [0.1.9] - 2025-11-19
- Parameter name change for ``infer_task_for_image`` and give ability
to handle different mapping
- Decorator to check if NIfTI image is raw

## [0.1.8] - 2025-11-15
- Return acquisition parameter to slice_acquisition_method

## [0.1.7] - 2025-11-15
- Added new functions and changed accepted values for for parameters relates to
axes (from x, y, z to i, j, k)

## [0.1.6] - 2025-11-14
- Typing and docs fixes, including change for ``create_affine`` to
accept tuples and lists

## [0.1.5] - 2025-11-13
- Doc and type fixes plus accept Path objects

## [0.1.4] - 2025-11-12
- Change parameter name from "interleave_pattern" to "interleaved_pattern"

## [0.1.3] - 2025-11-11
- Add logic for indivisible multiband factor for all cases except for philips
interleaved pattern

## [0.1.2] - 2025-11-11
- Inspired by the following article to write better code: https://pmc.ncbi.nlm.nih.gov/articles/PMC5274797/
    - Essentially add Philip's as an interleave pattern while only retaining sequential or interleave for acquisition
- Some parameter name changing

## [0.1.1] - 2025-11-10
- Add Philip's specific interleaved order and multiband slice acquisition
- Other parameter name changes

## [0.1.0] - 2025-11-06
- Change utils module name to metadata
- Change logger module name to logging
- Create new bids module and move ``create_bids_file``, ``create_dataset_description``, ``save_dataset_description``, and ``create_participant_tsv`` to it
- Add ``save_df`` and ``return_df`` parameters to ``create_participant_tsv``

## [0.0.9] - 2025-11-05
- Add ``slice_axis`` parameter to ``create_slice_timing``

## [0.0.8] - 2025-11-05
- Change function and parameter names ending in "dim" to "axis"
- Change custom exception name
- Add new function to infer task based on number of volumes
- Add level parameter to ``setup_logger``
- Rename package from ``BidsPrep`` to ``Nifti2Bids``

## [0.0.7] - 2025-11-05
- Add  ``get_n_volumes`` function and change custom exceptions names

## [0.0.6] - 2025-11-04
- Add exception to ``create_slice_timing`` for safety

## [0.0.5] - 2025-11-04
- Fix ``create_bids_filename`` to not add "desc"
- Return numeric values as regular Python integers and float
- Add function to extract entity value
- Change ``destination_dir`` and ``output_dir`` to ``dst_dir``

## [0.0.4] - 2025-11-04
- Add function to create participants tsv file.
- ``get_files`` changed to ``glob_contents``.

## [0.0.3] - 2025-11-04
- Add function for extracting date from filenames.

## [0.0.2] - 2025-11-04
- Change output of ``create_slice_timing`` from a dictionary to a list.

## [0.0.1] - 2025-11-03
- First non-alpha release of ``BIDSPrep``.
