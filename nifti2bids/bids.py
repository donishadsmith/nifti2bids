"""Module for creating BIDS compliant files."""

import json, itertools
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
import numpy as np

from nifti2bids.logging import setup_logger
from nifti2bids.io import _copy_file, glob_contents
from nifti2bids.parsers import (
    load_eprime_log,
    load_presentation_log,
    _convert_time,
    _is_float,
)

LGR = setup_logger(__name__)


def create_bids_file(
    nifti_file: str | Path,
    subj_id: str | int,
    desc: str,
    ses_id: Optional[str | int] = None,
    task_id: Optional[str] = None,
    run_id: Optional[str | int] = None,
    dst_dir: str | Path = None,
    remove_src_file: bool = False,
    return_bids_filename: bool = False,
) -> Path | None:
    """
    Create a BIDS compliant filename with required and optional entities.

    Parameters
    ----------
    nifti_file: :obj:`str` or :obj:`Path`
        Path to NIfTI image.

    sub_id: :obj:`str` or :obj:`int`
        Subject ID (i.e. 01, 101, etc).

    desc: :obj:`str`
        Description of the file (i.e., T1w, bold, etc).

    ses_id: :obj:`str` or :obj:`int` or :obj:`None`, default=None
        Session ID (i.e. 001, 1, etc). Optional entity.

    ses_id: :obj:`str` or :obj:`int` or :obj:`None`, default=None
        Session ID (i.e. 001, 1, etc). Optional entity.

    task_id: :obj:`str` or :obj:`None`, default=None
        Task ID (i.e. flanker, n_back, etc). Optional entity.

    run_id: :obj:`str` or :obj:`int` or :obj:`None`, default=None
        Run ID (i.e. 001, 1, etc). Optional entity.

    dst_dir: :obj:`str`, :obj:`Path`, or :obj:`None`, default=None
        Directory name to copy the BIDS file to. If None, then the
        BIDS file is copied to the same directory as

    remove_src_file: :obj:`str`, default=False
        Delete the source file if True.

    return_bids_filename: :obj:`str`, default=False
        Returns the full BIDS filename if True.

    Returns
    -------
    Path or None
        If ``return_bids_filename`` is True, then the BIDS filename is
        returned.

    Note
    ----
    There are additional entities that can be used that are
    not included in this function.
    """
    bids_filename = f"sub-{subj_id}_ses-{ses_id}_task-{task_id}_" f"run-{run_id}_{desc}"
    bids_filename = _strip_none_entities(bids_filename)

    ext = f"{str(nifti_file).partition('.')[-1]}"
    bids_filename += f"{ext}"
    bids_filename = (
        Path(nifti_file).parent / bids_filename
        if dst_dir is None
        else Path(dst_dir) / bids_filename
    )

    _copy_file(nifti_file, bids_filename, remove_src_file)

    return bids_filename if return_bids_filename else None


def _strip_none_entities(bids_filename: str | Path) -> str:
    """
    Removes entities with None in a BIDS compliant filename.

    Parameters
    ----------
    bids_filename: :obj:`str` or :obj:`Path`
        The BIDS filename.

    Returns
    -------
    str
        BIDS filename with entities ending in None removed.

    Example
    -------
    >>> from nifti2bids.bids import _strip_none_entities
    >>> bids_filename = "sub-101_ses-None_task-flanker_bold.nii.gz"
    >>> _strip_none_entities(bids_filename)
        "sub-101_task-flanker_bold.nii.gz"
    """
    basename, _, ext = str(bids_filename).partition(".")
    retained_entities = [
        entity for entity in basename.split("_") if not entity.endswith("-None")
    ]

    return f"{'_'.join(retained_entities)}.{ext}"


def create_dataset_description(
    dataset_name: str, bids_version: str = "1.0.0"
) -> dict[str, str]:
    """
    Generate a dataset description dictionary.

    Creates a dictionary containing the name and BIDs version of a dataset.

    .. versionadded:: 0.34.1

    Parameters
    ----------
    dataset_name: :obj:`str`
        Name of the dataset.

    bids_version: :obj:`str`,
        Version of the BIDS dataset.

    derivative: :obj:`bool`, default=False
        Determines if "GeneratedBy" key is added to dictionary.

    Returns
    -------
    dict[str, str]
        The dataset description dictionary
    """
    return {"Name": dataset_name, "BIDSVersion": bids_version}


def save_dataset_description(
    dataset_description: dict[str, str], dst_dir: str | Path
) -> None:
    """
    Save a dataset description dictionary.

    Saves the dataset description dictionary as a file named "dataset_description.json" to the
    directory specified by ``output_dir``.

    Parameters
    ----------
    dataset_description: :obj:`dict`
        The dataset description dictionary.

    dst_dir: :obj:`str` or :obj:`Path`
        Path to save the JSON file to.
    """
    with open(Path(dst_dir) / "dataset_description.json", "w", encoding="utf-8") as f:
        json.dump(dataset_description, f)


def create_participant_tsv(
    bids_dir: str | Path, save_df: bool = False, return_df: bool = True
) -> pd.DataFrame | None:
    """
    Creates a basic participant dataframe for the "participants.tsv" file.

    Parameters
    ----------
    bids_dir: :obj:`str` or :obj:`Path`
        The root of BIDS compliant directory.

    save_df: :obj:`bool`, bool=False
        Save the dataframe to the root of the BIDS compliant directory.

    return_df: :obj:`str`
        Returns dataframe if True else return None.

    Returns
    -------
    pd.DataFrame or None
        The dataframe if ``return_df`` is True.
    """
    participants = [folder.name for folder in glob_contents(bids_dir, "*sub-*")]
    df = pd.DataFrame({"participant_id": participants})

    if save_df:
        df.to_csv(Path(bids_dir) / "participants.tsv", sep="\t", index=None)

    return df if return_df else None


def _process_log_or_df(
    log_or_df: str | Path | pd.DataFrame,
    convert_to_seconds: list[str] | None,
    initial_column_headers: tuple[str],
    divisor: float | int,
    software: Literal["Presentation", "E-Prime"],
):
    """
    Processes the event log from a neurobehavioral software.

    Parameters
    ----------
    log_or_df: :obj:`str`, :obj:`Path`, :obj:`pd.DataFrame`
        The log or DataFrame of event informaiton from a neurobehavioral software.

    convert_to_seconds: :obj:`list[str]` or :obj:`None`, default=None
        Convert the time resolution of the specified columns.

    initial_column_headers: :obj:`tuple[str]`
        The initial column headers for data. Only used when
        ``log_or_df`` is a file path.

    divisor: :obj:`float` or :obj:`int`
        Value to divide columns specified in ``convert_to_seconds`` by.

    software: :obj:`Literal["Presentation", "EPrime"]
        The specific neurobehavioral software.

    Returns
    -------
    pandas.DataFrame
        A dataframe of the task information.
    """
    loader = {"Presentation": load_presentation_log, "E-Prime": load_eprime_log}

    if not isinstance(log_or_df, pd.DataFrame):
        df = loader[software](
            log_or_df,
            convert_to_seconds=convert_to_seconds,
            initial_column_headers=tuple(initial_column_headers),
        )
    elif convert_to_seconds:
        df = _convert_time(
            log_or_df, convert_to_seconds=convert_to_seconds, divisor=divisor
        )
    else:
        df = log_or_df

    return df.replace("", np.nan, inplace=False)


def _get_starting_block_indices(
    log_df: pd.DataFrame, trial_column_name: str, block_cue_codes: tuple[str]
) -> list[int]:
    """
    Get starting indices for blocks.

    Parameters
    ----------
    log_df: :obj:`pandas.DataFrame`
        DataFrame of neurobehavioral log data.

    trial_column_name: :obj:`str`
        Name of the column containing the trial information.

    block_cue_codes: :obj:`tuple[str]`
        The names of the trial types.

    Returns
    -------
    list[int]
        The starting index of each block.
    """
    trial_series = log_df[trial_column_name]

    # Get the starting index for each block via grouping indices with the same
    trial_list = trial_series.tolist()
    starting_block_indices = []
    current_index = 0
    for _, group in itertools.groupby(trial_list):
        starting_block_indices.append(current_index)
        current_index += len(list(group))

    starting_block_indices = set(starting_block_indices)
    for trial_type in trial_series.unique():
        if trial_type not in block_cue_codes:
            trial_indxs = trial_series[trial_series == trial_type].index.tolist()
            starting_block_indices = starting_block_indices.difference(trial_indxs)

    # Remove empty
    missing_indices = trial_series[trial_series.isna()].index.tolist()
    starting_block_indices = starting_block_indices.difference(missing_indices)

    return sorted(list(starting_block_indices))


def _get_next_block_index(
    trial_series: pd.Series,
    block_start_indx: int,
    rest_block_code: Optional[str],
    rest_code_frequency: Literal["fixed", "variable"],
    block_cue_codes: tuple[str],
    quit_code: Optional[str] = None,
) -> int:
    """
    Get the starting index for each block.

    Parameters
    ----------
    trial_series: :obj:`pandas.Series`
        A Pandas Series of the column containing the trial type information.

    block_start_indx: :obj:`int`
        The current row index.

    rest_block_code: :obj:`str` or :obj:`None`
        The name of the rest block.

    rest_code_frequency: :obj:`Literal["fixed", "variable"]`, default="fixed"
        Frequency of the rest block. For "fixed", the rest code is assumed to
        appear between each trial or at least each trial. For "variable",
        it is assumed that the rest code does not appear between each
        trial.

    block_cue_codes: :obj:`tuple[str]`
        The names of the trial types. When used, identifies
        the indices of all trial types minus the indices
        corresponding to the current trial type. Used when
        ``rest_block_code`` is not None and ``rest_code_frequency``
        is not "fixed".

    quit_code: :obj:`str` or :obj:`None`, default=None
        The quit code. Ideally, this should be a unique code.

    Returns
    -------
    int
        The starting index of the next block.
    """
    curr_trial = trial_series.at[block_start_indx]
    filtered_trial_series = trial_series[trial_series.index > block_start_indx]
    filtered_trial_series = filtered_trial_series.astype(str)

    if rest_block_code and rest_code_frequency == "fixed":
        target_codes = [rest_block_code] + ([quit_code] if quit_code else [])
        next_block_indxs = filtered_trial_series[
            filtered_trial_series.isin(tuple(target_codes))
        ].index.tolist()
    else:
        target_block_names = set(tuple(block_cue_codes))
        target_block_names.discard(curr_trial)
        additional_codes = []
        additional_codes += (
            [rest_block_code]
            if rest_block_code and rest_code_frequency == "variable"
            else []
        )
        additional_codes += [quit_code] if quit_code else []
        target_block_names = tuple(list(target_block_names) + additional_codes)
        next_block_indxs = filtered_trial_series[
            filtered_trial_series.isin(target_block_names)
        ].index.tolist()

    return next_block_indxs[0] if next_block_indxs else block_start_indx


class LogExtractor(ABC):
    """Abstract Base Class for Extractors."""

    @abstractmethod
    def extract_onsets(self):
        """Extract onsets."""

    @abstractmethod
    def extract_durations(self):
        """Extract durations."""

    @abstractmethod
    def extract_trial_types(self):
        """Extract the trial types."""


class BlockExtractor(LogExtractor):
    """Abstract Base Class for Block Extractors."""

    @abstractmethod
    def extract_mean_reaction_times(self):
        """Extract mean reaction times for each block."""

    @abstractmethod
    def extract_mean_accuracies(self):
        """Extract mean accuracy for each block."""


class EventExtractor(LogExtractor):
    """Abstract Base Class for Event Extractors."""

    @abstractmethod
    def extract_reaction_times(self):
        """Extract reaction time for each trial."""


class PresentationExtractor:
    """
    Base class for Presentation log extractors.

    Provides shared initialization and extraction logic for both block
    and event design extractors.
    """

    def __init__(
        self,
        log_or_df: str | Path | pd.DataFrame,
        condition_codes: tuple[str],
        scanner_event_type: str,
        scanner_trigger_code: str,
        trial_column_name: str = "Code",
        convert_to_seconds: Optional[list[str]] = None,
        initial_column_headers: tuple[str] = ("Trial", "Event Type"),
        n_discarded_volumes: int = 0,
        tr: Optional[float | int] = None,
    ):

        df = _process_log_or_df(
            log_or_df,
            convert_to_seconds,
            initial_column_headers,
            divisor=1e4,
            software="Presentation",
        )
        self.trial_column_name = trial_column_name
        self.scanner_event_type = scanner_event_type
        self.scanner_trigger_code = scanner_trigger_code

        scanner_start_index_list = df.loc[
            (df["Event Type"] == self.scanner_event_type)
            & (df["Code"] == self.scanner_trigger_code)
        ].index.tolist()

        if scanner_start_index_list:
            scanner_start_index = scanner_start_index_list[0]
            self.scanner_start_time = df.loc[scanner_start_index, "Time"]
            df = df.loc[scanner_start_index:, :]
            self.df = df.reset_index(inplace=False)
        else:
            LGR.warning(
                f"No scanner trigger under 'Event Type': {self.scanner_event_type} "
                f"and 'Code': {self.scanner_trigger_code} "
            )
            self.scanner_start_time = None
            self.df = df

        self.n_discarded_volumes = n_discarded_volumes
        self.tr = tr
        if self.n_discarded_volumes > 0:
            if not self.tr:
                raise ValueError(
                    "``tr`` must be provided when ``n_discarded_volumes`` is greater than 0."
                )

            if not self.scanner_start_time:
                raise ValueError(
                    "``scanner_start_time`` is None so time shift cannot be added."
                )

            self.scanner_start_time += self.n_discarded_volumes * self.tr

    def _extract_onsets(
        self, row_indices: list[str], scanner_start_time: Optional[float | int]
    ) -> list[float]:
        """Extract onset times for each block or event."""
        if scanner_start_time is not None:
            self.scanner_start_time = scanner_start_time

        if self.scanner_start_time is None:
            raise ValueError(
                "A value for `scanner_start_time` needs to be given "
                "since ``self.scanner_event_type`` and ``self.scanner_trigger_code`` "
                "did not identify a time."
            )

        return [
            self.df.loc[index, "Time"] - self.scanner_start_time
            for index in row_indices
        ]

    def _extract_trial_types(self, row_indices: list[int]) -> list[str]:
        """Extract trial types for each block or event."""
        return [self.df.loc[index, self.trial_column_name] for index in row_indices]


class PresentationBlockExtractor(PresentationExtractor, BlockExtractor):
    """
    Extract onsets, durations, and trial types from Presentation logs using a block design.

    Parameters
    ----------
    log_or_df: :obj:`str`, :obj:`Path`, :obj:`pandas.DataFrame`
        The Presentation log as a file path or the Presentation DataFrame
        returned by :code:`nifti2bids.parsers.load_presentation_log`.

        .. important::
           If a text file is used, data are assumed to have at least one element
           that is an digit or float during parsing.

    block_cue_codes: :obj:`tuple[str]`
        The names of the block cue codes (i.e "Face", "Place").

    trial_column_name: :obj:`str`, default="Code"
        Name of the column containing the trial types.

    scanner_event_type: :obj:`str`
        The event type in the "Event Type" column the scanner
        trigger is listed under (e.g., "Pulse", "Response", "Picture", etc).

    scanner_trigger_code: :obj:`str`
        Code listed under "Code" for the scanner start (e.g., "54", "99", "trigger).
        Used with ``scanner_event_type`` to compute the onset
        times of the trials relative to the scanner start time then
        clip the dataframe to ensure that no trials
        before the start of the scanner is initiated.

        .. note::
           Uses the first index of the rows in the dataframe with values
           provided for ``scanner_event_type`` and ``scanner_trigger_code``.

    convert_to_seconds: :obj:`list[str]` or :obj:`None`, default=None
        Convert the time resolution of the specified columns from 0.1ms to seconds.

        .. important:: Recommend time resolution of the "Time" column to be converted.

    initial_column_headers: :obj:`tuple[str]`, default=("Trial", "Event Type")
        The initial column headers for data. Only used when
        ``log_or_df`` is a file path.

    n_discarded_volumes: :obj:`int`, default=0
        Number of non-steady state scans discarded by the scanner at the start of the sequence.

        .. important::
           - Only used when ``trigger_column_name`` is specified.
           - Only set this parameter if scanner trigger is sent **before** these volumes are
             acquired so that the start time of the first retained volume is shifted forward
             by (``n_discarded_volumes * tr``). If the scanner sends trigger **after**
             discarding the volumes, do not set this parameter.
             `Explanation from Neurostars <https://neurostars.org/t/how-to-sync-time-from-e-prime-behavior-data-with-fmri/6887>`_.

    tr: :obj:`float`, :obj:`int`, or :obj:`None`, default=None
        The repetition time provided in seconds if data was converted to seconds or
        in ten thousand seconds if not converted.

    rest_block_code: :obj:`str` or :obj:`None`, default=None
        The name of the code for the rest block. Used when a resting state
        block is between the events to compute the correct block duration.
        If None, the block duration will be computed based on the starting
        index of the block cue codes given by ``block_cue_codes``. If specified
        and ``rest_code_frequency`` is "variable", will be used with
        ``block_cue_codes`` to compute the correct duration.

    rest_code_frequency: :obj:`Literal["fixed", "variable"]`, default="fixed"
        Frequency of the rest block. For "fixed", the rest code is assumed to
        appear between each block or at least each block. For "variable",
        it is assumed that the rest code does not appear between each
        block.

    quit_code: :obj:`str` or :obj:`None`, default=None
        The quit code. Suggest to use in cases when a quit code, as opposed
        to a rest code, is preceded by a trial block. Ideally, this should
        be a unique code.

    Attributes
    ----------
    df: :obj:`pandas.DataFrame`
        DataFrame containing the log data. If the scanner trigger is identified
        using ``scanner_event_type`` and ``scanner_trigger_code``, then rows
        preceding the first scanner are dropped and the index is reset.

    block_cue_codes: :obj:`tuple[str]`
        The names of the block cue codes.

    scanner_event_type: :obj:`str`
        Event type of scanner trigger.

    scanner_trigger_code: :obj:`str`
        Code for the scanner trigger.

    trial_column_name: :obj:`str`
        Name of column containing the trial types.

    n_discarded_scans: :obj:`int`
        Number of non-steady state scans discarded by scanner.

    tr: :obj:`float`, :obj:`int`, or :obj:`None`
        The repetition time.

    scanner_start_time: :obj:`float` or :obj:`None`
        Time when scanner sends the pulse. If ``n_discarded_volumes``
        is not 0 and ``tr`` is specified, then this time will be
        shifted forward (``scanner_start_time = scanner_start_time + n_discarded_volumes * tr``)
        to reflect the time when the first steady state volume was retained. Otherwise, the time
        extracted from the log data is assumed to be the time when the first steady state
        volume was retained.

    starting_block_indices: :obj:`list[int]`
        The indices of when each trial block of interest (specified by ``block_cue_codes``)
        begins.

    rest_block_code: :obj:`str` or :obj:`None`
        The name of the code for the rest block.

    rest_code_frequency: :obj:`Literal["fixed", "variable"]`
        Frequency of the rest block.

    quit_code: :obj:`str` or :obj:`None`
        The quit code.

    Example
    -------
    >>> import pandas as pd
    >>> from nifti2bids.bids import PresentationBlockExtractor
    >>> extractor = PresentationBlockExtractor(
    ...     log_file,
    ...     block_cue_codes=("Face", "Place"),
    ...     scanner_event_type="Pulse",
    ...     scanner_trigger_code="99",
    ...     convert_to_seconds=["Time"],
    ...     rest_block_code="crosshair",
    ... )
    >>> events = {}
    >>> events["onset"] = extractor.extract_onsets()
    >>> events["duration"] = extractor.extract_durations()
    >>> events["trial_type"] = extractor.extract_trial_types()
    >>> # Mean reaction time for all trials with a response
    >>> events["mean_rt"] = extractor.extract_mean_reaction_times()
    >>> # Mean reaction time for correct trials only
    >>> response_map = {"hit": 1, "miss": 0}
    >>> events["mean_rt_correct"] = extractor.extract_mean_reaction_times(
    ...     response_map=response_map,
    ...     response_type="correct",
    ... )
    >>> # Mean accuracy
    >>> events["mean_accuracy"] = extractor.extract_mean_accuracies(response_map=response_map)
    >>> df = pd.DataFrame(events)
    """

    def __init__(
        self,
        log_or_df,
        block_cue_codes,
        scanner_event_type,
        scanner_trigger_code,
        trial_column_name="Code",
        convert_to_seconds=None,
        initial_column_headers=("Trial", "Event Type"),
        n_discarded_volumes=0,
        tr=None,
        rest_block_code=None,
        rest_code_frequency="fixed",
        quit_code=None,
    ):
        super().__init__(
            log_or_df,
            block_cue_codes,
            scanner_event_type,
            scanner_trigger_code,
            trial_column_name,
            convert_to_seconds,
            initial_column_headers,
            n_discarded_volumes,
            tr,
        )

        self.block_cue_codes = block_cue_codes

        assert rest_code_frequency in [
            "fixed",
            "variable",
        ], "`rest_code_frequency` must be either 'fixed' or 'variable'."

        self.rest_block_code = rest_block_code
        self.rest_code_frequency = rest_code_frequency
        self.quit_code = quit_code

        self.starting_block_indices = _get_starting_block_indices(
            self.df, self.trial_column_name, self.block_cue_codes
        )

    def extract_onsets(
        self, scanner_start_time: Optional[float | int] = None
    ) -> list[float]:
        """
        Extract the onset times for each event.

        Onset is calculated as the difference between the event time and
        the scanner start time (e.g. first pulse).

        Parameters
        ----------
        scanner_start_time: :obj:`float`, :obj:`int`, or :obj:`None`, default=None
            The start time for the scanner.

            .. important::
               Scanner start time will be detected during class initialization, unless
               the ``self.scanner_event_code`` and ``self.scanner_trigger_code`` does
               not return an index.

        Returns
        -------
        list[float]
            A list of onset times for each block.
        """
        return self._extract_onsets(self.starting_block_indices, scanner_start_time)

    def extract_durations(self) -> list[float]:
        """
        Extract the duration for each block.

        Duration is computed as the difference between the start of the block
        and the start of the next block (either a rest block or some task block).

        Returns
        -------
        list[float]
            A list of durations for each block.
        """
        durations = []
        for block_start_indx in self.starting_block_indices:
            block_start_row = self.df.loc[block_start_indx, :]
            block_end_indx = _get_next_block_index(
                trial_series=self.df[self.trial_column_name],
                block_start_indx=block_start_indx,
                rest_block_code=self.rest_block_code,
                rest_code_frequency=self.rest_code_frequency,
                block_cue_codes=self.block_cue_codes,
                quit_code=self.quit_code,
            )

            block_end_row = self.df.loc[block_end_indx, :]
            durations.append((block_end_row["Time"] - block_start_row["Time"]))

        return durations

    def extract_trial_types(self) -> list[str]:
        """
        Extract the trial type for each block.

        Returns
        -------
        list[str]
            A list of trial types for each block.
        """
        return self._extract_trial_types(self.starting_block_indices)

    def _get_block_trials(
        self,
        block_start_indx: int,
        response_trial_codes: Optional[tuple[str]] = None,
    ) -> pd.DataFrame:
        """
        Get trials within a block. Filtered by ``response_trial_codes`` if not None.

        Parameters
        ----------
        block_start_indx: :obj:`int`
            The starting index of the block.

        response_trial_codes: :obj:`tuple[str]` or :obj:`None`, default=None
            The codes identifying trials to include. If None, includes all
            trials after the block cue row.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the filtered block trials (reset index).
        """
        block_end_indx = _get_next_block_index(
            trial_series=self.df[self.trial_column_name],
            block_start_indx=block_start_indx,
            rest_block_code=self.rest_block_code,
            rest_code_frequency=self.rest_code_frequency,
            block_cue_codes=self.block_cue_codes,
            quit_code=self.quit_code,
        )

        # Note: iloc excludes end index
        if response_trial_codes is not None:
            block_df = self.df.iloc[block_start_indx:block_end_indx, :]
            block_df = block_df[
                block_df[self.trial_column_name].isin(response_trial_codes)
            ]
        else:
            # Skip first row (cue) since responses are in subsequent rows
            block_df = self.df.iloc[(block_start_indx + 1) : block_end_indx, :]

        return block_df.reset_index()

    def _extract_rts_and_responses(
        self,
        block_df: pd.DataFrame,
    ) -> tuple[list[float], list[str]]:
        """
        Extract reaction times and responses for Picture events in a block.

        Parameters
        ----------
        block_df: :obj:`pandas.DataFrame`
            DataFrame containing the block trials (should be reset index).

        Returns
        -------
        tuple[list[float], list[str]]
            A tuple of (reaction_times, responses) for each Picture event.
        """
        picture_indices = block_df[block_df["Event Type"] == "Picture"].index
        reaction_times = []
        responses = []

        for row_indx in picture_indices:
            stimulus_row = block_df.loc[row_indx, :]
            trial_num = stimulus_row["Trial"]
            response_row = block_df[
                (block_df["Trial"] == trial_num)
                & (block_df["Event Type"] == "Response")
            ]

            response = stimulus_row["Stim Type"]
            if not response_row.empty:
                reaction_time = float(
                    response_row.iloc[0]["Time"] - stimulus_row["Time"]
                )
            else:
                reaction_time = np.nan

            reaction_times.append(reaction_time)
            responses.append(response)

        return reaction_times, responses

    def extract_mean_reaction_times(
        self,
        response_map: Optional[dict[str, int]] = None,
        response_type: Literal["correct", "incorrect"] = "correct",
        response_trial_codes: Optional[tuple[str]] = None,
    ) -> list[float]:
        """
        Extract mean reaction times for each block.

        Parameters
        ----------
        response_map: :obj:`dict[str, int]`
            A dictionary mapping response codes, from "Stim Type" column
            (ie. "hit", "miss", "other", "false alarm", "incorrect"), to accuracy
            values (0 for incorrect, 1 for correct). If provided, reaction times
            are filtered based on ``response_type``. If None, all reaction times
            with a response are included.

        response_type: :obj:`Literal["correct", "incorrect"]`, default="correct"
            Whether to compute mean RT for correct or incorrect trials.
            Only used when ``response_map`` is provided.

        response_trial_codes: :obj:`tuple[str]` or :obj:`None`, default=None
            The codes identifying trials to include. If None, includes all
            trials after the block cue row. For reaction, this should typically
            be set to only include trials where a response is expected
            (e.g., "Go" but not "NoGo").

            .. important::
               In cases such as switch tasks, where ``block_cue_codes`` are
               non-responsive blocks indicating a switch, then this parameter
               should include all trial names that are not in ``block_cue_codes``
               so that reaction time is computed for non-cue codes.

        Returns
        -------
        list[float]
            A list of mean reaction times for each block.

        Notes
        -----
        The reaction time is computed for the first response only.

        Example
        -------
        >>> # Mean RT for all trials with a response
        >>> mean_rts = extractor.extract_mean_reaction_times()
        >>> # Mean RT for correct trials only
        >>> response_map = {"hit": 1, "miss": 0}
        >>> mean_rts = extractor.extract_mean_reaction_times(
        ...     response_map=response_map,
        ...     response_type="correct",
        ... )
        """
        mean_reaction_times = []
        for block_start_indx in self.starting_block_indices:
            block_df = self._get_block_trials(block_start_indx, response_trial_codes)
            reaction_times, responses = self._extract_rts_and_responses(block_df)

            if response_map is not None:
                target_correctness = 1 if response_type == "correct" else 0
                filtered_rts = [
                    rt
                    for rt, resp in zip(reaction_times, responses)
                    if response_map.get(resp) == target_correctness
                ]
                mean_reaction_times.append(
                    np.nanmean(filtered_rts)
                    if len(filtered_rts) > 0 and not np.all(np.isnan(filtered_rts))
                    else np.nan
                )
            else:
                mean_reaction_times.append(
                    np.nanmean(reaction_times)
                    if len(reaction_times) > 0 and not np.all(np.isnan(reaction_times))
                    else np.nan
                )

        return mean_reaction_times

    def extract_mean_accuracies(
        self,
        response_map: dict[str, int],
        response_trial_codes: Optional[tuple[str]] = None,
    ) -> list[float]:
        """
        Extract mean accuracy for each block.

        Parameters
        ----------
        response_map: :obj:`dict[str, int]`
            A dictionary mapping response codes, from "Stim Type" column
            (ie. "hit", "miss", "other", "false alarm", "incorrect"), to accuracy
            values (0 for incorrect, 1 for correct).

        response_trial_codes: :obj:`tuple[str]` or :obj:`None`, default=None
            The codes identifying trials to include. If None, includes all
            trials after the block cue row. For reaction, this should typically
            be set to only include trials where a response is expected
            (e.g., "Go" but not "NoGo").

            .. important::
               In cases such as switch tasks, where ``block_cue_codes`` are
               non-responsive blocks indicating a switch, then this parameter
               should include all trial names that are not in ``block_cue_codes``
               so that reaction time is computed for non-cue codes.

        Returns
        -------
        list[float]
            A list of mean accuracies for each block.

        Example
        -------
        >>> response_map = {"hit": 1, "miss": 0}
        >>> mean_accs = extractor.extract_mean_accuracies(response_map=response_map)
        """
        mean_accuracies = []
        for block_start_indx in self.starting_block_indices:
            block_df = self._get_block_trials(block_start_indx, response_trial_codes)
            _, responses = self._extract_rts_and_responses(block_df)

            converted_responses = [response_map[resp] for resp in responses]

            if len(converted_responses) > 0:
                mean_accuracies.append(
                    sum(converted_responses) / len(converted_responses)
                )
            else:
                mean_accuracies.append(np.nan)

        return mean_accuracies


class PresentationEventExtractor(PresentationExtractor, EventExtractor):
    """
    Extract onsets, durations, trial types, reaction times, and responses
    from Presentation logs using an event design.

    Parameters
    ----------
    log_or_df: :obj:`str`, :obj:`Path`, :obj:`pandas.DataFrame`
        The Presentation log as a file path or the Presentation DataFrame
        returned by :code:`nifti2bids.parsers.load_presentation_log`.

        .. important::
           If a text file is used, data are assumed to have at least one element
           that is an digit or float during parsing.

    trial_types: :obj:`tuple[str]`
        The names of the trial types (i.e "congruentleft", "seen").

        .. note::
           If your block design does not include a rest block or
           crosshair code, include the code immediately after the
           final block.

    scanner_event_type: :obj:`str`
        The event type in the "Event Type" column the scanner
        trigger is listed under (e.g., "Pulse", "Response", "Picture", etc).

    scanner_trigger_code: :obj:`str`
        Code listed under "Code" for the scanner start (e.g., "54", "99", "trigger).
        Used with ``scanner_event_type`` to compute the onset
        times of the trials relative to the scanner start time then
        clip the dataframe to ensure that no trials before the start
        of the scanner is initiated.

        .. note::
           Uses the first index of the rows in the dataframe with values
           provided for ``scanner_event_type`` and ``scanner_trigger_code``.

    trial_column_name: :obj:`str`, default="Code"
        Name of the column containing the trial types.

    convert_to_seconds: :obj:`list[str]` or :obj:`None`, default=None
        Convert the time resolution of the specified columns from 0.1ms to seconds.

        .. important::
           Recommend time resolution of the "Time" column and "Duration" column
           to be converted.

    initial_column_headers: :obj:`tuple[str]`, default=("Trial", "Event Type")
        The initial column headers for data. Only used when
        ``log_or_df`` is a file path.

    n_discarded_volumes: :obj:`int`, default=0
        Number of non-steady state scans discarded by the scanner at the start of the sequence.

        .. important::
           - Only used when ``trigger_column_name`` is specified.
           - Only set this parameter if scanner trigger is sent **before** these volumes are
             acquired so that the start time of the first retained volume is shifted forward
             by (``n_discarded_volumes * tr``). If the scanner sends trigger **after**
             discarding the volumes, do not set this parameter.
             `Explanation from Neurostars <https://neurostars.org/t/how-to-sync-time-from-e-prime-behavior-data-with-fmri/6887>`_.

    tr: :obj:`float`, :obj:`int`, or :obj:`None`, default=None
        The repetition time provided in seconds if data was converted to seconds or
        in ten thousand seconds if not converted.

    Attributes
    ----------
    df: :obj:`pandas.DataFrame`
        DataFrame containing the log data. If the scanner trigger is identified
        using ``scanner_event_type`` and ``scanner_trigger_code``, then rows
        preceeding the first scanner are dropped and the index is reset.

    trial_types: :obj:`tuple[str]`
        The names of the trial types.

    scanner_event_type: :obj:`str`
        Event type of scanner trigger.

    scanner_trigger_code: :obj:`str`
        Code for the scanner trigger.

    trial_column_name: :obj:`str`
        Name of column containing the trial types.

    n_discarded_scans: :obj:`int`
        Number of non-steady state scans discarded by scanner.

    tr: :obj:`float`, :obj:`int`, or :obj:`None`
        The repetition time.

    scanner_start_time :obj:`float` or :obj:`None`
        Time when scanner sends the pulse. If ``n_discarded_volumes``
        is not 0 and ``tr`` is specified, then this time will be
        shifted forward (``scanner_start_time = scanner_start_time + n_discarded_volumes * tr ``)
        to reflect the time when the first steady state volume was retained. Otherwise, the time
        extracted from the log data is assumed to be the time when the first steady state
        volume was retained.

    event_trial_indices: :obj:`list[int]`
        The indices of when each trial event of interest (specified by ``trial_types``)
        begins.

    Example
    -------
    >>> import pandas as pd
    >>> from nifti2bids.bids import PresentationEventExtractor
    >>> extractor = PresentationEventExtractor(
    ...     log_file,
    ...     trial_types=("congruentleft", "congruentright", "incongruentleft", "incongruentright", "nogo"),
    ...     scanner_event_type="Pulse",
    ...     scanner_trigger_code="99",
    ...     convert_to_seconds=["Time"],
    ... )
    >>> events = {}
    >>> events["onset"] = extractor.extract_onsets()
    >>> events["duration"] = extractor.extract_durations()
    >>> events["trial_type"] = extractor.extract_trial_types()
    >>> events["reaction_time"] = extractor.extract_reaction_times()
    >>> events["response"] = extractor.extract_responses()
    >>> events["accuracy"] = extractor.extract_responses(response_map={"hit": 1, "miss": 0})
    >>> df = pd.DataFrame(events)
    """

    def __init__(
        self,
        log_or_df,
        trial_types,
        scanner_event_type,
        scanner_trigger_code,
        trial_column_name="Code",
        convert_to_seconds=None,
        initial_column_headers=("Trial", "Event Type"),
        n_discarded_volumes=0,
        tr=None,
    ):
        super().__init__(
            log_or_df,
            trial_types,
            scanner_event_type,
            scanner_trigger_code,
            trial_column_name,
            convert_to_seconds,
            initial_column_headers,
            n_discarded_volumes,
            tr,
        )

        self.trial_types = trial_types
        trial_series = self.df.loc[
            self.df[self.trial_column_name].isin(trial_types), self.trial_column_name
        ]
        self.event_trial_indices = trial_series.index.tolist()

    def extract_onsets(
        self, scanner_start_time: Optional[float | int] = None
    ) -> list[float]:
        """
        Extract the onset times for each event.

        Onset is calculated as the difference between the event time and
        the scanner start time (e.g. first pulse).

        Parameters
        ----------
        scanner_start_time: :obj:`float`, :obj:`int`, or :obj:`None`, default=None
            The start time for the scanner.

            .. important::
               Scanner start time will be detected during class initialization, unless
               the ``self.scanner_event_code`` and ``self.scanner_trigger_code`` does
               not return an index.

        Returns
        -------
        list[float]
            A list of onset times for each event.
        """
        return self._extract_onsets(self.event_trial_indices, scanner_start_time)

    def _extract_rt_and_responses(self) -> tuple[list[float], list[str]]:
        """
        Extracts and reaction time and responses for each event.

        Reaction time is computed as the difference between the event stimulus
        and the response. When no response is given, the reaction is the
        difference between the starting time of that trial and the starting
        time of the subsequent stimuli.

        Returns
        -------
        tuple[list[float], list[str]]
            A tuple containing a list of durations and a list of responses.

        Note
        ----
        When no response is given the response will be assigned "nan".
        """
        reaction_times, responses = [], []
        for row_indx in self.event_trial_indices:
            row = self.df.loc[row_indx, :]
            trial_num = row["Trial"]
            response_row = self.df[
                (self.df["Trial"] == trial_num) & (self.df["Event Type"] == "Response")
            ]
            if not response_row.empty:
                reaction_time = float(response_row.iloc[0]["Time"] - row["Time"])
                response = row["Stim Type"]
            else:
                reaction_time = float("nan")
                response = float("nan")

            reaction_times.append(reaction_time)
            responses.append(response)

        return reaction_times, responses

    def extract_durations(self) -> list[float]:
        """
        Extract the duration for each event. Will extract the duration from the
        "Duration" column.

        Returns
        -------
        list[float]
            A list of durations for each event.
        """
        return self.df.loc[self.event_trial_indices, "Duration"].tolist()

    def extract_trial_types(self) -> list[str]:
        """
        Extract the trial type for each event.

        Returns
        -------
        list[str]
            A list of trial types for each event.
        """
        return self._extract_trial_types(self.event_trial_indices)

    def extract_reaction_times(self) -> list[float]:
        """
        Extract the reaction time for each event.

        .. note::
           - The reaction time is computed for the first response.
           - Will provide reaction time for all trials subject responds too.

        Reaction time is computed as the difference between the trial onset and the time
        the response if recorded. If no reponse is recorded, then the reaction time is NaN
        """
        reaction_times, _ = self._extract_rt_and_responses()

        return reaction_times

    def extract_responses(self) -> list[str]:
        """
        Extract the response for each event.

        .. important::
           NaN means that no response was recorded for the trial
           (i.e. "miss").

        Returns
        -------
        list[str]
            A list of responses for each event.

        Note
        ----
        When no response is given the response will be assigned "nan".
        """
        _, responses = self._extract_rt_and_responses()

        return responses

    def extract_accuracies(
        self,
        response_map: dict[str, int],
    ) -> list[int]:
        """
        Extract the accuracy (correct or incorrect) for each event.

        Parameters
        ----------
        response_map: :obj:`dict[str, int]`
            A dictionary mapping response codes, from "Stim Type" column
            (ie. "hit", "miss", "other", "false alarm", "incorrect"), to accuracy
            values (0 for incorrect, 1 for correct).

        Returns
        -------
        list[int]
            A list of accuracy values for each event (0 = incorrect, 1 = correct).
        """
        _, responses = self._extract_rt_and_responses()

        return [response_map.get(response) for response in responses]


class EPrimeExtractor:
    """
    Base class for E-Prime 3 log extractors.

    Provides shared initialization and extraction logic for both block
    and event design extractors.
    """

    def __init__(
        self,
        log_or_df: str | Path | pd.DataFrame,
        condition_codes: tuple[str],
        onset_column_name: str,
        procedure_column_name: str,
        trigger_column_name: Optional[None] = None,
        convert_to_seconds: Optional[list[str]] = None,
        initial_column_headers: tuple[str] = ("ExperimentName", "Subject"),
        n_discarded_volumes: int = 0,
        tr: Optional[float | int] = None,
    ):

        self.df = _process_log_or_df(
            log_or_df,
            convert_to_seconds,
            initial_column_headers,
            divisor=1e3,
            software="E-Prime",
        )
        self.onset_column_name = onset_column_name
        self.procedure_column_name = procedure_column_name
        self.trigger_column_name = trigger_column_name

        if self.trigger_column_name:
            self.scanner_start_time = (
                self.df[self.trigger_column_name].dropna(inplace=False).unique()[0]
            )
        else:
            self.scanner_start_time = None

        self.n_discarded_volumes = n_discarded_volumes
        self.tr = tr
        if self.n_discarded_volumes > 0:
            if not self.tr:
                raise ValueError(
                    "``tr`` must be provided when ``n_discarded_volumes`` is greater than 0."
                )

            if not self.scanner_start_time:
                raise ValueError(
                    "``scanner_start_time`` is None so time shift cannot be added."
                )

            self.scanner_start_time += self.n_discarded_volumes * self.tr

    def _extract_onsets(
        self, row_indices: list[str], scanner_start_time: Optional[float | int]
    ) -> list[float]:
        """Extract onset times for each block or event."""
        if scanner_start_time is not None:
            self.scanner_start_time = scanner_start_time

        if self.scanner_start_time is None:
            raise ValueError(
                "``scanner_start_time`` must be provided either due to ``trigger_column_name`` not being provided "
                "or due to a non-NaN value could not be extracted from ``trigger_column_name``."
            )

        return [
            self.df.loc[index, self.onset_column_name] - self.scanner_start_time
            for index in row_indices
        ]

    def _extract_trial_types(self, row_indices: list[int]) -> list[str]:
        """Extract trial types for each block or event."""
        return [self.df.loc[index, self.procedure_column_name] for index in row_indices]


class EPrimeBlockExtractor(EPrimeExtractor, BlockExtractor):
    """
    Extract onsets, durations, and trial types from E-Prime 3 logs using a block design.

    Parameters
    ----------
    log_or_df: :obj:`str`, :obj:`Path`, :obj:`pandas.DataFrame`
        The Eprime log as a file path or the Eprime DataFrame
        returned by :code:`nifti2bids.parsers.load_eprime_log`.

        .. important::
           If a text file is used, data are assumed to have at least one element
           that is an digit or float during parsing.

    block_cue_codes: :obj:`tuple[str]`
        The names of the block cue codes (i.e. "Face", "Place").

    onset_column_name: :obj:`str`
        The name of the column containing stimulus onset time.

    procedure_column_name: :obj:`str`
        The name of the column containing the procedure names.

    trigger_column_name: :obj:`str` or :obj:`None`, default=None
        The name of the column containing the scanner start time.
        Uses the first value that is not NaN as the scanner start
        time. If None, the scanner start time will need to be
        given when using ``self.extract_onsets``.

    convert_to_seconds: :obj:`list[str]` or :obj:`None`, default=None
        Convert the time resolution of the specified columns from milliseconds to seconds.

        .. important::
           Recommend time resolution of the columns containing the onset time and scanner
           start time (``trigger_column_name``) be converted to seconds.

    initial_column_headers: :obj:`tuple[str]`, default=("ExperimentName", "Subject")
        The initial column headers for data. Only used when
        ``log_or_df`` is a file path.

    n_discarded_volumes: :obj:`int`, default=0
        Number of non-steady state scans discarded by the scanner at the start of the sequence.

        .. important::
           - Only used when ``trigger_column_name`` is specified.
           - Only set this parameter if scanner trigger is sent **before** these volumes are
             acquired so that the start time of the first retained volume is shifted forward
             by (``n_discarded_volumes * tr``). If the scanner sends trigger **after**
             discarding the volumes, do not set this parameter.
             `Explanation from Neurostars <https://neurostars.org/t/how-to-sync-time-from-e-prime-behavior-data-with-fmri/6887>`_.

    tr: :obj:`float`, :obj:`int`, or :obj:`None`, default=None
        The repetition time provided in seconds if data was converted to seconds or
        in milliseconds if not converted.

    rest_block_code: :obj:`str` or :obj:`None`, default=None
        The name of the code for the rest block. Used when a resting state
        block is between the events to compute the correct block duration.
        If None, the block duration will be computed based on the starting
        index of the block cue codes given by ``block_cue_codes``. If specified
        and ``rest_code_frequency`` is "variable", will be used with
        ``block_cue_codes`` to compute the correct duration.

    rest_code_frequency: :obj:`Literal["fixed", "variable"]`, default="fixed"
        Frequency of the rest block. For "fixed", the rest code is assumed to
        appear between each block or at least each block. For "variable",
        it is assumed that the rest code does not appear between each
        block.

    quit_code: :obj:`str` or :obj:`None`, default=None
        The quit code. Suggest to use in cases when a quit code, as opposed
        to a rest code, is preceded by a trial block. Ideally, this should
        be a unique code.

    Attributes
    ----------
    df: :obj:`pandas.DataFrame`
        DataFrame containing the log data.

    block_cue_codes: :obj:`tuple[str]`
        The names of the block cue codes.

    onset_column_name: :obj:`str`
        Name of column containing the onset time.

    procedure_column_name: :obj:`str`
        Name of column containing the procedure names.

    trigger_column_name: :obj:`str` or :obj:`None`
        Name of column containing time when scanner sent pulse/scanner start time.

    n_discarded_scans: :obj:`int`
        Number of non-steady state scans discarded by scanner.

    tr: :obj:`float`, :obj:`int`, or :obj:`None`
        The repetition time.

    scanner_start_time: :obj:`float` or :obj:`None`
        Time when scanner sends the pulse. If ``n_discarded_volumes``
        is not 0 and ``tr`` is specified, then this time will be
        shifted forward (``scanner_start_time = scanner_start_time + n_discarded_volumes * tr``)
        to reflect the time when the first steady state volume was retained. Otherwise, the time
        extracted from the log data is assumed to be the time when the first steady state
        volume was retained.

    starting_block_indices: :obj:`list[int]`
        The indices of when each trial block of interest (specified by ``block_cue_codes``)
        begins.

    rest_block_code: :obj:`str` or :obj:`None`
        The name of the code for the rest block.

    rest_code_frequency: :obj:`Literal["fixed", "variable"]`
        Frequency of the rest block.

    quit_code: :obj:`str` or :obj:`None`
        The quit code.

    Example
    -------
    >>> import pandas as pd
    >>> from nifti2bids.bids import EPrimeBlockExtractor
    >>> extractor = EPrimeBlockExtractor(
    ...     log_file,
    ...     block_cue_codes=("Face", "Place"),
    ...     onset_column_name="Stimulus.OnsetTime",
    ...     procedure_column_name="Procedure",
    ...     trigger_column_name="ScannerTrigger.RTTime",
    ...     convert_to_seconds=["Stimulus.OnsetTime", "ScannerTrigger.RTTime", "Stimulus.RT"],
    ...     rest_block_code="Rest",
    ... )
    >>> events = {}
    >>> events["onset"] = extractor.extract_onsets()
    >>> events["duration"] = extractor.extract_durations()
    >>> events["trial_type"] = extractor.extract_trial_types()
    >>> # Mean reaction time for correct Face trials only
    >>> events["mean_rt"] = extractor.extract_mean_reaction_times(
    ...     reaction_time_column_name="Stimulus.RT",
    ...     subject_response_column="Stimulus.RESP",
    ...     correct_response_column="Simulus.CRESP",
    ...     response_type="correct",
    ...     response_trial_codes=("Face")
    ... )
    >>> # Mean accuracy across all trial types
    >>> events["mean_accuracy"] = extractor.extract_mean_accuracies(
    ...     subject_response_column="Stimulus.RESP",
    ...     correct_response_column="Stimulus.CRESP",
    ... )
    >>> df = pd.DataFrame(events)
    """

    def __init__(
        self,
        log_or_df,
        block_cue_codes,
        onset_column_name,
        procedure_column_name,
        trigger_column_name=None,
        convert_to_seconds=None,
        initial_column_headers=("ExperimentName", "Subject"),
        n_discarded_volumes=0,
        tr=None,
        rest_block_code=None,
        rest_code_frequency="fixed",
        quit_code=None,
    ):
        super().__init__(
            log_or_df,
            block_cue_codes,
            onset_column_name,
            procedure_column_name,
            trigger_column_name,
            convert_to_seconds,
            initial_column_headers,
            n_discarded_volumes,
            tr,
        )

        self.block_cue_codes = block_cue_codes

        assert rest_code_frequency in [
            "fixed",
            "variable",
        ], "`rest_code_frequency` must be either 'fixed' or 'variable'."

        self.rest_block_code = rest_block_code
        self.rest_code_frequency = rest_code_frequency
        self.quit_code = quit_code

        self.starting_block_indices = _get_starting_block_indices(
            self.df, self.procedure_column_name, self.block_cue_codes
        )

    def extract_onsets(
        self, scanner_start_time: Optional[float | int] = None
    ) -> list[float]:
        """
        Extract the onset times for each block.

        Onset is calculated as the difference between the event time and
        the scanner start time.

        Parameters
        ----------
        scanner_start_time: :obj:`float`, :obj:`int`, or :obj:`None`, default=None
            The scanner start time. Used to compute onset relative to
            the start of the scan.

            .. note:: Does not need to be given if ``trigger_column_name`` was provided.

        Returns
        -------
        list[float]
            A list of onset times for each block.
        """
        return self._extract_onsets(self.starting_block_indices, scanner_start_time)

    def extract_durations(self) -> list[float]:
        """
        Extract the duration for each block.

        Duration is computed as the difference between the start of the block
        and the start of the next block (either a rest block or some task block).

        Returns
        -------
        list[float]
            A list of durations for each block.
        """
        durations = []
        for block_start_indx in self.starting_block_indices:
            block_start_row = self.df.loc[block_start_indx, :]
            block_end_indx = _get_next_block_index(
                trial_series=self.df[self.procedure_column_name],
                block_start_indx=block_start_indx,
                rest_block_code=self.rest_block_code,
                rest_code_frequency=self.rest_code_frequency,
                block_cue_codes=self.block_cue_codes,
                quit_code=self.quit_code,
            )
            block_end_row = self.df.loc[block_end_indx, :]
            duration = (
                block_end_row[self.onset_column_name]
                - block_start_row[self.onset_column_name]
            )

            durations.append(duration)

        return durations

    def extract_trial_types(self) -> list[str]:
        """
        Extract the trial type for each block.

        Returns
        -------
        list[str]
            A list of trial types for each block.
        """
        return self._extract_trial_types(self.starting_block_indices)

    def _get_block_trials(
        self,
        block_start_indx: int,
        response_trial_codes: Optional[tuple[str]] = None,
    ) -> pd.DataFrame:
        """
        Get trials within a block. Filtered by ``response_trial_codes`` if not None.

        Parameters
        ----------
        block_start_indx: :obj:`int`
            The starting index of the block.

        response_trial_codes: :obj:`tuple[str]` or :obj:`None`, default=None
            The codes identifying trials to include. If None, includes all
            trials after the block cue row. For reaction, this should typically
            be set to only include trials where a response is expected
            (e.g., "Go" but not "NoGo").

            .. important::
               In cases such as switch tasks, where ``block_cue_codes`` are
               non-responsive blocks indicating a switch, then this parameter
               should include all trial names that are not in ``block_cue_codes``
               so that reaction time is computed for non-cue codes.
        Returns
        -------
        pandas.DataFrame
            DataFrame containing the filtered block trials.
        """
        block_end_indx = _get_next_block_index(
            trial_series=self.df[self.procedure_column_name],
            block_start_indx=block_start_indx,
            rest_block_code=self.rest_block_code,
            rest_code_frequency=self.rest_code_frequency,
            block_cue_codes=self.block_cue_codes,
        )

        # Note: iloc excludes end index
        if response_trial_codes is not None:
            block_df = self.df.iloc[block_start_indx:block_end_indx, :]
            block_df = block_df[
                block_df[self.procedure_column_name].isin(response_trial_codes)
            ]
        else:
            # Skip first row (cue) since responses are in subsequent rows
            block_df = self.df.iloc[(block_start_indx + 1) : block_end_indx, :]

        return block_df

    def _compute_trial_correctness(
        self,
        block_df: pd.DataFrame,
        subject_response_column: str,
        correct_response_column: str,
        response_required_only: bool,
    ) -> pd.Series:
        """
        Compute correctness for each trial in a block.

        A trial is correct if:
        - Subject response matches correct response, OR
        - Both are NaN (correct withhold on no-go trial)

        Parameters
        ----------
        block_df: :obj:`pd.DataFrame`
            DataFrame containing block trials.

        subject_response_column: :obj:`str`
            Column name for subject's response.

        correct_response_column: :obj:`str`
            Column name for correct response.

        response_required_only: :obj:`bool`
            Compute accuracy only for trials expecting a response.
            Non-response trials are assumed to be assigned NaN
            in ``correct_response_column``.

        Returns
        -------
        tuple[pandas.Series, pandas.DataFrame]
            Boolean series indicating correctness for each trial and the original or
            filtered ``block_df``.
        """

        subject_resp = block_df[subject_response_column].apply(
            lambda x: float(x) if _is_float(x) else x
        )
        correct_resp = block_df[correct_response_column].apply(
            lambda x: float(x) if _is_float(x) else x
        )

        if response_required_only:
            block_df = block_df[~correct_resp.isna()]
            subject_resp = subject_resp[~correct_resp.isna()]
            correct_resp = correct_resp[~correct_resp.isna()]

        both_nan = subject_resp.isna() & correct_resp.isna()
        both_equal = subject_resp == correct_resp

        return both_nan | both_equal, block_df

    def extract_mean_reaction_times(
        self,
        reaction_time_column_name: str,
        subject_response_column: str,
        correct_response_column: str,
        response_type: Literal["correct", "incorrect"] = "correct",
        response_trial_codes: Optional[tuple[str]] = None,
        response_required_only: bool = False,
    ) -> list[float]:
        """
        Extract mean reaction times for each block.

        Computes mean reaction time filtered for correct or incorrect trials.

        Parameters
        ----------
        reaction_time_column_name: :obj:`str`
            The name of the column containing reaction time values.
            Usually the column name ending in ".RT" not the column
            ending in ".RTTime".

        subject_response_column: :obj:`str`
            The name of the column containing the subject's response.
            Usually the column name ending in ".RESP".

        correct_response_column: :obj:`str`
            The name of the column containing the correct response.
            For no-go trials, this should be empty/NaN.

        response_type: :obj:`Literal["correct", "incorrect"]`, default="correct"
            Whether to compute mean reaction time for correct or incorrect trials.

        response_trial_codes: :obj:`tuple[str]` or :obj:`None`, default=None
            The codes identifying trials to include. If None, includes all
            trials after the block cue row. For reaction, this should typically
            be set to only include trials where a response is expected
            (e.g., "Go" but not "NoGo").

            .. important::
               In cases such as switch tasks, where ``block_cue_codes`` are
               non-responsive blocks indicating a switch, then this parameter
               should include all trial names that are not in ``block_cue_codes``
               so that reaction time is computed for non-cue codes.

        response_required_only: :obj:`bool`, default=False
            Compute reaction times only for trials expecting a response.
            Non-response trials are assumed to be assigned NaN
            in ``correct_response_column``.

        Returns
        -------
        list[float]
            A list of mean reaction times for each block. Returns NaN for
            blocks with no valid reaction times.

        Notes
        -----
        Correctness is determined by comparing ``subject_response_column`` to
        ``correct_response_column``:

        - A trial is **correct** if the responses match, OR if both are NaN
          (e.g. correct withhold on a no-go trial).
        - A trial is **incorrect** if they differ (wrong response, miss, or
          false alarm).

        Trials with NaN reaction times (due to filtering) are excluded via ``np.nanmean``.

        Example
        -------
        >>> # Get mean reaction time for correct Go trials only
        >>> mean_rts = extractor.extract_mean_reaction_times(
        ...     reaction_time_column_name="Stimulus.RT",
        ...     subject_response_column="Stimulus.RESP",
        ...     correct_response_column="CorrectResponse",
        ...     response_type="correct",
        ...     response_trial_codes=("Go")
        ... )
        """
        target_correctness = response_type == "correct"
        mean_rts = []

        for block_start_indx in self.starting_block_indices:
            block_df = self._get_block_trials(block_start_indx, response_trial_codes)

            correctness, block_df = self._compute_trial_correctness(
                block_df,
                subject_response_column,
                correct_response_column,
                response_required_only,
            )

            if not correctness.empty:
                filtered_rts = block_df.loc[
                    correctness == target_correctness, reaction_time_column_name
                ]
                mean_rts.append(
                    np.nanmean(filtered_rts)
                    if filtered_rts.size > 0 and not np.all(np.isnan(filtered_rts))
                    else np.nan
                )
            else:
                mean_rts.append(np.nan)

        return mean_rts

    def extract_mean_accuracies(
        self,
        subject_response_column: str,
        correct_response_column: str,
        response_trial_codes: Optional[tuple[str]] = None,
        response_required_only: bool = False,
    ) -> list[float]:
        """
        Extract mean accuracy for each block.

        Computes accuracy by comparing the subject's response to the correct
        response for each trial.

        Parameters
        ----------
        subject_response_column: :obj:`str`
            The name of the column containing the subject's response.
            Usually the column name ending in ".RESP".

        correct_response_column: :obj:`str`
            The name of the column containing the correct response. Trials
            where the subject should not respond should be NaN.
            Usually column name ending in ".CRESP".

        response_trial_codes: :obj:`tuple[str]` or :obj:`None`, default=None
            The codes identifying trials to include. If None, includes all
            trials after the block cue row.

            .. important::
               In cases such as switch tasks, where ``block_cue_codes`` are
               non-responsive blocks indicating a switch, then this parameter
               should include all trial names that are not in ``block_cue_codes``
               so that accuracy is computed only for non-cue codes.

        response_required_only: :obj:`bool`, default=False
            Compute accuracy only for trials expecting a response.
            Non-response trials are assumed to be assigned NaN
            in ``correct_response_column``.

        Returns
        -------
        list[float]
            A list of mean accuracies for each block.

        Notes
        -----
        Correctness is determined by comparing ``subject_response_column`` to
        ``correct_response_column`` using the following logic:

        +------------------+-------------------+--------------------------+----------+
        | Subject Response | Correct Response  | Interpretation           | Accuracy |
        +==================+===================+==========================+==========+
        | "1"              | "1"               | Correct response         | 1        |
        +------------------+-------------------+--------------------------+----------+
        | NaN              | NaN               | Correct response         | 1        |
        +------------------+-------------------+--------------------------+----------+
        | NaN              | "1"               | Incorrect response       | 0        |
        +------------------+-------------------+--------------------------+----------+
        | "1"              | NaN               | Incorrect response       | 0        |
        +------------------+-------------------+--------------------------+----------+
        | "2"              | "3"               | Incorrect response       | 0        |
        +------------------+-------------------+--------------------------+----------+

        Note: If ``response_required_only`` is True, only accuracy for the first,
        third, and fifth row are computed since those are trials that expect a response.

        Example
        -------
        >>> # Get mean accuracy for all trial types
        >>> mean_accs = extractor.extract_mean_accuracies(
        ...     subject_response_column="Stimulus.RESP",
        ...     correct_response_column="CorrectResponse"
        ... )
        >>> # Get mean accuracy for specific trial types only
        >>> mean_accs = extractor.extract_mean_accuracies(
        ...     subject_response_column="Stimulus.RESP",
        ...     correct_response_column="CorrectResponse",
        ...     response_trial_codes=("Go", "NoGo")
        ... )
        """
        mean_accs = []
        for block_start_indx in self.starting_block_indices:
            block_df = self._get_block_trials(block_start_indx, response_trial_codes)

            correctness, _ = self._compute_trial_correctness(
                block_df,
                subject_response_column,
                correct_response_column,
                response_required_only,
            )

            mean_accs.append(correctness.mean())

        return mean_accs


class EPrimeEventExtractor(EPrimeExtractor, EventExtractor):
    """
    Extract onsets, durations, trial types, reaction times, and responses
    from E-Prime 3 logs using an event design.

    Parameters
    ----------
    log_or_df: :obj:`str`, :obj:`Path`, :obj:`pandas.DataFrame`
        The Eprime log as a file path or the Eprime DataFrame
        returned by :code:`nifti2bids.parsers.load_eprime_log`.

        .. important::
           If a text file is used, data are assumed to have at least one element
           that is an digit or float during parsing.

    trial_types: :obj:`tuple[str]`
        The names of the trial types (i.e "congruentleft", "seen").

        .. note::
            Depending on the way your Eprime data is structured, for block
            design the rest block may have to be included as a "trial_type"
            to compute the correct duration. These rows can then be dropped
            from the events DataFrame.

    onset_column_name: :obj:`str`
        The name of the column containing stimulus onset time.

    procedure_column_name: :obj:`str`
        The name of the column containing the procedure names.

    trigger_column_name: :obj:`str` or :obj:`None`, default=None
        The name of the column containing the scanner start time.
        Uses the first value that is not NaN as the scanner start
        time. If None, the scanner start time will need to be
        given when using ``self.extract_onsets``.

    convert_to_seconds: :obj:`list[str]` or :obj:`None`, default=None
        Convert the time resolution of the specified columns from milliseconds to seconds.

        .. important::
           Recommend time resolution of the columns containing the onset times,
           offset times (duration), reaction times, and scanner onset time (``trigger_column_name``)
           be converted to seconds.

    initial_column_headers: :obj:`tuple[str]`, default=("ExperimentName", "Subject")
        The initial column headers for data. Only used when
        ``log_or_df`` is a file path.

    n_discarded_volumes: :obj:`int`, default=0
        Number of non-steady state scans discarded by the scanner at the start of the sequence.

        .. important::
           - Only used when ``trigger_column_name`` is specified.
           - Only set this parameter if scanner trigger is sent **before** these volumes are
             acquired so that the start time of the first retained volume is shifted forward
             by (``n_discarded_volumes * tr``). If the scanner sends trigger **after**
             discarding the volumes, do not set this parameter.
             `Explanation from Neurostars <https://neurostars.org/t/how-to-sync-time-from-e-prime-behavior-data-with-fmri/6887>`_.

    tr: :obj:`float`, :obj:`int`, or :obj:`None`, default=None
        The repetition time provided in seconds if data was converted to seconds or
        in milliseconds if not converted.

    Attributes
    ----------
    df: :obj:`pandas.DataFrame`
        DataFrame containing the log data.

    trial_types: :obj:`tuple[str]`
        The names of the trial types.

    onset_column_name: :obj:`str`
        Name of column containing the onset time.

    procedure_column_name: :obj:`str`
        Name of column containing the trial types.

    trigger_column_name: :obj:`str` or :obj:`None`
        Name of column containing time when scanner sent pulse/scanner start time.

    n_discarded_scans: :obj:`int`
        Number of non-steady state scans discarded by scanner.

    tr: :obj:`float`, :obj:`int`, or :obj:`None`
        The repetition time.

    scanner_start_time: :obj:`float` or :obj:`None`
        Time when scanner sends the pulse. If ``n_discarded_volumes``
        is not 0 and ``tr`` is specified, then this time will be
        shifted forward (``scanner_start_time = scanner_start_time + n_discarded_volumes * tr``)
        to reflect the time when the first steady state volume was retained. Otherwise, the time
        extracted from the log data is assumed to be the time when the first steady state
        volume was retained.

    event_trial_indices: :obj:`list[int]`
        The indices of when each trial event of interest (specified by ``trial_types``)
        begins.

    Example
    -------
    >>> import pandas as pd
    >>> from nifti2bids.bids import EPrimeEventExtractor
    >>> extractor = EPrimeEventExtractor(
    ...     log_file,
    ...     trial_types=("Go", "NoGo"),
    ...     onset_column_name="Stimulus.OnsetTime",
    ...     procedure_column_name="Procedure",
    ...     trigger_column_name="ScannerTrigger.RTTime",
    ...     convert_to_seconds=["Stimulus.OnsetTime", "Stimulus.OffsetTime", "Stimulus.RT", "ScannerTrigger.RTTime"],
    ... )
    >>> events = {}
    >>> events["onset"] = extractor.extract_onsets()
    >>> events["duration"] = extractor.extract_durations(offset_column_name="Stimulus.OffsetTime")
    >>> events["trial_type"] = extractor.extract_trial_types()
    >>> events["reaction_time"] = extractor.extract_reaction_times(reaction_time_column_name="Stimulus.RT")
    >>> events["accuracy"] = extractor.extract_accuracies(
    ...     subject_response_column="Stimulus.RESP",
    ...     correct_response_column="CorrectResponse",
    ... )
    >>> df = pd.DataFrame(events)
    """

    def __init__(
        self,
        log_or_df,
        trial_types,
        onset_column_name,
        procedure_column_name,
        trigger_column_name=None,
        convert_to_seconds=None,
        initial_column_headers=("ExperimentName", "Subject"),
        n_discarded_volumes=0,
        tr=None,
    ):
        super().__init__(
            log_or_df,
            trial_types,
            onset_column_name,
            procedure_column_name,
            trigger_column_name,
            convert_to_seconds,
            initial_column_headers,
            n_discarded_volumes,
            tr,
        )

        self.trial_types = trial_types
        trial_series = self.df.loc[
            self.df[self.procedure_column_name].isin(trial_types),
            self.procedure_column_name,
        ]
        self.event_trial_indices = trial_series.index.tolist()

    def extract_onsets(
        self,
        scanner_start_time: Optional[float | int] = None,
    ) -> list[float]:
        """
        Extract the onset times for each event.

        Onset is calculated as the difference between the event time and
        the scanner start time.

        Parameters
        ----------
        scanner_start_time: :obj:`float`, :obj:`int`, or :obj:`None`, default=None
            The scanner start time. Used to compute onset relative to
            the start of the scan.

            .. note:: Does not need to be given if ``trigger_column_name`` was set.

        Returns
        -------
        list[float]
            A list of onset times for each event.
        """
        return self._extract_onsets(self.event_trial_indices, scanner_start_time)

    def extract_durations(self, offset_column_name: str) -> list[float]:
        """
        Extract the duration for each event.

        Parameters
        ----------
        offset_column_name: :obj:`str`
            The name of the column containing the offset time of trial.
            Duration is computed as the difference between the trial onset
            time and the trial offset time.

        Returns
        -------
        list[float]
            A list of durations for each event.
        """
        return [
            self.df.loc[index, offset_column_name]
            - self.df.loc[index, self.onset_column_name]
            for index in self.event_trial_indices
        ]

    def extract_trial_types(self) -> list[str]:
        """
        Extract the trial type for each event.

        Returns
        -------
        list[str]
            A list of trial types for each event.
        """
        return self._extract_trial_types(self.event_trial_indices)

    def extract_reaction_times(self, reaction_time_column_name: str) -> list[float]:
        """
        Extract the reaction time for each event.

        Parameters
        ----------
        reaction_time_column_name: :obj:`str`
            The name of the column containing reaction time values.
            Usually the column name ending in ".RT".

        Returns
        -------
        list[float]
            A list of reaction times for each event. NaN for trials where
            the subject did not respond.

        Notes
        -----
        This function returns all reaction times as-is, regardless of whether
        the trial was correct or incorrect, or whether the subject was expected
        to respond or not. Filter the resulting DataFrame based on trial type and
        response if needed.

        Example
        -------
        >>> rts = extractor.extract_reaction_times("Stimulus.RT")
        >>> responses = extractor.extract_accuracies(
        ...     subject_response_column="Stimulus.RESP",
        ...     correct_response_column="Stimulus.CRESP",
        ... )
        >>> # Filter to only correct Go trials:
        >>> df = pd.DataFrame({"rt": rts, "trial_type": trial_types, "response": responses})
        >>> correct_go_rts = df[(df["trial_type"] == "Go") & (df["response"] == 1)]["rt"]
        """
        return [
            self.df.loc[index, reaction_time_column_name]
            for index in self.event_trial_indices
        ]

    def extract_accuracies(
        self,
        subject_response_column: str,
        correct_response_column: str,
        response_required_only: bool = False,
    ) -> list[int]:
        """
        Extract the accuracy (correct or incorrect) for each event.

        Parameters
        ----------
        subject_response_column: :obj:`str`
            The name of the column containing the subject's response.
            Usually the column name ending in ".RESP".

        correct_response_column: :obj:`str`
            The name of the column containing the correct response. Trials
            where the subject should not respond should be NaN.
            Usually column name ending in ".CRESP".

        response_required_only: :obj:`bool`
            Compute accuracy only for trials expecting a response.
            Non-response trials are assumed to be assigned NaN
            in ``correct_response_column``.

        Returns
        -------
        list[int]
            A list of accuracy values for each event (0 = incorrect, 1 = correct,
            NaN = trial did not require response when ``response_required_only=True``).

        Notes
        -----
        Correctness is determined by comparing ``subject_response_column`` to
        ``correct_response_column``:

        +------------------+-------------------+--------------------------+----------+
        | Subject Response | Correct Response  | Interpretation           | Accuracy |
        +==================+===================+==========================+==========+
        | "1"              | "1"               | Correct response         | 1        |
        +------------------+-------------------+--------------------------+----------+
        | NaN              | NaN               | Correct response         | 1        |
        +------------------+-------------------+--------------------------+----------+
        | NaN              | "1"               | Incorrect response       | 0        |
        +------------------+-------------------+--------------------------+----------+
        | "1"              | NaN               | Incorrect response       | 0        |
        +------------------+-------------------+--------------------------+----------+
        | "2"              | "3"               | Incorrect response       | 0        |
        +------------------+-------------------+--------------------------+----------+

        Note: If ``response_required_only`` is True, only accuracy for the first,
        third, and fifth row are assigned 0 and 1, the second and fourth trials as NaN
        since those are trials that expect a response.

        Example
        -------
        >>> responses = extractor.extract_accuracies(
        ...     subject_response_column="Stimulus.RESP",
        ...     correct_response_column="Stimulus.CRESP",
        ... )
        """
        responses = []
        for row_indx in self.event_trial_indices:
            subject_resp = self.df.loc[row_indx, subject_response_column]
            correct_resp = self.df.loc[row_indx, correct_response_column]

            subject_resp = (
                float(subject_resp) if _is_float(subject_resp) else subject_resp
            )
            correct_resp = (
                float(correct_resp) if _is_float(correct_resp) else correct_resp
            )

            both_nan = pd.isna(subject_resp) and pd.isna(correct_resp)
            both_equal = subject_resp == correct_resp

            if response_required_only and pd.isna(correct_resp):
                response = np.nan
            else:
                response = 1 if (both_nan or both_equal) else 0

            responses.append(response)

        return responses
