"""Module for creating BIDS compliant files."""

import json
import itertools
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal, Optional

import pandas as pd

from nifti2bids.logging import setup_logger
from nifti2bids.io import _copy_file, glob_contents
from nifti2bids.parsers import load_eprime_log, load_presentation_log, _convert_time

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

    return df


def _get_starting_block_indices(
    log_df: pd.DataFrame, trial_column_name: str, trial_types: tuple[str]
) -> list[int]:
    """
    Get starting indices for blocks.

    Parameters
    ----------
    log_df: :obj:`pandas.DataFrame`
        DataFrame of neurobehavioral log data.

    trial_column_name: :obj:`str`
        Name of the column containing the trial information.

    trial_types: :obj:`tuple[str]`
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
        if trial_type not in trial_types:
            trial_indxs = trial_series[trial_series == trial_type].index.tolist()
            starting_block_indices = starting_block_indices.difference(trial_indxs)

    # Remove empty
    missing_indices = trial_series[trial_series.isna()].index.tolist()
    starting_block_indices = starting_block_indices.difference(missing_indices)

    return sorted(list(starting_block_indices))


def _get_next_block_index(
    trial_series: pd.Series,
    curr_row_indx: int,
    rest_block_code: Optional[str],
    rest_code_frequency: Literal["fixed", "variable"],
    trial_types: tuple[str],
    quit_code: Optional[str] = None,
) -> int:
    """
    Get the starting index for each block.

    Parameters
    ----------
    trial_series: :obj:`pandas.Series`
        A Pandas Series of the column containing the trial type information.

    curr_row_indx: :obj:`int`
        The current row index.

    rest_block_code: :obj:`str` or :obj:`None`
        The name of the rest block.

    rest_code_frequency: :obj:`Literal["fixed", "variable"]`, default="fixed"
        Frequency of the rest block. For "fixed", the rest code is assumed to
        appear between each trial or at least each trial. For "variable",
        it is assumed that the rest code does not appear between each
        trial.

    trial_types: :obj:`tuple[str]`
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
    curr_trial = trial_series.at[curr_row_indx]
    filtered_trial_series = trial_series[trial_series.index > curr_row_indx]
    filtered_trial_series = filtered_trial_series.astype(str)

    if rest_block_code and rest_code_frequency == "fixed":
        target_codes = [rest_block_code] + ([quit_code] if quit_code else [])
        next_block_indxs = filtered_trial_series[
            filtered_trial_series.isin(tuple(target_codes))
        ].index.tolist()
    else:
        target_block_names = set(tuple(trial_types))
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

    return next_block_indxs[0] if next_block_indxs else curr_row_indx


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


class EventExtractor(LogExtractor):
    """Abstract Base Class for Event Extractors."""

    @abstractmethod
    def extract_reaction_times(self):
        """Extract reaction time for each trial."""

    @abstractmethod
    def extract_responses(self):
        """Extract response for each trial."""


class PresentationExtractor:
    """
    Base class for Presentation log extractors.

    Provides shared initialization and extraction logic for both block
    and event design extractors.
    """

    def __init__(
        self,
        log_or_df: str | Path | pd.DataFrame,
        trial_types: tuple[str],
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
            divisor=10000,
            software="Presentation",
        )
        self.trial_types = trial_types
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
        if scanner_start_time:
            self.scanner_start_time = scanner_start_time

        if not self.scanner_start_time:
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

    trial_types: :obj:`tuple[str]`
        The names of the trial types (i.e "congruentleft", "seen").

        .. note::
           If your block design does not include a rest block or
           crosshair code, include the code immediately after the
           final block.

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

    starting_block_indices: :obj:`list[int]`
        The indices of when each trial block of interest (specified by ``trial_types``)
        begins.

    Example
    -------
    >>> import pandas as pd
    >>> from nifti2bids.bids import PresentationBlockExtractor
    >>> extractor = PresentationBlockExtractor(
    ...     log_file,
    ...     trial_types=("Face", "Place"),
    ...     scanner_event_type="Pulse",
    ...     scanner_trigger_code="99",
    ...     convert_to_seconds=["Time"],
    ... )
    >>> events = {"onset": None, "duration": None, "trial_type": None}
    >>> events["onset"] = extractor.extract_onsets()
    >>> events["duration"] = extractor.extract_durations(rest_block_code="crosshair")
    >>> events["trial_type"] = extractor.extract_trial_types()
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

        self.starting_block_indices = _get_starting_block_indices(
            self.df, self.trial_column_name, self.trial_types
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

    def extract_durations(
        self,
        rest_block_code: Optional[str] = None,
        rest_code_frequency: Literal["fixed", "variable"] = "fixed",
        quit_code: Optional[str] = None,
    ) -> list[float]:
        """
        Extract the duration for each block.

        Duration is computed as the difference between the start of the block
        and the start of the next block (either a rest block or some task block).

        Parameters
        ----------
        rest_block_code: :obj:`str` or :obj:`None`, default=None
            The name of the code for the rest block. Used when a resting state
            block is between the events to compute the correct block duration.
            If None, the block duration will be computed based on the starting
            index of the trial types given by ``trial_types``. If specified
            and ``rest_code_frequency`` is "variable", will be used with
            ``trial_types`` to compute the correct duration.

        rest_code_frequency: :obj:`Literal["fixed", "variable"]`, default="fixed"
            Frequency of the rest block. For "fixed", the rest code is assumed to
            appear between each trial or at least each trial. For "variable",
            it is assumed that the rest code does not appear between each
            trial.

        quit_code: :obj:`str` or :obj:`None`, default=None,
            The quit code. Suggest to use in cases when a quit code, as opposed
            to a rest code is, is preceeded by a trial block. Ideally, this should
            be a unique code.

        Returns
        -------
        list[float]
            A list of durations for each block.
        """
        assert rest_code_frequency in [
            "fixed",
            "variable",
        ], "`rest_code_frequency` must be either 'fixed' or 'variable'."

        durations = []
        for row_indx in self.starting_block_indices:
            row = self.df.loc[row_indx, :]
            block_end_indx = _get_next_block_index(
                trial_series=self.df[self.trial_column_name],
                curr_row_indx=row_indx,
                rest_block_code=rest_block_code,
                rest_code_frequency=rest_code_frequency,
                trial_types=self.trial_types,
                quit_code=quit_code,
            )
            block_end_row = self.df.loc[block_end_indx, :]
            durations.append((block_end_row["Time"] - row["Time"]))

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
    >>> events = {"onset": None, "duration": None, "trial_type": None, "reaction_time": None, "response": None}
    >>> events["onset"] = extractor.extract_onsets()
    >>> events["duration"] = extractor.extract_durations()
    >>> events["trial_type"] = extractor.extract_trial_types()
    >>> events["reaction_time"] = extractor.extract_reaction_times()
    >>> events["response"] = extractor.extract_responses()
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
        When no response is given the response will be assigned "nan" and the
        reaction time is the difference between the starting time of that
        trial and the starting time of the subsequent stimuli.
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
                response = "nan"

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


class EPrimeExtractor:
    """
    Base class for E-Prime 3 log extractors.

    Provides shared initialization and extraction logic for both block
    and event design extractors.
    """

    def __init__(
        self,
        log_or_df: str | Path | pd.DataFrame,
        trial_types: tuple[str],
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
            divisor=1000,
            software="E-Prime",
        )
        self.trial_types = trial_types
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
        if scanner_start_time:
            self.scanner_start_time = scanner_start_time

        if not self.scanner_start_time:
            raise ValueError(
                "``trigger_column_name`` was not supplied and ``scanner_start_time`` must be given."
                "did not identify a time."
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

    trigger_column_name :obj:`str` or :obj:`str`
        Name of column containing time when scanner sent pulse/scanner start time.

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

    starting_block_indices: :obj:`list[int]`
        The indices of when each trial block of interest (specified by ``trial_types``)
        begins.

    Example
    -------
    >>> import pandas as pd
    >>> from nifti2bids.bids import EPrimeBlockExtractor
    >>> extractor = EPrimeBlockExtractor(
    ...     log_file,
    ...     trial_types=("Face", "Place"),
    ...     onset_column_name="Stimulus.OnsetTime",
    ...     procedure_column_name="Procedure",
    ...     trigger_start_time="EndTime",
    ...     convert_to_seconds=["Stimulus.OnsetTime", "EndTime"],
    ... )
    >>> events = {"onset": None, "duration": None, "trial_type": None}
    >>> events["onset"] = extractor.extract_onsets()
    >>> events["duration"] = extractor.extract_durations(rest_block_code="Rest")
    >>> events["trial_type"] = extractor.extract_trial_types()
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

        self.starting_block_indices = _get_starting_block_indices(
            self.df, self.procedure_column_name, self.trial_types
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

    def extract_durations(
        self,
        rest_block_code: Optional[str] = None,
        rest_code_frequency: Literal["fixed", "variable"] = "fixed",
    ) -> list[float]:
        """
        Extract the duration for each block.

        Duration is computed as the difference between the start of the block
        and the start of the next block (either a rest block or some task block).

        Parameters
        ----------
        rest_block_code: :obj:`str` or :obj:`None`, default=None
            The name of the code for the rest block. Used when a resting state
            block is between the events to compute the correct block duration.
            If None, the block duration will be computed based on the starting
            index of the trial types given by ``trial_types``. If specified
            and ``rest_code_frequency`` is "variable", will be used with
            ``trial_types`` to compute the correct duration.

        rest_code_frequency: :obj:`Literal["fixed", "variable"]`, default="fixed"
            Frequency of the rest block. For "fixed", the rest code is assumed to
            appear between each trial or at least each trial. For "variable",
            it is assumed that the rest code does not appear between each
            trial.

        Returns
        -------
        list[float]
            A list of durations for each block.
        """
        assert rest_code_frequency in [
            "fixed",
            "variable",
        ], "`rest_code_frequency` must be either 'fixed' or 'variable'."

        durations = []
        for row_indx in self.starting_block_indices:
            row = self.df.loc[row_indx, :]
            block_end_indx = _get_next_block_index(
                trial_series=self.df[self.procedure_column_name],
                curr_row_indx=row_indx,
                rest_block_code=rest_block_code,
                rest_code_frequency=rest_code_frequency,
                trial_types=self.trial_types,
            )
            block_end_row = self.df.loc[block_end_indx, :]
            duration = (
                block_end_row[self.onset_column_name] - row[self.onset_column_name]
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

    trigger_column_name :obj:`str` or :obj:`str`
        Name of column containing time when scanner sent pulse/scanner start time.

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
    >>> from nifti2bids.bids import EPrimeEventExtractor
    >>> extractor = EPrimeEventExtractor(
    ...     log_file,
    ...     trial_types=("Congruent", "Incongruent"),
    ...     onset_column_name="Stimulus.OnsetTime",
    ...     procedure_column_name="Procedure",
    ...     trigger_start_time="EndTime",
    ...     convert_to_seconds=["Stimulus.OnsetTime", "Stimulus.OffsetTime", "EndTime"],
    ... )
    >>> events = {"onset": None, "duration": None, "trial_type": None, "reaction_time": None,  "response": None}
    >>> events["onset"] = extractor.extract_onsets()
    >>> events["duration"] = extractor.extract_durations(offset_column_name="Stimulus.OffsetTime")
    >>> events["trial_type"] = extractor.extract_trial_types()
    >>> events["reaction_time"] = extractor.extract_reaction_times(reaction_time_column_name="Stimulus.RT")
    >>> events["response"] = extractor.extract_responses(accuracy_column_name="Stimulus.ACC")
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
        """
        return [
            self.df.loc[index, reaction_time_column_name]
            for index in self.event_trial_indices
        ]

    def extract_responses(self, accuracy_column_name: str) -> list[str]:
        """
        Extract the response for each event.

        Parameters
        ----------
        accuracy_column_name: :obj:`str`
            The name of the column containing accuracy information.
            Assumes accuracy is coded as 0 (incorrect) or 1 (correct).
            Usually the column name ending in ".ACC".

        Returns
        -------
        list[str]
            A list of responses for each event. Values are "correct",
            "incorrect", or "nan" if no response was given.

        Note
        ----
        When no response is given the response will be assigned "nan".
        """
        responses = []
        for row_indx in self.event_trial_indices:
            row = self.df.loc[row_indx, :]
            try:
                response_val = int(row[accuracy_column_name])
            except ValueError:
                response_val = "nan"

            if response_val != "nan":
                response = {"0": "incorrect", "1": "correct"}.get(str(response_val))
            else:
                response = "nan"

            responses.append(response)

        return responses
