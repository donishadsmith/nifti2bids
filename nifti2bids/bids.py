"""Module for creating BIDS compliant files."""

import json
import itertools
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal, Optional

import pandas as pd

from nifti2bids.io import _copy_file, glob_contents
from nifti2bids.parsers import load_eprime_log, load_presentation_log, _convert_time


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


def _get_next_block_indx(
    trial_series: pd.Series,
    curr_row_indx: int,
    rest_block_code: str,
    trial_types: tuple[str],
) -> int:
    """
    Get the starting index for each block.

    Parameters
    ----------
    trial_series: :obj:`pd.Series`
        A pandas Series of the column containing the trail type information.

    curr_row_indx: :obj:`int`
        The current row index.

    rest_block_code: :obj:`str` or :obj:`None`
        The name of the rest block.

    trial_types: :obj:`tuple[str]`
        The names of the trial types. Only used when ``rest_block_code``.
        When used, identifies the indices of all trial types minus
        the indices corresponding to the current trial type.

        .. important::
           ``trial_types=("congruent", "incongruent")`` will identify
           all trial types beginning with "congruent" and "incongruent"

    Returns
    -------
    int
        The starting index of the next block.
    """
    curr_trial = trial_series[curr_row_indx]
    filtered_trial_series = trial_series[curr_row_indx + 1 :]
    filtered_trial_series = filtered_trial_series.astype(str)

    if rest_block_code:
        next_block_indxs = filtered_trial_series[
            filtered_trial_series == rest_block_code
        ].index.tolist()
    else:
        target_block_names = set(tuple(trial_types))
        target_block_names.discard(curr_trial)
        next_block_indxs = filtered_trial_series[
            filtered_trial_series.isin(target_block_names)
        ].index.tolist()

    return next_block_indxs[0] if next_block_indxs else curr_row_indx


# TODO: Do more refactoring to refine code
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
    def extract_responses():
        """Extract responses for each trial."""


class PresentationExtractor:
    """
    Base class for Presentation log extractors.

    Provides shared initialization and extraction logic for both block
    and event design extractors.

    Parameters
    ----------
    log_or_df: :obj:`str`, :obj:`Path`, :obj:`pd.DataFrame`
        The Presentation log as a file path or the Presentation DataFrame
        returned by :code:`nifti2bids.parsers.load_presentation_log`.

        .. important::
           If a text file is used, data are assumed to have at least one element
           that is an digit or float during parsing.

    trial_types: :obj:`tuple[str]`
        The names of the trial types (i.e "congruentleft", "seen").

    use_first_pulse: :obj:`bool`, default=True
        Uses the timing of the first pulse as the start time for the scanner,
        which is used to compute the onset times of the trials relative
        to the scanner start time.

        .. note::
           If set to False, a scanner start time can be supplied to
           ``self.extract_onsets``.

    convert_to_seconds: :obj:`list[str]` or :obj:`None`, default=None
        Convert the time resolution of the specified columns from 0.1ms to seconds.

        .. note:: Recommend time resolution of the "Time" column to be converted.

    initial_column_headers: :obj:`tuple[str]`, default=("Trial", "Event Type")
        The initial column headers for data. Only used when
        ``log_or_df`` is a file path.
    """

    def __init__(
        self,
        log_or_df: str | Path | pd.DataFrame,
        trial_types: tuple[str],
        use_first_pulse: bool = True,
        convert_to_seconds: Optional[list[str]] = None,
        initial_column_headers: tuple[str] = ("Trial", "Event Type"),
    ):

        df = _process_log_or_df(
            log_or_df,
            convert_to_seconds,
            initial_column_headers,
            divisor=10000,
            software="Presentation",
        )
        self.trial_types = trial_types

        if use_first_pulse:
            scanner_start_index = df.loc[
                df["Event Type"] == "Pulse", "Time"
            ].index.tolist()[0]
            self.scanner_start_time = df.loc[scanner_start_index, "Time"]
            self.df = df.loc[(scanner_start_index + 1) :, :]
        else:
            self.scanner_start_time = None
            self.df = df

    def _extract_onsets(
        self, scanner_start_time: Optional[float | int] = None
    ) -> list[float]:
        """Extract onset times for each block or event."""
        if scanner_start_time:
            self.scanner_start_time = scanner_start_time

        if not self.scanner_start_time:
            raise ValueError("A value for `scanner_start_time` needs to be given.")

        onsets = []
        for _, row in self.df.iterrows():
            if row["Event Type"] == "Picture" and row["Code"] in self.trial_types:
                onset = row["Time"] - self.scanner_start_time
                onsets.append(onset)

        return onsets

    def _extract_trial_types(self) -> list[str]:
        """Extract trial types for each block or event."""
        trial_types = []
        for _, row in self.df.iterrows():
            if row["Event Type"] == "Picture" and row["Code"] in self.trial_types:
                trial_types.append(row["Code"])

        return trial_types


class PresentationBlockExtractor(PresentationExtractor, BlockExtractor):
    """
    Extract onsets, durations, and trial types from Presentation logs using a block design.

    .. warning::
       - May not capture all edge cases.
       - If duration is fixed, it may be best to simply changed all
         values of "duration" in the events DataFrame to that
         fixed value.

    Parameters
    ----------
    log_or_df: :obj:`str`, :obj:`Path`, :obj:`pd.DataFrame`
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

    use_first_pulse: :obj:`bool`, default=True
        Uses the timing of the first pulse as the start time for the scanner,
        which is used to compute the onset times of the trials relative
        to the scanner start time.

        .. note::
           If set to False, a scanner start time can be supplied to
           ``self.extract_onsets``.

    convert_to_seconds: :obj:`list[str]` or :obj:`None`, default=None
        Convert the time resolution of the specified columns from 0.1ms to seconds.

        .. note:: Recommend time resolution of the "Time" column to be converted.

    initial_column_headers: :obj:`tuple[str]`, default=("Trial", "Event Type")
        The initial column headers for data. Only used when
        ``log_or_df`` is a file path.
    """

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
               If ``use_first_pulse`` is set to True during class initialization, then
               the ``self.scanner_start_time`` is set to the first detected pulse in the log
               data and this value can remain as None. If this value needs to be
               overriden, then the value supplied to ``scanner_start_time`` can be
               set.

        Returns
        -------
        list[float]
            A list of onset times for each block.
        """
        return self._extract_onsets(scanner_start_time=scanner_start_time)

    def extract_durations(self, rest_block_code: Optional[str] = None) -> list[float]:
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
            index of the trial types given by ``trial_types``.

        Returns
        -------
        list[float]
            A list of durations for each block.
        """
        durations = []
        for row_indx, row in self.df.iterrows():
            if row["Event Type"] == "Picture" and row["Code"] in self.trial_types:
                block_end_indx = _get_next_block_indx(
                    trial_series=self.df["Code"],
                    curr_row_indx=row_indx,
                    rest_block_code=rest_block_code,
                    trial_types=self.trial_types,
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
        return self._extract_trial_types()


class PresentationEventExtractor(PresentationExtractor, EventExtractor):
    """
    Extract onsets, durations, and trial types from Presentation logs using an event design.

    .. warning::
       - May not capture all edge cases.
       - If duration is fixed, it may be best to simply changed all
         values of "duration" in the events DataFrame to that
         fixed value.

    Parameters
    ----------
    log_or_df: :obj:`str`, :obj:`Path`, :obj:`pd.DataFrame`
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

    use_first_pulse: :obj:`bool`, default=True
        Uses the timing of the first pulse as the start time for the scanner,
        which is used to compute the onset times of the trials relative
        to the scanner start time.

        .. note::
           If set to False, a scanner start time can be supplied to
           ``self.extract_onsets``.

    convert_to_seconds: :obj:`list[str]` or :obj:`None`, default=None
        Convert the time resolution of the specified columns from 0.1ms to seconds.

        .. note:: Recommend time resolution of the "Time" column to be converted.

    initial_column_headers: :obj:`tuple[str]`, default=("Trial", "Event Type")
        The initial column headers for data. Only used when
        ``log_or_df`` is a file path.
    """

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
               If ``use_first_pulse`` is set to True during class initialization, then
               the ``self.scanner_start_time`` is set to the first detected pulse in the log
               data and this value can remain as None. If this value needs to be
               overriden, then the value supplied to ``scanner_start_time`` can be
               set.

        Returns
        -------
        list[float]
            A list of onset times for each event.
        """
        return self._extract_onsets(scanner_start_time=scanner_start_time)

    def _extract_durations_and_responses(self) -> tuple[list[float], list[str]]:
        """
        Extract durations and responses for each event.

        Duration is computed as the difference between the event stimulus
        and the response. When no response is given, the duration is the
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
        durations, responses = [], []
        for row_indx, row in self.df.iterrows():
            if row["Event Type"] == "Picture" and row["Code"] in self.trial_types:
                trial_num = row["Trial"]
                response_row = self.df[
                    (self.df["Trial"] == trial_num)
                    & (self.df["Event Type"] == "Response")
                ]
                if not response_row.empty:
                    duration = response_row.iloc[0]["Time"] - row["Time"]
                    response = row["Stim Type"]
                else:
                    duration = self.df.loc[(row_indx + 1), "Time"] - row["Time"]
                    response = "nan"

                durations.append(duration)
                responses.append(response)

        return durations, responses

    def extract_durations(self) -> list[float]:
        """
        Extract the duration for each event.

        Duration is computed as the difference between the event stimulus
        and the response. When no response is given, the duration is the
        difference between the starting time of that trial and the starting
        time of the subsequent stimuli.

        Returns
        -------
        list[float]
            A list of durations for each event.
        """
        durations, _ = self._extract_durations_and_responses()

        return durations

    def extract_trial_types(self) -> list[str]:
        """
        Extract the trial type for each event.

        Returns
        -------
        list[str]
            A list of trial types for each event.
        """
        return self._extract_trial_types()

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
        _, responses = self._extract_durations_and_responses()

        return responses


class EPrimeBlockExtractor(BlockExtractor):
    """
    Extract onsets, durations, and trial types from E-Prime 3 logs using a block design.

    .. warning::
        - May not capture all edge cases.
        - If duration is fixed, it may be best to simply changed all
          values of "duration" in the events DataFrame to that
          fixed value.

    Parameters
    ----------
    log_or_df: :obj:`str`, :obj:`Path`, :obj:`pd.DataFrame`
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

    convert_to_seconds: :obj:`list[str]` or :obj:`None`, default=None
        Convert the time resolution of the specified columns from milliseconds to seconds.

        .. note::
           Recommend time resolution of the columns containing the onset time
           be converted to seconds.

    initial_column_headers: :obj:`tuple[str]`, default=("ExperimentName", "Subject")
        The initial column headers for data. Only used when
        ``log_or_df`` is a file path.
    """

    def __init__(
        self,
        log_or_df: str | Path | pd.DataFrame,
        trial_types: tuple[str],
        onset_column_name: str,
        procedure_column_name: str,
        convert_to_seconds: Optional[list[str]] = None,
        initial_column_headers: tuple[str] = ("ExperimentName", "Subject"),
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

        trials_list = self.df[self.procedure_column_name].tolist()
        # Get the starting index for each block via grouping indices with the same
        # trial type
        starting_block_indices = []
        current_index = 0
        for _, group in itertools.groupby(trials_list):
            starting_block_indices.append(current_index)
            current_index += len(list(group))

        for trial_type in self.df[self.procedure_column_name].unique():
            if trial_type not in trial_types:
                trial_indxs = self.df[
                    self.df[self.procedure_column_name] == trial_type
                ].index.to_list()
                starting_block_indices = set(starting_block_indices).difference(
                    trial_indxs
                )

        self.starting_block_indices = sorted(list(starting_block_indices))

    def extract_onsets(self, scanner_start_time: float | int) -> list[float]:
        """
        Extract the onset times for each block.

        Onset is calculated as the difference between the event time and
        the scanner start time.

        Parameters
        ----------
        scanner_start_time: :obj:`float` or :obj:`int`
            The scanner start time. Used to compute onset relative to
            the start of the scan.

        Returns
        -------
        list[float]
            A list of onset times for each block.
        """
        onsets = []
        for row_indx, row in self.df.iterrows():
            if row[self.procedure_column_name] in self.trial_types:
                if row_indx not in self.starting_block_indices:
                    continue

                onsets.append((row[self.onset_column_name] - scanner_start_time))

        return onsets

    def extract_durations(self, rest_block_code: Optional[str] = None) -> list[float]:
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
            index of the trial types given by ``trial_types``.

        Returns
        -------
        list[float]
            A list of durations for each block.
        """
        durations = []
        for row_indx, row in self.df.iterrows():
            if row_indx not in self.starting_block_indices:
                continue

            block_end_indx = _get_next_block_indx(
                trial_series=self.df[self.procedure_column_name],
                curr_row_indx=row_indx,
                rest_block_code=rest_block_code,
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
        trial_types = []
        for row_indx, row in self.df.iterrows():
            if row_indx not in self.starting_block_indices:
                continue

            trial_types.append(row[self.procedure_column_name])

        return trial_types


class EPrimeEventExtractor(EventExtractor):
    """
    Extract onsets, durations, and trial types from E-Prime 3 logs using an event design.

    .. warning::
        - May not capture all edge cases.
        - If duration is fixed, it may be best to simply changed all
          values of "duration" in the events DataFrame to that
          fixed value.

    Parameters
    ----------
    log_or_df: :obj:`str`, :obj:`Path`, :obj:`pd.DataFrame`
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

    procedure_column_name: :obj:`str`
        The name of the column containing the procedure names.

    convert_to_seconds: :obj:`list[str]` or :obj:`None`, default=None
        Convert the time resolution of the specified columns from milliseconds to seconds.

        .. note::
           Recommend time resolution of the columns containing the onset time
           and reaction time be converted to seconds.

    initial_column_headers: :obj:`tuple[str]`, default=("ExperimentName", "Subject")
        The initial column headers for data. Only used when
        ``log_or_df`` is a file path.
    """

    def __init__(
        self,
        log_or_df: str | Path | pd.DataFrame,
        trial_types: tuple[str],
        procedure_column_name: str,
        convert_to_seconds: Optional[list[str]] = None,
        initial_column_headers: tuple[str] = ("ExperimentName", "Subject"),
    ):

        self.df = _process_log_or_df(
            log_or_df,
            convert_to_seconds,
            initial_column_headers,
            divisor=1000,
            software="E-Prime",
        )
        self.trial_types = trial_types
        self.procedure_column_name = procedure_column_name

    def extract_onsets(
        self,
        scanner_start_time: float | int,
        onset_column_name: str,
    ) -> list[float]:
        """
        Extract the onset times for each event.

        Onset is calculated as the difference between the event time and
        the scanner start time.

        Parameters
        ----------
        scanner_start_time: :obj:`float` or :obj:`int`
            The scanner start time. Used to compute onset relative to
            the start of the scan.

        onset_column_name: :obj:`str`
            The name of the column containing stimulus onset time.

        Returns
        -------
        list[float]
            A list of onset times for each event.
        """
        onsets = []
        for _, row in self.df.iterrows():
            if row[self.procedure_column_name] in self.trial_types:
                onsets.append((row[onset_column_name] - scanner_start_time))

        return onsets

    def extract_durations(self, duration_column_name: str) -> list[float]:
        """
        Extract the duration for each event.

        Duration is typically the reaction time for event-related designs.

        Parameters
        ----------
        duration_column_name: :obj:`str`
            The name of the column containing the duration or reaction time.

        Returns
        -------
        list[float]
            A list of durations for each event.
        """
        durations = []
        for _, row in self.df.iterrows():
            if row[self.procedure_column_name] in self.trial_types:
                durations.append(row[duration_column_name])

        return durations

    def extract_trial_types(self) -> list[str]:
        """
        Extract the trial type for each event.

        Returns
        -------
        list[str]
            A list of trial types for each event.
        """
        trial_types = []
        for _, row in self.df.iterrows():
            if row[self.procedure_column_name] in self.trial_types:
                trial_types.append(row[self.procedure_column_name])

        return trial_types

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
        for _, row in self.df.iterrows():
            if row[self.procedure_column_name] in self.trial_types:
                try:
                    response_val = int(row[accuracy_column_name])
                except:
                    response_val = "nan"

                if response_val != "nan":
                    response = {"0": "incorrect", "1": "correct"}.get(str(response_val))
                else:
                    response = "nan"

                responses.append(response)

        return responses
