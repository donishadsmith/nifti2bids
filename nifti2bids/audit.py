from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
import numpy as np

from ._exceptions import PathDoesNotExist


class BIDSAuditor:
    """
    Check availability of different files for each subject in the
    BIDS directory.

    Parameters
    ----------
    bids_dir: :obj:`str` or :obj:`Path`
        The root of the BIDS compliant directory.

    Attributes
    ----------
    bids_dir: :obj:`str` or :obj:`Path`
        The root of the BIDS compliant directory.

    derivatives_dir: obj:`bool`, :obj:`str`, :obj:`Path`, or obj:`None`, default=None
        The root to the fMRIPrep directory.
    """

    def __init__(
        self, bids_dir: str | Path, derivatives_dir: Optional[bool | str | Path] = None
    ):
        bids_dir = Path(bids_dir)
        if not bids_dir.exists():
            raise PathDoesNotExist(bids_dir)

        self.bids_dir = bids_dir

        if derivatives_dir and isinstance(derivatives_dir, str):
            derivatives_dir = Path(derivatives_dir)

        if derivatives_dir and isinstance(derivatives_dir, (Path, str)):
            if not derivatives_dir.exists():
                raise PathDoesNotExist(derivatives_dir)

        self.derivatives_dir = derivatives_dir

    @staticmethod
    @lru_cache(maxsize=2)
    def _call_layout(bids_dir: str | Path, derivatives_dir: Optional[str | Path]):
        """
        Return the ``BIDSLayout``. Up to four layouts are cached.

        Parameters
        ----------
        bids_dir: :obj:`str` or :obj:`Path`
            The root of the BIDS compliant directory.

        derivatives_dir: :obj:`str`, :obj:`Path`, or obj:`None`
            The root to the fMRIPrep directory.

        Returns
        -------
        BIDSLayout
            A layout of the raw BIDS directory.
        """
        try:
            import bids
        except ModuleNotFoundError:
            raise ModuleNotFoundError("The pybids package must be installed.")

        layout_dict = {"root": bids_dir}
        if derivatives_dir:
            layout_dict.update({"derivatives": derivatives_dir})

        return bids.BIDSLayout(**layout_dict)

    @staticmethod
    @lru_cache(maxsize=2)
    def _get_subjects_and_sessions(
        bids_dir: str | Path, derivatives_dir: Optional[str | Path], layout
    ) -> tuple[list[str], list[str]]:
        """
        Gets dictionary of subject and their sessions. Caches results.
        ``bids_dir`` and ``derivatives_dir`` are included as parameters
        so that results are cached properly.

        Parameters
        ----------
        bids_dir: :obj:`str` or :obj:`Path`
            The root of the BIDS compliant directory.

        derivatives_dir: :obj:`str`, :obj:`Path`, or obj:`None`
            The root to the fMRIPrep directory.

        layout: :obj:`BIDSLayout`
            The layout of the raw BIDS directory.

        Returns
        -------
        tuple[list[str], list[str]]
            A tuple consisting of lists, where the first list are the subject IDs and the
            second list are the session IDs. If a subject as multiple sessions, then the ID
            is repeated.
        """
        subject_list, session_list = [], []
        for subject in layout.get_subjects():
            session_ids = layout.get(
                return_type="id", subject=subject, target="session"
            )
            if session_ids:
                subject_list.extend([subject] * len(session_ids))
                session_list.extend(session_ids)
            else:
                subject_list.append(subject)
                session_list.append(np.nan)

        return subject_list, session_list

    @staticmethod
    @lru_cache(maxsize=4)
    def _get_file_availability(
        bids_dir: str | Path,
        derivatives_dir: Optional[str | Path],
        layout,
        file_type: Literal["nifti", "events", "sidecar"],
        scope: Literal["raw", "derivatives"],
        template_space: Optional[str] = None,
        run_id: Optional[str | int] = None,
    ) -> dict[str, list[str]]:
        """
        Creates file availability dictionary. Caches results.

        Parameters
        ----------
        bids_dir: :obj:`str` or :obj:`Path`
            The root of the BIDS compliant directory.

        derivatives_dir: :obj:`str`, :obj:`Path`, or obj:`None`
            The root to the fMRIPrep directory.

        layout: :obj:`BIDSLayout`
            The layout of the raw BIDS directory.

        file_type: :obj:`Literal["nifti", "events", "sidecar"]`
            The type of file to query.

        scope: :obj:`Literal["raw", "derivatives"]
            Whether to check in the raw or derivatives directory.

        template_space: :obj:`str` or :obj:`None`, default=None
            The template space to check for. Only relevent when scope is
            ``derivatives``.

        run_id: obj:`str`, obj:`int`, or obj:`None`, default=None
            The specific run ID to check for.

        Return
        ------
        dict[str, list[str]]:
            A dictionary denoting file availability, where "Yes" means the file
            is available and "No" means that the file is not available.

        """
        ext_dict = {"nifti": "nii.gz", "events": "tsv", "sidecar": "json"}

        subjects, sessions = BIDSAuditor._get_subjects_and_sessions(
            bids_dir, derivatives_dir, layout
        )
        targets = layout.get_tasks()
        if file_type != "events" and scope == "raw":
            targets = ["T1w"] + targets

        df_dict = {"subject": subjects, "session": sessions} | {
            target: [] for target in targets
        }

        for subject, session in zip(subjects, sessions):
            for target in targets:
                suffix = (
                    "T1w"
                    if target == "T1w"
                    else ("bold" if file_type != "events" else "events")
                )

                query_dict = {
                    "return_type": "file",
                    "scope": scope,
                    "subject": subject,
                    "suffix": suffix,
                    "extension": ext_dict[file_type],
                }

                if isinstance(session, (int, str)):
                    query_dict.update({"session": session})

                if scope == "derivatives" and template_space:
                    query_dict.update({"space": template_space})

                if run_id:
                    query_dict.update({"run": run_id})

                file = (
                    layout.get(**query_dict)
                    if suffix == "T1w"
                    else layout.get(**query_dict, task=target)
                )
                df_dict[target].append(("Yes" if file else "No"))

        return df_dict

    def _create_df(
        self,
        file_type: Literal["nifti", "events", "sidecar"],
        scope: Literal["raw", "derivatives"],
        template_space: Optional[str] = None,
        run_id: Optional[str | int] = None,
    ):
        """
        Creates DataFrame of file availability.

        Parameters
        ----------
        file_type: :obj:`Literal["nifti", "events", "sidecar"]`
            The type of file to query.

        scope: :obj:`Literal["raw", "derivatives"]
            Whether to check in the raw or derivatives directory.

        template_space: :obj:`str` or :obj:`None`, default=None
            The template space to check for. Only relevent when scope is
            ``derivatives``.

        run_id: obj:`str`, obj:`int`, or obj:`None`, default=None
            The specific run ID to check for.

        Returns
        -------
        pandas.DataFrame
            A Pandas DataFrame denoting file availability, where "Yes" means the file
            is available and "No" means that the file is not available.
        """
        layout = self._call_layout(self.bids_dir, self.derivatives_dir)

        return pd.DataFrame(
            self._get_file_availability(
                self.bids_dir,
                self.derivatives_dir,
                layout,
                file_type,
                scope,
                template_space,
                run_id,
            )
        )

    @staticmethod
    def clear_caches() -> None:
        """
        Clear all cached data.

        Example
        -------
        >>> from nifti2bids.audit import BIDSAuditor
        >>> BIDSAuditor.clear_caches()
        """
        BIDSAuditor._call_layout.cache_clear()
        BIDSAuditor._get_subjects_and_sessions.cache_clear()
        BIDSAuditor._get_file_availability.cache_clear()

    def check_raw_nifti_availability(
        self, run_id: Optional[str | int] = None
    ) -> pd.DataFrame:
        """
        Checks the availability of the unpreprocessed NIfTI files for each subject and their sessions.
        Specifically checks if the T1w image is available and if unpreprocessed NIfTI images for
        all tasks (i.e. "rest", "flanker", etc) is available.

        .. important::
            Checks if at least one run of data is available.

        Parameters
        ----------
        run_id: obj:`str`, obj:`int`, or obj:`None`, default=None
            The specific run ID to check for.

        Returns
        -------
        pandas.DataFrame
            A Pandas DataFrame denoting file availability, where "Yes" means the file
            is available and "No" means that the file is not available.

        Notes
        -----
        Example of output table:

        +----------+---------+-----+-------+---------+------+------+----------+
        | subject  | session | T1w | nback | flanker | mtle | mtlr | princess |
        +==========+=========+=====+=======+=========+======+======+==========+
        | 101      | 01      | Yes | Yes   | Yes     | Yes  | Yes  | Yes      |
        +----------+---------+-----+-------+---------+------+------+----------+
        | 101      | 02      | Yes | Yes   | No      | Yes  | Yes  | Yes      |
        +----------+---------+-----+-------+---------+------+------+----------+
        | 102      | 01      | Yes | Yes   | Yes     | Yes  | Yes  | Yes      |
        +----------+---------+-----+-------+---------+------+------+----------+
        | 103      | 01      | No  | Yes   | Yes     | Yes  | Yes  | Yes      |
        +----------+---------+-----+-------+---------+------+------+----------+
        """
        return self._create_df(file_type="nifti", scope="raw", run_id=run_id)

    def check_events_availability(
        self, run_id: Optional[str | int] = None
    ) -> pd.DataFrame:
        """
        Checks the availability of events TSV files for each subject and their sessions.
        Specifically checks if event TSV files are available for all tasks (i.e. "rest", "flanker", etc).

        .. important::
            Checks if at least one run of data is available.

        Parameters
        ----------
        run_id: obj:`str`, obj:`int`, or obj:`None`, default=None
            The specific run ID to check for.

        Returns
        -------
        pandas.DataFrame
            A Pandas DataFrame denoting file availability, where "Yes" means the file
            is available and "No" means that the file is not available.

        Notes
        -----
        Example of output table:

        +----------+---------+-------+---------+------+------+----------+
        | subject  | session | nback | flanker | mtle | mtlr | princess |
        +==========+=========+=======+=========+======+======+==========+
        | 101      | 01      | Yes   | Yes     | Yes  | Yes  | Yes      |
        +----------+---------+-------+---------+------+------+----------+
        | 101      | 02      | Yes   | No      | Yes  | Yes  | Yes      |
        +----------+---------+-------+---------+------+------+----------+
        | 102      | 01      | Yes   | Yes     | Yes  | Yes  | Yes      |
        +----------+---------+-------+---------+------+------+----------+
        | 103      | 01      | Yes   | Yes     | Yes  | Yes  | Yes      |
        +----------+---------+-------+---------+------+------+----------+
        """
        return self._create_df(file_type="events", scope="raw", run_id=run_id)

    def check_raw_sidecar_availability(
        self, run_id: Optional[str | int] = None
    ) -> pd.DataFrame:
        """
        Checks the availability of JSON sidecar files for each subject and their sessions.
        Specifically checks if the JSON sidecar for the T1w image and all task NIfTI images
        (i.e. "rest", "flanker", etc) are available.

        .. important::
            Checks if at least one run of data is available.

        Parameters
        ----------
        run_id: obj:`str`, obj:`int`, or obj:`None`, default=None
            The specific run ID to check for.

        Returns
        -------
        pandas.DataFrame
            A Pandas DataFrame denoting file availability, where "Yes" means the file
            is available and "No" means that the file is not available.

        Notes
        -----
        Example of output table:

        +----------+---------+-----+-------+---------+------+------+----------+
        | subject  | session | T1w | nback | flanker | mtle | mtlr | princess |
        +==========+=========+=====+=======+=========+======+======+==========+
        | 101      | 01      | Yes | Yes   | Yes     | Yes  | Yes  | Yes      |
        +----------+---------+-----+-------+---------+------+------+----------+
        | 101      | 02      | Yes | Yes   | No      | Yes  | Yes  | Yes      |
        +----------+---------+-----+-------+---------+------+------+----------+
        | 102      | 01      | Yes | Yes   | Yes     | Yes  | Yes  | Yes      |
        +----------+---------+-----+-------+---------+------+------+----------+
        | 103      | 01      | No  | Yes   | Yes     | Yes  | Yes  | Yes      |
        +----------+---------+-----+-------+---------+------+------+----------+
        """
        return self._create_df(file_type="sidecar", scope="raw", run_id=run_id)

    def check_preprocessed_nifti_availability(
        self, template_space: str, run_id: Optional[str | int] = None
    ) -> pd.DataFrame:
        """
        Checks the availability of the preprocessed NIfTI files for each subject and their sessions.
        Specifically checks if the preprocessed NIfTI images for all tasks (i.e. "rest", "flanker", etc)
        are available.

        .. important::
            Checks if at least one run of data is available.

        Parameters
        ----------
        template_space: :obj:`str`:
            The template space to check for (e.g., "MNIPediatricAsym").

        run_id: obj:`str`, obj:`int`, or obj:`None`, default=None
            The specific run ID to check for.

        Returns
        -------
        pandas.DataFrame
            A Pandas DataFrame denoting file availability, where "Yes" means the file
            is available and "No" means that the file is not available.

        Notes
        -----

        Example of output table:

        +----------+---------+-----+-------+---------+------+------+----------+
        | subject  | session | T1w | nback | flanker | mtle | mtlr | princess |
        +==========+=========+=====+=======+=========+======+======+==========+
        | 101      | 01      | Yes | Yes   | Yes     | Yes  | Yes  | Yes      |
        +----------+---------+-----+-------+---------+------+------+----------+
        | 101      | 02      | Yes | Yes   | No      | Yes  | Yes  | Yes      |
        +----------+---------+-----+-------+---------+------+------+----------+
        | 102      | 01      | Yes | Yes   | Yes     | Yes  | Yes  | Yes      |
        +----------+---------+-----+-------+---------+------+------+----------+
        | 103      | 01      | No  | Yes   | Yes     | Yes  | Yes  | Yes      |
        +----------+---------+-----+-------+---------+------+------+----------+
        """
        return self._create_df(
            file_type="sidecar",
            scope="derivatives",
            template_space=template_space,
            run_id=run_id,
        )


__all__ = ["BIDSAuditor"]
