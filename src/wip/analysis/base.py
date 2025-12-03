"""Base classes for analyses and analysis backends"""

import pathlib
import shutil
import tempfile
from abc import ABC, abstractmethod
from collections.abc import MutableMapping
from typing import Any, TypeVar, Generic

from pydantic import BaseModel

from awsem.infrastructure import DefaultAttr
from awsem.typing import StrPath


B = TypeVar("B", bound="AnalysisBackend")


class WorkingDirectoryInfo(BaseModel):
    path: pathlib.Path
    is_temporary: bool


class BackendParameters(BaseModel):
    pass


class AnalysisParameters(BaseModel):
    cleanup: bool = True
    parse_output: bool = True
    backend_parameters: BackendParameters = None
    env: MutableMapping[str, str] | None = None


class AnalysisInput(BaseModel):
    pass


class Analysis(Generic[B]):
    """Base class for analysis tasks

    Provides default mechanisms to:
       * Manage (temporary) working directories
       * Handle input parameters

    Interfaces to orchestrate an analysis can be based on this class
    and make use of different backends (plugins) to realise that analysis.

    Analyses not using a backend can implement :meth:`run` which
    otherwise needs to be implemented by the backend. In general, all
    methods looked up but not found on the :class:`Analysis` instance
    are delegated to the backend if available.

    Analyses can be associated with a number of :attr:`_registered_backends`
    which can be selected by providing the backend name as string.

    Analyses specify Pydantic models for  :attr:`Input`
    (files/objects to perform the analysis on)
    and :attr:`Parameters` (analysis parameters controlling the analysis).
    Analyses supporting backends will also make use of the :attr:`Parameters`
    model specified on the chosen backend.

    On init, given `input` should be passed as the sole positional argument
    (if any).
    Additional keyword arguments are passed to the :attr:`Parameters` model
    and if available to the backend's :attr:`Parameters`.

    Attributes:
        input: Validated Analysis input data.
        backend: Optional backend matching the type required by this analysis.
        working_directory: Optional working directory seed for the analysis. The
            actually used working directory can be obtained
            via :meth:`get_working_directory`
            which can also optionally be cached in the :attr:`results` mapping.
        results: Analysis results after running the analysis.
        parameters: Validated analysis parameters.
    """

    _registered_backends: dict[str, type[B]] = {}
    Input: type[AnalysisInput] = AnalysisInput
    Parameters: type[AnalysisParameters] = AnalysisParameters
    results: DefaultAttr[MutableMapping[str, Any]] = DefaultAttr(lambda: {})

    def __init__(
        self,
        input: AnalysisInput | MutableMapping[str, Any] | None = None,
        *,
        backend: str | B | None = None,
        working_directory: StrPath | None = None,
        **kwargs,
    ) -> None:
        if input is None:
            input = {}
        self.input = self.Input.model_validate(input)

        self.backend = backend
        self.working_directory = working_directory

        explicit_backend_params = kwargs.pop("backend_parameters", None)
        self.parameters = self.Parameters.model_validate(kwargs)

        if self.backend is not None:
            if explicit_backend_params is not None:
                backend_params = self.backend.Parameters.model_validate(
                    explicit_backend_params
                )
            else:
                backend_params = self.backend.Parameters.model_validate(kwargs)
            self.parameters.backend_parameters = backend_params

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown method calls to backend"""

        if self.backend is not None:
            attr = getattr(self.backend, name, None)
            if callable(attr):
                # Wrap backend method so it receives self automatically
                return lambda *a, **kw: attr(self, *a, **kw)

        raise AttributeError(
            f"{type(self).__name__} has no attribute '{name}' (backed={self.backend!r})"
        )

    @property
    def backend(self) -> B | None:
        return self._backend

    @backend.setter
    def backend(self, backend: str | B | None) -> None:
        if isinstance(backend, str):
            backend_cls = self._registered_backends.get(backend.lower())
            if backend_cls is None:
                raise ValueError(f"Unknown analysis backend: {backend}")
            self._backend = backend_cls()
        else:
            self._backend = backend

    @property
    def working_directory(self) -> pathlib.Path | None:
        return self._working_directory

    @working_directory.setter
    def working_directory(self, path: StrPath | None) -> None:
        self._working_directory = pathlib.Path(path) if path else None

    def get_working_directory(
        self, cache: bool = True, **kwargs
    ) -> WorkingDirectoryInfo:
        """Get or create a working directory for an analysis

        If no :attr:`working_directory` was set, a temporary directory
        will be created. The working directory information will be returned and
        stored in the :attr:`results` mapping under the key
        `"working_directory"` if `cache=True`. If such information is already
        present there, that will be returned directly.

        Keyword args:
            cache: Whether to store the working directory info under
                :attr:`results["working_directory"]`. Useful to keep track
                of the working directory across multiple calls during an analysis
                and avoid duplicate creation.
            kwargs: Additional arguments passed to `tempfile.mkdtemp`
        """

        results = self.results
        working_directory_info = results.get("working_directory")
        if working_directory_info is not None:
            return working_directory_info

        working_directory = self.working_directory
        if working_directory is None:
            working_directory = tempfile.mkdtemp(**kwargs)
            working_directory = pathlib.Path(working_directory)
            working_directory_info = WorkingDirectoryInfo(
                path=working_directory, is_temporary=True
            )
        else:
            working_directory.mkdir(parents=True, exist_ok=True)
            working_directory_info = WorkingDirectoryInfo(
                path=working_directory, is_temporary=False
            )

        if cache:
            results["working_directory"] = working_directory_info
            self.results = results

        return working_directory_info

    def reset_results(self) -> None:
        """Reset the analysis"""
        self.results = None

    def cleanup(
        self, working_directory_info: WorkingDirectoryInfo | None = None
    ) -> None:
        """Cleanup temporary working directory

        Keyword args:
            working_directory_info: Optional working directory info to cleanup. If not provided,
                the info stored in :attr:`results` under `"working_directory"` will be used.
        """

        if working_directory_info is None:
            working_directory_info = self.results.get("working_directory")

        if not working_directory_info:
            return

        working_directory = working_directory_info.path
        if (working_directory is None) or (not working_directory.exists()):
            return

        if working_directory_info.is_temporary:
            shutil.rmtree(working_directory)


class AnalysisBackend(ABC):
    """Abstract base class for analysis backends

    Concrete backends need to implement :meth:`run` to be
    usable as a plugin on an :class:`Analysis` object.
    """

    Parameters: type[BackendParameters] = BackendParameters

    @abstractmethod
    def run(self, main: Analysis) -> None:
        """Run the analysis using the provided main analysis instance"""
