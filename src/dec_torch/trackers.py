"""A module for experiment tracking."""

from collections.abc import Mapping
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol, Self

import pandas as pd

if TYPE_CHECKING:
    from types import TracebackType


class Phase(StrEnum):
    """The phase of an experiment step.

    Example:
        >>> print(Phase.TRAIN)
        train
        >>> Phase.TRAIN == "train"
        True
    """

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class MetricTracker(Protocol):
    """An interface for logging step metrics.

    Example:
        >>> class MyTracker:
        ...     def log_metrics(self, phase, step, metrics):
        ...         print(f"{phase} step {step}: {metrics}")
        >>> tracker = MyTracker()
        >>> tracker.log_metrics("train", 1, {"loss": 0.5})
        train step 1: {'loss': 0.5}
    """

    def log_metrics(
        self, phase: str, step: int, metrics: Mapping[str, float]
    ) -> None:
        """Logs the metrics of the current step (e.g., epoch, iteration).

        Args:
            phase: The phase of the experiment run.
            step: The step of the experiment run.
            metrics: The metrics to log.

        Example:
            >>> class MyTracker:
            ...     def log_metrics(self, phase, step, metrics):
            ...         pass
            >>> tracker = MyTracker()
            >>> tracker.log_metrics("val", 5, {"loss": 0.32, "acc": 0.95})
        """


class ParamTracker(Protocol):
    """An interface for logging experiment parameters.

    Example:
        >>> class MyParamTracker:
        ...     def log_params(self, params):
        ...         print(params)
        >>> tracker = MyParamTracker()
        >>> tracker.log_params({"lr": 0.001, "batch_size": 32})
        {'lr': 0.001, 'batch_size': 32}
    """

    def log_params(self, params: Mapping[str, Any]) -> None:
        """Logs the parameters of the current experiment run.

        Example:
            >>> class MyParamTracker:
            ...     def log_params(self, params):
            ...         pass
            >>> tracker = MyParamTracker()
            >>> tracker.log_params({"lr": 0.01, "epochs": 100})
        """


class ExperimentTracker(ParamTracker, MetricTracker, Protocol):
    """An interface for experiment tracking.

    Example:
        >>> from typing import Any
        >>> class MyTracker:
        ...     def log_metrics(self, phase, step, metrics):
        ...         pass
        ...     def log_params(self, params):
        ...         pass
        >>> tracker: ExperimentTracker = MyTracker()
        >>> tracker.log_params({"lr": 0.01})
        >>> tracker.log_metrics("train", 1, {"loss": 0.5})
    """


class HistoryTracker:
    """In-memory experiment tracker that records metrics and parameters.

    This class provides a simple way to track and store training/validation
    metrics during model training. It implements the ExperimentTracker
    protocol and provides convenient conversion to pandas DataFrame for
    analysis.

    Example:
        >>> tracker = HistoryTracker()
        >>> tracker.log_metrics(phase=Phase.TRAIN, step=1, metrics={'loss': 0.45})
        >>> tracker.log_metrics(phase=Phase.VAL, step=1, metrics={'loss': 0.42})
        >>> df = tracker.history
        >>> print(df.head())
           step phase metric  score
        0     1  train   loss   0.45
        1     1    val   loss   0.42
    """

    def __init__(self) -> None:
        """Initialize HistoryTracker with empty storage."""
        self._history: list[dict[str, Any]] = []
        self._params: dict[str, Any] = {}

    def log_metrics(
        self, phase: str, step: int, metrics: Mapping[str, float]
    ) -> None:
        """Record metrics in history log.

        Args:
            phase: The phase of the experiment run.
            step: The step number (e.g., epoch).
            metrics: Dictionary of metric names to values.
        """
        for metric, score in metrics.items():
            self._history.append(
                {
                    "step": int(step),
                    "phase": str(phase),
                    "metric": metric,
                    "score": float(score),
                }
            )

    def log_params(self, params: Mapping[str, Any]) -> None:
        """Record experiment parameters.

        Args:
            params: Dictionary of parameter names to values.
        """
        self._params.update(params)

    @property
    def history(self) -> pd.DataFrame:
        """Access all history at once as a DataFrame.

        Converts the internal storage format to a pandas DataFrame
        for analysis and visualization.

        Returns:
            DataFrame with columns ['step', 'phase', 'metric', 'score'].

        Example:
            >>> df = tracker.history
            >>> print(df[df['metric'] == 'loss'].head())
        """
        df = pd.DataFrame(self._history)
        for col, dtype in [
            ("step", "int"),
            ("phase", "category"),
            ("metric", "category"),
            ("score", "float"),
        ]:
            if col not in df.columns:
                df[col] = pd.Series(dtype=dtype)
        if not df.empty:
            df["phase"] = df["phase"].astype("category")
            df["metric"] = df["metric"].astype("category")
            df = df.sort_values(by=["step", "phase", "metric"])
        return df

    @property
    def params(self) -> dict[str, Any]:
        """Access logged parameters.

        Returns:
            Dictionary of logged parameter names to values.
        """
        return self._params.copy()

    def __enter__(self) -> Self:
        """Enter the context."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit the context."""

    def __str__(self) -> str:
        """String representation of the history tracker."""
        return str(self.history)
