"""run_time.py - the run time of a pipeline run and the selection of time points by date and time.

The NCP files can cover 24 hours or several days. The run time comes from the CGM (its scenario time, as pypowsybl
reports it). When it lies inside a time-point series of the NCP files, the run takes that date and hour: every reader of
time points (bids, PATL, participation factors, unit costs, group limits) uses only the points of the run's date and
clock time (a timestamp earlier on the clock than the first one belongs to the next day; a point without a date
counts by clock time). Otherwise the run keeps the pipeline's start hour and each reader falls back: a series on one
day is matched by clock time as before, a series on several days uses its earliest day, with a counting warning.
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
import re
from dataclasses import dataclass
from datetime import date
from typing import Iterable, Optional

log = logging.getLogger(__name__)

_DATE_TIME = re.compile(r"(?:(\d{4}-\d{2}-\d{2}))?T(\d{2}):(\d{2}):")


@dataclass(frozen=True)
class RunTime:
    """day: date.toordinal() of the run's date; minute: minute of the day of the run's hour."""
    day: int
    minute: int


_CURRENT: contextvars.ContextVar = contextvars.ContextVar("rao_run_time", default=None)


@contextlib.contextmanager
def run_time_scope():
    """A run's scope: the run time set inside it is gone when it ends."""
    token = _CURRENT.set(None)
    try:
        yield
    finally:
        _CURRENT.reset(token)


def set_run_time(run_time: Optional[RunTime]) -> None:
    _CURRENT.set(run_time)


def current() -> Optional[RunTime]:
    return _CURRENT.get()


def iso_day(iso: str) -> Optional[int]:
    """Day number (date.toordinal) of an ISO datetime string; None when it carries no date."""
    m = _DATE_TIME.search(iso)
    if not m or m.group(1) is None:
        return None
    try:
        return date.fromisoformat(m.group(1)).toordinal()
    except ValueError:
        return None


def iso_datetime_minutes(iso: str) -> Optional[int]:
    """Minutes since day 0 of an ISO datetime with a date, else None (used for the time span of the NCP files)."""
    day, m = iso_day(iso), _DATE_TIME.search(iso)
    if day is None or m is None:
        return None
    return day * 1440 + int(m.group(2)) * 60 + int(m.group(3))


_AT_TIME = re.compile(rb"atTime>\s*([^<\s]+)\s*<")


_TIME_POINT = re.compile(rb"<nc:(\w+TimePoint)\b[^>]*>(.*?)</nc:\1>", re.S)
_PARENT = re.compile(rb'rdf:resource="#?([^"]+)"')


def ncp_series_spans(paths: Iterable) -> list[tuple[int, int]]:
    """The time span (first, last, minutes since day 0, dated points only) of every time-point series of the NCP files:
    the points of one schedule (the first reference of a time point names its schedule or group)."""
    series: dict[tuple, list[int]] = {}
    for path in paths:
        for match in _TIME_POINT.finditer(open(path, "rb").read()):
            body = match.group(2)
            at_time, parent = _AT_TIME.search(body), _PARENT.search(body)
            if at_time is None:
                continue
            minutes = iso_datetime_minutes(at_time.group(1).decode("utf-8", "replace"))
            if minutes is not None:
                key = (str(path), match.group(1), parent.group(1) if parent else b"")
                bounds = series.setdefault(key, [minutes, minutes])
                bounds[0], bounds[1] = min(bounds[0], minutes), max(bounds[1], minutes)
    return [(first, last) for first, last in series.values()]


def _stamp(minutes: int) -> str:
    return f"{date.fromordinal(minutes // 1440).isoformat()}T{minutes % 1440 // 60:02d}:{minutes % 60:02d}Z"


def choose_run_time(cgm_time, spans: list[tuple[int, int]]) -> tuple[Optional[RunTime], str]:
    """(RunTime or None, text for the log). cgm_time: the scenario time of the CGM as a datetime (UTC) or None; spans:
    the time spans of the time-point series of the NCP files. The run time is the CGM time when it lies inside the span
    of at least one series (inclusive: a single point at that time counts), else None (fallback, the text says why)."""
    if cgm_time is None:
        return None, "the CGM has no scenario time"
    stamp = cgm_time.strftime("%Y-%m-%dT%H:%MZ")
    if not spans:
        return None, f"the CGM scenario time is {stamp} but the NCP files have no dated time-point series"
    minutes = cgm_time.toordinal() * 1440 + cgm_time.hour * 60 + cgm_time.minute
    if not any(first <= minutes <= last for first, last in spans):
        return None, (f"the CGM scenario time {stamp} is inside no time-point series of the NCP files "
                      f"({len(spans)} series, from {_stamp(min(f for f, _ in spans))} to {_stamp(max(l for _, l in spans))})")
    return RunTime(day=cgm_time.toordinal(), minute=cgm_time.hour * 60 + cgm_time.minute), f"the CGM scenario time {stamp}"


def required_day(days: Iterable[Optional[int]], what: str, source: str) -> Optional[int]:
    """The day the points of one series must lie on, or None for clock time only. With a run time: its day. Without
    one (the fallback) None, as before the run time existed: the points are matched by clock time; a series on several
    days is then ambiguous (points of different days share a clock time and one of them wins), which one warning says."""
    run_time = current()
    if run_time is not None:
        return run_time.day
    dated = {day for day in days if day is not None}
    if len(dated) > 1:
        log.warning(
            "%s: %s has time points on %d days and the run time is not on a day of the NCP series (or the CGM has no "
            "scenario time) - matched by clock time only (fallback), points of different days with the same clock time "
            "are not told apart", what, source, len(dated),
        )
    return None


def day_matches(point_day: Optional[int], required: Optional[int], clock_minute: int, first_minute: int) -> bool:
    """True when a point on point_day is the one for the timestamp at clock_minute: no required day or no date on the
    point = clock time only; otherwise the required day, plus one for a clock time earlier than the first timestamp."""
    if required is None or point_day is None:
        return True
    return point_day == required + (1 if clock_minute < first_minute else 0)
