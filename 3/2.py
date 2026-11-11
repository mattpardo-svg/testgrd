"""
run_profile.py - duration and memory of each main step of a run, and a summary table at the end.

mark(name) ends the running step (logging its duration and memory) and starts the next; a step that comes back (the MILP
build in a loop) is added up under its name. Memory is the RSS of the process (it includes pypowsybl): at the end of the
step and the peak of the samples taken by a background thread every 0.5 s. Without psutil only the time is logged (one
warning). Log only: nothing here writes a result file.
"""

import logging
import threading
import time
from typing import Callable, Optional

try:
    import psutil
except ImportError:                       # optional: time only without it
    psutil = None

log = logging.getLogger(__name__)

MIB = 1024 * 1024
SAMPLE_INTERVAL_SECONDS = 0.5
_DEFAULT = object()


def default_rss_reader() -> Optional[Callable[[], int]]:
    """A function returning the RSS of this process in bytes, or None when psutil is not installed."""
    if psutil is None:
        return None
    process = psutil.Process()
    return lambda: int(process.memory_info().rss)


class RunProfile:
    def __init__(self, clock=time.perf_counter, rss_reader=_DEFAULT, interval_seconds: Optional[float] = SAMPLE_INTERVAL_SECONDS):
        self._clock = clock
        self._read_rss = default_rss_reader() if rss_reader is _DEFAULT else rss_reader
        self._interval = interval_seconds
        self._lock = threading.Lock()
        self._steps: dict[str, dict] = {}
        self._current: Optional[dict] = None
        self._run_start = 0.0
        self._run_end = 0.0
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._started = False

    # --- sampling -------------------------------------------------------------------------------------------------
    def _rss(self) -> Optional[int]:
        if self._read_rss is None:
            return None
        try:
            return int(self._read_rss())
        except Exception:
            return None

    def sample_now(self) -> Optional[int]:
        rss = self._rss()
        with self._lock:
            if rss is not None and self._current is not None:
                self._current["peak"] = max(self._current["peak"], rss)
        return rss

    def _sample_loop(self) -> None:
        while not self._stop.wait(self._interval):
            self.sample_now()

    def thread_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # --- steps ----------------------------------------------------------------------------------------------------
    def _begin(self, name: str) -> None:
        rss = self._rss()
        with self._lock:
            self._current = {"name": name, "start": self._clock(), "peak": rss or 0}

    def _end(self) -> None:
        end_rss = self.sample_now()
        with self._lock:
            step, self._current = self._current, None
        if step is None:
            return
        seconds = self._clock() - step["start"]
        row = self._steps.setdefault(step["name"], {"seconds": 0.0, "rss_end": None, "peak": None})
        row["seconds"] += seconds
        row["rss_end"] = end_rss
        if end_rss is not None:
            row["peak"] = max(row["peak"] or 0, step["peak"], end_rss)
        if end_rss is None:
            log.info("Step '%s' took %.1f s", step["name"], seconds)
        else:
            log.info(
                "Step '%s' took %.1f s, RSS %d MiB (peak %d MiB)",
                step["name"], seconds, round(end_rss / MIB), round(row["peak"] / MIB),
            )

    def start(self, name: str) -> None:
        if self._started:
            self.finish()
        self._steps, self._started = {}, True
        if self._read_rss is None:
            log.warning("psutil is not installed: the run log gives the duration of each step but no memory (pip install psutil)")
        self._run_start = self._clock()
        self._begin(name)
        if self._read_rss is not None and self._interval:
            self._stop.clear()
            self._thread = threading.Thread(target=self._sample_loop, name="run-profile-sampler", daemon=True)
            self._thread.start()

    def mark(self, name: str) -> None:
        """End the running step and start `name`; the same name again is not a new step."""
        if not self._started or (self._current is not None and self._current["name"] == name):
            return
        self._end()
        self._begin(name)

    def finish(self) -> None:
        """End the running step, stop the sampler and log the summary table (nothing without a start)."""
        if not self._started:
            return
        self._end()
        self._run_end = self._clock()
        self._started = False
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._log_table()

    # --- table ----------------------------------------------------------------------------------------------------
    def rows(self) -> list[dict]:
        total = sum(row["seconds"] for row in self._steps.values()) or 1.0
        return [
            {
                "step": name, "seconds": row["seconds"], "share_pct": 100.0 * row["seconds"] / total,
                "rss_end_mib": None if row["rss_end"] is None else round(row["rss_end"] / MIB),
                "peak_mib": None if row["peak"] is None else round(row["peak"] / MIB),
            }
            for name, row in self._steps.items()
        ]

    def _log_table(self) -> None:
        rows = self.rows()
        with_memory = any(row["peak_mib"] is not None for row in rows)
        lines = [
            "Run summary (step, duration, share of total, RSS at end, peak RSS in MiB):" if with_memory
            else "Run summary (step, duration, share of total):"
        ]
        for row in rows:
            text = f"  {row['step']:<46}{row['seconds']:>9.1f} s{row['share_pct']:>8.1f} %"
            if with_memory:
                end = "-" if row["rss_end_mib"] is None else row["rss_end_mib"]
                peak = "-" if row["peak_mib"] is None else row["peak_mib"]
                text += f"{end:>10}{peak:>10}"
            lines.append(text)
        total_text = f"Total run time {sum(r['seconds'] for r in rows):.1f} s"
        if with_memory:
            total_text += f", overall peak {max(r['peak_mib'] for r in rows if r['peak_mib'] is not None)} MiB"
        lines.append(total_text)
        log.info("\n".join(lines))


# --- the run's profile, used through these functions ------------------------------------------------------------------
_profile: Optional[RunProfile] = None


def start(name: str = "Setup") -> None:
    global _profile
    _profile = RunProfile()
    _profile.start(name)


def mark(name: str) -> None:
    if _profile is not None:
        _profile.mark(name)


def finish() -> None:
    if _profile is not None:
        _profile.finish()


def reset() -> None:
    global _profile
    if _profile is not None:
        _profile.finish()
    _profile = None
