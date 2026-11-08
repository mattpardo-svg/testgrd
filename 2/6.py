"""sensitivity_table.py - the N-X sensitivities of one contingency, {(branch, variable): value}, kept in a
numpy array instead of a dict of tuples.

A dict entry costs about 150 bytes (key tuple, float, hash slot); the array costs 8. With C contingencies, N
monitored branches and V variables the dict form needs C x N x V x 150 bytes, several times over (raw result,
loading filter, magnitude pruning, last valid set), so a pan-European run does not fit in memory. The table is an
immutable Mapping with the same keys, values and iteration order as the dict it replaces (branch by branch, within a
branch the variables in matrix order); the two filters of nx_sensitivity return new tables that share the array, a
row selection and an optional mask of the entries that were pruned.
"""

from __future__ import annotations

from collections.abc import ItemsView, Mapping, ValuesView
from typing import Callable, Iterable, Optional

import numpy as np

_CHUNK_ROWS = 256


class _TableItems(ItemsView):
    def __iter__(self):
        return self._mapping._iter_items()


class _TableValues(ValuesView):
    def __iter__(self):
        return (value for _key, value in self._mapping._iter_items())


class SensitivityTable(Mapping):
    """Immutable {(branch, variable): float}. labels: the branch of each row of `base`; variables: the variable of
    each column; rows: the rows of `base` that belong to the table (None = all), in order; mask: bool array of the
    shape (len(rows), len(variables)), True = the entry is in the table (None = all)."""

    __slots__ = ("_labels", "_variables", "_base", "_rows", "_mask", "_row_of", "_col_of", "_size")

    def __init__(self, labels, variables, base, rows=None, mask=None, col_of=None):
        self._labels = labels
        self._variables = variables
        self._base = base
        self._rows = rows
        self._mask = mask
        base_rows = range(len(labels)) if rows is None else rows.tolist()
        self._row_of = {labels[base_row]: position for position, base_row in enumerate(base_rows)}
        self._col_of = col_of if col_of is not None else {variable: column for column, variable in enumerate(variables)}
        self._size = int(mask.sum()) if mask is not None else len(self._row_of) * len(variables)

    # ------------------------------------------------------------------ Mapping
    def _base_row(self, position: int) -> int:
        return position if self._rows is None else int(self._rows[position])

    def get(self, key, default=None):
        try:
            branch, variable = key
            position = self._row_of.get(branch)
            column = self._col_of.get(variable)
        except (TypeError, ValueError):
            return default
        if position is None or column is None:
            return default
        if self._mask is not None and not self._mask.item(position, column):
            return default
        return self._base.item(self._base_row(position), column)

    def __getitem__(self, key):
        value = self.get(key, _MISSING)
        if value is _MISSING:
            raise KeyError(key)
        return value

    def __contains__(self, key):
        return self.get(key, _MISSING) is not _MISSING

    def __len__(self):
        return self._size

    def _iter_items(self):
        variables = self._variables
        rows = range(len(self._row_of)) if self._rows is None else self._rows.tolist()
        mask = self._mask
        for position, base_row in enumerate(rows):
            label = self._labels[base_row]
            values = self._base[base_row].tolist()
            if mask is None:
                for variable, value in zip(variables, values):
                    yield (label, variable), value
            else:
                for variable, value, present in zip(variables, values, mask[position].tolist()):
                    if present:
                        yield (label, variable), value

    def __iter__(self):
        return (key for key, _value in self._iter_items())

    def items(self):
        return _TableItems(self)

    def values(self):
        return _TableValues(self)

    def __repr__(self):
        return f"SensitivityTable({len(self._row_of)} branches x {len(self._variables)} variables, {self._size} entries)"

    # ------------------------------------------------------------------ views and checks
    @property
    def branches(self) -> list:
        """The branches of the table in row order."""
        rows = range(len(self._row_of)) if self._rows is None else self._rows.tolist()
        return [self._labels[base_row] for base_row in rows]

    def _chunks(self):
        """(first position, values block, mask block or None) of at most _CHUNK_ROWS rows."""
        count = len(self._row_of)
        for start in range(0, count, _CHUNK_ROWS):
            stop = min(start + _CHUNK_ROWS, count)
            block = self._base[start:stop] if self._rows is None else self._base[self._rows[start:stop]]
            yield start, block, None if self._mask is None else self._mask[start:stop]

    def all_zero(self) -> bool:
        """True when every entry of the table is exactly 0.0 (NaN is not 0.0); True for an empty table."""
        for _start, block, mask in self._chunks():
            if np.any((block if mask is None else block[mask]) != 0.0):
                return False
        return True

    def nan_branches(self) -> set:
        """The branches that have at least one NaN entry."""
        labels = self.branches
        found = set()
        for start, block, mask in self._chunks():
            is_nan = np.isnan(block)
            if mask is not None:
                is_nan &= mask
            for offset in np.flatnonzero(is_nan.any(axis=1)).tolist():
                found.add(labels[start + offset])
        return found

    # ------------------------------------------------------------------ filters (new tables, same array)
    def restricted_to_branches(self, keep: Callable[[str], bool]) -> "SensitivityTable":
        """The rows whose branch passes `keep`, in the same order."""
        labels = self.branches
        selected = np.array([position for position, label in enumerate(labels) if keep(label)], dtype=np.intp)
        if len(selected) == len(labels):
            return self
        rows = selected if self._rows is None else self._rows[selected]
        mask = None if self._mask is None else self._mask[selected]
        return SensitivityTable(self._labels, self._variables, self._base, rows, mask, self._col_of)

    def without_entries_below(
        self, threshold_by_branch: Callable[[str], float], exempt: Optional[Callable[[str], bool]] = None,
        scale_by_variable: Optional[Callable[[str], float]] = None,
    ) -> "SensitivityTable":
        """The table without the entries whose absolute value is below the threshold of their branch (NaN is never
        below: it stays, as with abs(value) < threshold on a float). The variables for which `exempt` is true are kept.
        scale_by_variable: the value is first multiplied by the scale of its variable (degrees per tap: the effect of
        one tap); a variable with a NaN scale is kept."""
        labels = self.branches
        thresholds = np.array([threshold_by_branch(label) for label in labels], dtype=np.float64)
        scales = None if scale_by_variable is None else np.abs(
            np.array([scale_by_variable(variable) for variable in self._variables], dtype=np.float64))
        pruned = np.zeros((len(labels), len(self._variables)), dtype=bool)
        for start, block, _mask in self._chunks():
            effect = np.abs(block) if scales is None else np.abs(block) * scales[None, :]
            pruned[start:start + len(block)] = effect < thresholds[start:start + len(block), None]
        if exempt is not None:
            pruned[:, [column for column, variable in enumerate(self._variables) if exempt(variable)]] = False
        if not pruned.any():
            return self
        mask = ~pruned if self._mask is None else (~pruned) & self._mask
        return SensitivityTable(self._labels, self._variables, self._base, self._rows, mask, self._col_of)


    def compacted(self) -> "SensitivityTable":
        """The same entries in a new, smaller array: only the rows and columns that still have an entry are copied, so
        the full array of the contingency can be freed. Order of rows and columns is kept; an entry-less branch is no
        longer a row (it has no key either way)."""
        mask = self._mask
        if mask is None:
            return self
        keep_rows, keep_columns = mask.any(axis=1), mask.any(axis=0)
        if keep_rows.all() and keep_columns.all():
            return self
        base = self._base if self._rows is None else self._base[self._rows]
        labels = self.branches
        return SensitivityTable(
            tuple(label for label, keep in zip(labels, keep_rows.tolist()) if keep),
            tuple(variable for variable, keep in zip(self._variables, keep_columns.tolist()) if keep),
            np.ascontiguousarray(base[keep_rows][:, keep_columns]),
            mask=np.ascontiguousarray(mask[keep_rows][:, keep_columns]),
        )


_MISSING = object()


def table_from_matrices(blocks: Iterable[tuple[list, list, np.ndarray]]) -> Optional[SensitivityTable]:
    """One table from the sensitivity matrices of one contingency, one block per side as pypowsybl returns them
    (branch ids, variable ids, array of the shape (variables, branches)), in the order the dict of the old
    reader was filled. None when the blocks cannot be one table with the same keys and order as that dict (a
    duplicated id, a branch on two sides, different variables): the caller then builds the dict."""
    blocks = list(blocks)
    if not blocks:
        return SensitivityTable((), (), np.zeros((0, 0)))
    variables = list(blocks[0][1])
    labels: list = []
    for branch_ids, variable_ids, values in blocks:
        if list(variable_ids) != variables or len(set(variable_ids)) != len(variable_ids):
            return None
        if len(set(branch_ids)) != len(branch_ids) or values.shape != (len(variable_ids), len(branch_ids)):
            return None
        labels.extend(branch_ids)
    if len(set(labels)) != len(labels):
        return None
    base = np.ascontiguousarray(
        np.concatenate([np.asarray(values, dtype=np.float64).T for _b, _v, values in blocks], axis=0)
    )
    return SensitivityTable(tuple(labels), tuple(variables), base)


def copy_of(sensitivities):
    """An independent copy of a {(branch, variable): value} mapping: a table is immutable and is returned as it is,
    any other mapping is copied into a dict."""
    return sensitivities if isinstance(sensitivities, SensitivityTable) else dict(sensitivities)
