"""
operators.py - the TSO of an assessed element or a remedial action, taken from its operator in the content
(AssessedElement.AssessedSystemOperator, RemedialAction.RemedialActionSystemOperator):
  - an EIC URI: its last path segment, whatever the base (the name of the files' TSO when a file names that EIC as its
    creator, so one TSO has one name);
  - a local id: the name of that TSO in the Organisation (CD) file when there is one and it knows the id;
  - missing, or a local id nothing resolves: the TSO of the file the object is in (its header creator, as before); a local
    id nothing resolves is counted and logged once per file (count, first 5 ids).
"""
from __future__ import annotations

import logging
from typing import Optional

from rdflib import Graph, Namespace

from pre_milp import log_summary

log = logging.getLogger(__name__)

NC = Namespace("https://cim4.eu/ns/nc#")


class OperatorDirectory:
    """name_by_id: {TSO id: name} of the Organisation file (empty without one); label_by_eic: {creator EIC: TSO name of the
    files with that creator}. eic_by_label is filled with the EIC of every TSO named by an EIC URI."""

    def __init__(self, name_by_id: Optional[dict] = None, label_by_eic: Optional[dict] = None, has_organisation: bool = False):
        self.name_by_id = dict(name_by_id or {})
        self.label_by_eic = dict(label_by_eic or {})
        self.has_organisation = has_organisation or bool(self.name_by_id)
        self.eic_by_label: dict[str, str] = {label: eic for eic, label in self.label_by_eic.items()}
        self._unresolved: dict[str, list[str]] = {}

    def tso_of(self, graph: Graph, subject, predicate: str, default: str, source_name: str, object_id: str) -> str:
        """The TSO of `subject`: its operator under `predicate`, else `default` (the TSO of its file)."""
        reference = graph.value(subject, NC[predicate])
        if reference is None:
            return default
        text = str(reference)
        if "#" in text:                                             # a local id: the Organisation file resolves it
            local_id = text.rsplit("#", 1)[-1].lstrip("_")
            name = self.name_by_id.get(local_id)
            if name:
                return name
            self._unresolved.setdefault(source_name, []).append(object_id)
            return default
        eic = text.rstrip("/").rsplit("/", 1)[-1]                   # an EIC URI: the last path segment
        label = self.label_by_eic.get(eic, eic)
        self.eic_by_label.setdefault(label, eic)
        return label

    def log_unresolved(self) -> None:
        """One warning per file whose objects name an operator by a local id that nothing resolves."""
        for source_name, ids in sorted(self._unresolved.items()):
            reason = "the Organisation file does not know" if self.has_organisation else "there is no Organisation file to resolve"
            log_summary.warn_ids(
                log, f"{source_name}: %d object(s) name their operator by a local id that {reason} - the TSO of the file "
                "(its header creator) is used", ids,
            )
        self._unresolved.clear()
