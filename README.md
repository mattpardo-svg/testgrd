READ-ONLY CHECK, same rules as before: text search only (Select-String / findstr), no pipeline, no CGM loading with pypowsybl, no Python on large files, no file changes. Report counts and CIM class/property names only: never ids, names, EIC codes or values.

Scope: the NCP files whose names contain ELIA (AE, CO, RA) and the ELIA EQ file inside the CGM (and, for lookups, all EQ files of the CGM).

A. Assessed elements (ELIA AE file): for each reference property used (e.g. AssessedElement.ConductingEquipment, .OperationalLimit, .Line, others):
 1. count of references
 2. how many referenced ids are found in the ELIA EQ / in any EQ file, as rdf:ID or rdf:about, exact match
 3. how many are found only after normalisation (case-insensitive, leading "_" removed, "#" removed, "urn:uuid:" removed)
 4. for the found ones: count per CIM class of the target (ACLineSegment, PowerTransformer with 2 ends, PowerTransformer with 3 ends, SeriesCompensator, BusbarSection, Switch, OperationalLimitSet, CurrentLimit, Terminal, other: name it)
 5. not found anywhere: count

B. Contingencies (ELIA CO file): same as A for ContingencyElement / ContingencyEquipment references (the property that points to equipment): counts found exact, found after normalisation, per target class, not found.

C. PST actions (ELIA RA file): for TapPositionAction.TapChanger references: count per target class (PhaseTapChangerAsymmetrical / Symmetrical / Linear / Tabular, RatioTapChanger, PowerTransformer, other), and not found.

D. Header of the ELIA EQ file: the md:Model.profile value(s) and md:Model.modelingAuthoritySet (only whether it differs from the other EQ files' pattern; no value), and whether the ELIA AE/CO/RA files reference ids that appear in any EQ file at all (yes/no).

Reply with short tables. Nothing else.
