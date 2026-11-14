READ-ONLY CHECK on the NCP files. A pipeline run is in progress on this machine: do NOT run the pipeline, do NOT load the CGM, do NOT run Python on large files, do NOT change any file. Use only text search (Select-String / findstr) on the extracted NCP files in output\_extracted_ncp (or extract the NCP zip to a temporary folder outside output\). Report numbers and class/property names only: never print ids, names, EIC codes or file contents.

A. In the AE file whose header creator is the French TSO, count:
 1. elements with inBaseCase false
 2. AssessedElement.SecuredForRegion
 3. AssessedElement.ScannedForRegion
 4. AssessedElement.exclusionReason (and the count per distinct value of its last word, e.g. systemOperator, capacityCalculationRegion)
 5. elements with inBaseCase false that have NO SecuredForRegion and NO ScannedForRegion
 6. elements with inBaseCase false that have an exclusionReason

B. For the RD RAs whose operator is a TSO that has no files of its own (RemedialAction.RemedialActionSystemOperator pointing to an EIC that is not any file's dcterms:creator), answer yes/no with counts:
 1. Do they have both rdf:ID and IdentifiedObject.mRID, and do the two differ (other than a leading "_")?
 2. In all SIS files: how many PowerBidSchedule elements reference these RAs, matching by rdf:ID fragment, and how many by mRID?
 3. Are their legacy shift key links (GLSKBidActionDistribution) in a different SIS file than the GLSKSchedule they point to?
 4. Do any PowerShiftKeyStrategy elements reference these RAs, and are they in a different RA file than the RA?

Reply with a short table of counts. Nothing else.
