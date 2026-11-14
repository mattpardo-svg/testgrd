READ-ONLY, same rules as before (text search only, no pipeline, no CGM, counts and property names only, never ids, names or values).
For the PowerBidSchedule elements that reference the RD RAs whose operator has no files of its own (the 11 found before), report counts:
1. with a direction property (which property name, and the count per last word of its value)
2. whose name contains "up" or "down" anywhere, case-insensitive (count per pattern: suffix, prefix, middle; and the separator used, e.g. "_", "-", " ")
3. by sign of their p values at the time points: all positive, all negative, mixed
4. any other property on these bids or their time points not present on bids that do have a direction (property names only)
Reply with a short table. Nothing else.
