# Corrected data attribution and provenance

The cited upstream dataset is Daniel Mulat (2025), *Process monitoring and weather
datasets on farm-scale biogas digesters*, Mendeley Data, version 1,
DOI https://doi.org/10.17632/gk3f363sfg.1. The publisher's dataset page
https://data.mendeley.com/datasets/gk3f363sfg/1 lists CC BY 4.0, verified on
19 October 2026. Original upstream materials remain subject to that license:
https://creativecommons.org/licenses/by/4.0/.

The supplied workbook is an **author-confirmed corrected derivative** of that
dataset. On 19 October 2026, the author confirmed that erroneous original
entries had been corrected and instructed the revision to use the current
corrected workbook as the sole numerical input. It is not a byte-identical
upstream release. Independent field correction logs were not supplied; the
correction rationale is author-attested. No further measurements were changed
while preparing or executing this reproduction package.

Corrected workbook SHA-256:
`1b061d2b93ebdb4f1f7cc3c1f7934bb4fbe7bb32149b2f8ecaf18ad80c3554cd`.
The exact copied input and code hashes are listed in `checksums.sha256` and
`evidence/snapshot_manifest.json`. The current core title and description are
read directly from the workbook and recorded by the analysis; historical
hard-coded metadata text is not substituted for those fields.

`data/weather_daily.csv` is the archived daily weather matrix used by the
analysis. This package reproduces the model from that matrix; it does not
reconstruct the matrix from an original weather workbook. Both data files are
fixed by checksum. This deliverable contains no original uncorrected workbook
or cell-by-cell list of replaced historical values.

Successful reproduction verifies the numerical calculations conditional on
these identified corrected inputs. It does not independently verify field
measurements, prospective predictor availability or external model validity.

## Meter-reading rounding update

At the author’s request, 159 fractional cumulative meter readings in column G
of the four reactor sheets in `data/Farm-scale_Biodigester.xlsx` were rounded
to the nearest integer (half-up). Counts by sheet are 27, 32, 68 and 32.
Only those numeric cell values were changed; other workbook entries, formulas,
and cached formula values were preserved. This is a numerical transformation,
not verification against field records. The historical provenance statements
remain applicable.

Existing `results/`, reference results and execution evidence predate this
rounding update. The modelling pipeline and dependent workbook formula caches
have NOT been recalculated for the rounded input; existing numerical results
must not be described as verified outputs of this updated workbook.
