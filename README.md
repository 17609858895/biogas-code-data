# Biogas revision 2: corrected-data reproduction package

This reproduction package reproduces the fixed corrected-data analysis accompanying the
second revision. The author confirmed on 19 October 2026 that the current workbook
contains corrections to erroneous original entries and instructed the revision
to use it as the sole meter-record input, alongside the supplied matched weather
file. See `DATA_PROVENANCE.md`. Independent
field correction logs were not supplied. Successful execution verifies the
calculations conditional on the identified corrected inputs.

## Environment

The execution environment tested here is Windows, Python **3.9.21**. The original
13 direct dependencies are pinned in `requirements.txt`. All installed versions
needed by those packages are pinned in `requirements.lock.txt`. The complete
observed environment is recorded in `requirements-full-environment.txt`; that
larger list also contains unrelated packages and is an audit snapshot.

```powershell
conda env create -f environment.yml
conda activate biogas-revision2
python run.py --preflight-only
```

`preflight-only` checks the Python version, all direct dependency versions and
SHA-256 hashes of the copied data and code. It writes `evidence/preflight.json`.
The required inputs are `data/Farm-scale_Biodigester.xlsx` and
`data/weather_daily.csv`. No model cache, Microsoft Office installation, online
data service or external file path is required by the analysis.

## Run order

From this directory, the following command starts from the copied input files
and generates `results/`:

```powershell
python run.py
```

`run.py` calls the checked `reproduce.py` entry point. The preserved baseline
runner `run_baseline.py` executes these scripts sequentially:

1. `code/run_reanalysis.py`: reconstruct accepted rates and record-level inputs;
   create temporal partitions; fit nine fixed candidates; calculate rolling-origin
   and validation results; refit the validation-selected RF; generate diagnostics,
   baseline comparisons, permutation importance and SHAP outputs.
2. `code/data_all_models.py`: refit all candidates on development records; create
   common evaluation predictions and descriptive model/baseline/TOPSIS rankings.
3. `code/uncertainty_analysis.py`: generate 1000 date-cluster bootstrap replicates,
   interval summaries and sensitivity-scenario predictions.
4. `code/export_tables.py`: export full-precision manuscript Table 1 and Tables
   S1–S7 into `results/publication_tables/`.
5. `code/refine_figures.py`: render Figs. 1–8 and their individual panels.
6. `code/ale_analysis.py`: generate the archived post-hoc ALE data and Fig. 9.

The recommended `reproduce.py` then runs `code/revision2_selection.py`,
`code/verify_revision2.py` and `code/export_revision2_tables.py`. They generate and independently verify the added
development-only nested rolling-origin analysis, fix the ALE variable choice
using training/validation data only, and produce the revised main Fig. 9 plus
the post-hoc comparison Fig. S1. Their numerical outputs are written to
`results/revision2/`. `--figures-only` instead runs
`code/render_revision2_ale.py` on those saved curves. The revision2 analysis
also renders its two ALE figures when called with `--no-figures`; that option
suppresses the baseline figure suite only.

The default command then checks all regenerated tables and figure files with
`verify_outputs.py`. The baseline and revision2 execution logs supplied in
`evidence/` record successful runs of these stages, including the added table
exports and independent nested-prediction check.

For numerical outputs only, add `--no-figures`. After numerical analysis, add
`--figures-only` to render the figures from the newly generated result files.
The original baseline runner is retained as `run_baseline.py`; `reproduce.py` is
the recommended entry point because it checks the environment and input hashes.

## Outputs and checks

| Publication item | Generated file under `results/` |
|---|---|
| Table 1 | `publication_tables/Table_1.csv` |
| Tables S1–S7 | `publication_tables/Table_S1.csv` … `Table_S7.csv` |
| Added Table 2 | `publication_tables/Table_2.csv`, copied from the explicitly curated literature source in `data/` |
| Added Table S8 | `publication_tables/Table_S8.csv`, copied from the explicitly curated issue-time audit in `data/` |
| Added Tables S9–S11 | `publication_tables/Table_S9.csv` … `Table_S11.csv`, calculated from nested/ALE outputs |
| Fig. 1 | `figures/Fig01_locked_split.{png,pdf}` |
| Fig. 2 | `figures/Fig02_association_analysis.{png,pdf}` |
| Fig. 3 | `figures/Fig03_validation_selection.{png,pdf}` |
| Fig. 4 | `figures/Fig04_test_prediction.{png,pdf}` |
| Fig. 5 | `figures/Fig05_residual_reliability.{png,pdf}` |
| Fig. 6 | `figures/Fig06_calibration_coverage.{png,pdf}` |
| Fig. 7 | `figures/Fig07_frozen_model_importance.{png,pdf}` |
| Fig. 8 | `figures/Fig08_sensitivity.{png,pdf}` |
| Archived Fig. 9 | `figures/Fig09_ALE_reproduction/Fig09_ALE.{png,pdf}` |
| Revised main Fig. 9 | `revision2/figures/Fig09_ALE_validation_predefined/Fig09_ALE_validation_predefined.{png,pdf}` |
| New Fig. S1 | `revision2/figures/FigS1_ALE_posthoc/FigS1_ALE_posthoc.{png,pdf}` |
| Added nested selection data | `revision2/nested_outer_metrics.csv`, `nested_pooled_summary.csv`, `nested_inner_selection.csv` |
| Added ALE selection/support data | `revision2/validation_only_RF_permutation_importance.csv`, `ALE_summary.csv`, `ALE_bin_support.csv`, `ALE_curves.csv` |

`reference_results/` is an unchanged archive for comparison, never an input to
the full pipeline. `verify_outputs.py` compares regenerated CSV cells (relative
tolerance 1e-9, absolute tolerance 1e-8), checks all nine PNG/PDF pairs and records
pixel comparisons against archived figures. PDF timestamps and binary model
serialization are not expected to be byte-identical. The figure map and result
are stored in `evidence/regeneration_verification.json`.

`evidence/full_run.log`, `evidence/run_status.json` and
`evidence/regeneration_verification.json` record the observed execution outcome;
do not infer success from this README alone. `checksums.sha256` contains file
checksums and is accompanied by `verify_checksums.py`. Run
`python verify_checksums.py` to verify the static package; add `--generated` to
verify the exact recorded output-file bytes. PDF metadata and gzip timestamps
can change on a new run; numerical and pixel checks assess their content.

The baseline comparison covers the submitted September analysis and its nine
figures. `reference_revision2/` holds the independently produced revision2
numeric and figure outputs for checking the portable added analysis. The
development-only selection outputs are verified again from held-out predictions
by `code/verify_revision2.py`; their source and fitting windows are recorded in
`results/revision2/selection_protocol.json`.

## Interpretation and licensing

The analysis is retrospective. Temporal partitions do not establish predictor
availability at a prospective issue time or independent external validation.
RF is the validation-selected reference; LightGBM has lower evaluation RMSE in
these fixed processed inputs. The archived Fig. 9 feature choice is post-hoc and
evaluation-informed. These facts are preserved in the code and audit files.

Project-authored code and documentation use the MIT license (`LICENSE`). The
cited upstream dataset is attributed to Daniel Mulat and listed as CC BY 4.0 by
Mendeley Data; the local corrections and evidence boundary are disclosed
in `DATA_PROVENANCE.md`. Third-party dependencies retain their own licenses.

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
