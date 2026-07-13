# archive/

Old scripts kept for reference. These are no longer used in the main pipeline.

## Contents

- train_combined.py      — combined rho+cs2 two-output model training (abandoned, liquid MAPE 0.38%)
- validate_combined.py   — validation for combined model
- export_combined.py     — weight export for combined model
- train_curves.py        — spinodal/saturation NN model training (replaced by Fortran lookup tables)
- validate_curves.py     — curve model validation
- export_curves.py       — curve model weight export
- plot_curves.py         — curve visualization
- timing_comparison.py   — old timing script
- datasets/              — old Python dataset generation (replaced by Fortran generator)
  - generate_dataset.py
  - generate_curves.py
  - visual_dataset.py
  - sampling.py
  - phase_boundaries.py
- src/scaling_curves.py  — old curve scaling (replaced)

## Why Archived

The combined model was abandoned because liquid MAPE (0.38%) was unacceptable.
The curve models (spinodal, saturation) were replaced by direct CSV lookup tables
in Fortran, which are faster and more accurate for smooth 1D curves.
The Python dataset generator was replaced by a Fortran generator that uses
Thermopack 3.0 directly, ensuring consistency with the Fortran inference library.
