# FRB20220529 Lomb-Scargle Analysis

This repository provides a small script to compute Lomb-Scargle periodograms for
the DM and RM columns in `FRB20220529_TableS5.csv`, both without errors and with
measurement uncertainties.

## Usage

```bash
python lsp_analysis.py FRB20220529_TableS5.csv --output-dir outputs
```

The script writes two plots:

- `outputs/dm_lsp.png`
- `outputs/rm_lsp.png`

