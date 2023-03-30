# Solvent-MLP

![overview figure](TOC.svg)

## Use of Conda and Dependencies

If one wishes to replicate the MLP training or plotting, a Conda enviornment is provided and can be used by the comand:
```bash
conda env create -f solvent.yml
```
Scripts can then be run in Python once Conda has been downloaded and the solvent enviornment has been created.

## Training

If one wishes to replicate the k-fold training of MLPs, a script is provided for such. This should be run from the root directory from this repo as:
```bash
python training-scripts/k-fold-training.py
```

## Plotting

If one wishes to replicate plotting results they should again be run from the root directory.
