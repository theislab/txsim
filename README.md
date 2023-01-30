# txsim
Python package to measure the similarity between matched single cell and targeted spatial transcriptomics data

# Installation
## Cloning and adding
In a clean `conda` environment with pip installed, run in the terminal:

```git clone https://github.com/theislab/txsim.git```

Navigate to the folder:

```cd txsim```

And install using `pip`:

```pip install -e .```

## Requirements
To import `txsim`, install `squidpy` (and all of its dependencies) into your environment
For full functionality, the following are required as well:
- `alphashape`
- `descartes`
- `pciSeq`
- `cellpose`

# Using Functions
All of the functions in txsim are currently either in the `metrics` or `preprocessing` module. 
A list of functions is as follows:

## Metrics
- `coexpression_similarity`
- `coexpression_similarity_celltype`
- `all_metrics`
## Preprocessing
- Normalization: `normalize_total`, `normalize_pearson_residuals`, `normalize_by_area`
- Segmentation: `segment_nuclei`, `segment_cellpose`
- Assignment: `basic_assign`, `run_pciSeq`
- Count Generation: `generate_adata`, `calculate_alpha_area`