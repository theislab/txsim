import pandas as pd
from anndata import AnnData
from typing import Union, Tuple


def mean_proportion_deviation(
        adata_sp: AnnData,
        adata_sc: AnnData,
        ct_set: str = "union",
        obs_key: str = "celltype",
        pipeline_output: bool = True
) -> Union[float, Tuple[float, pd.DataFrame]]:
    """Calculate the mean difference in proportions between cell types from both datasets.

    Parameters
    ----------
    adata_sp : AnnData
        Annotated ``AnnData`` object with counts from spatial data.
    adata_sc : AnnData
        Annotated ``AnnData`` object with counts from scRNAseq data.
    ct_set : str, default "union"
        Text string to determine which (sub)set of cell types to compare.
        Supported: ["union", "intersection", "sp_specific", "sc_specific"]
    obs_key : str, default "celltype"
        Key in adata_sp.obs for the cell type.
    pipeline_output : bool, default True
        Boolean that when set to ``False`` will return the DataFrame with cell type proportions for further analysis.

    Returns
    -------
    proportion_metric : float
        If ``ct_how`` is set to "union" (default) or "intersection":
            Score as 1 - mean absolute difference between the cell type proportions from both data sets.
            Values close to 1 indicate good consistency in cell type proportions.
            Values close to 0 indicate inconsistency between cell type proportions.
        If ``ct_how`` is set to "sp_specific" or "sc_specific":
            Proportion of cells in one data set that are assigned to a cell type not present in the other data set.
    df_props : pd.DataFrame, optional
        If ``pipeline_output`` is set to ``False``:
            DataFrame with proportions of each cell type, in which data set they occurred, and their difference.
    """

    # determine proportion of each cell type in each modality
    ct_props_sp = adata_sp.obs[obs_key].value_counts(normalize=True).rename("proportion_sp")
    ct_props_sc = adata_sc.obs[obs_key].value_counts(normalize=True).rename("proportion_sc")

    # merge cell type proportions from modalities together based on ct_how parameter
    merge_how = {"union": "outer", "intersection": "inner", "sp_specific": "left", "sc_specific": "right"}
    df_props = pd.merge(ct_props_sp, ct_props_sc,
                        how=merge_how[ct_set], left_index=True, right_index=True, indicator="ct_in")
    df_props["ct_in"] = df_props["ct_in"].cat.rename_categories({"left_only": "sp", "right_only": "sc"})

    # calculate difference (sp - sc), with NaN from merging assumed to be 0
    df_props["sp_minus_sc"] = df_props["proportion_sp"] - df_props["proportion_sc"]
    df_props.loc[df_props["ct_in"] == "sp", "sp_minus_sc"] = df_props[df_props["ct_in"] == "sp"]["proportion_sp"]
    df_props.loc[df_props["ct_in"] == "sc", "sp_minus_sc"] = -df_props[df_props["ct_in"] == "sc"]["proportion_sc"]

    proportion_metric = 0
    match ct_set:
        case "union" | "intersection":
            if df_props.shape[0] > 0:
                proportion_metric = 1 - df_props["sp_minus_sc"].abs().mean()
        case "sp_specific":
            if "sp" in df_props["ct_in"].values:
                proportion_metric = df_props[df_props["ct_in"] == "sp"]["proportion_sp"].sum()
            df_props = df_props[df_props["ct_in"] == "sp"]
        case "sc_specific":
            if "sc" in df_props["ct_in"].values:
                proportion_metric = df_props[df_props["ct_in"] == "sc"]["proportion_sc"].sum()
            df_props = df_props[df_props["ct_in"] == "sc"]

    if pipeline_output:
        return proportion_metric
    else:
        return proportion_metric, df_props
