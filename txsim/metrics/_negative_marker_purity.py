import scanpy as sc
import numpy as np
import pandas as pd
from anndata import AnnData

def negative_marker_purity(adata_sp: AnnData, adata_sc: AnnData,pipeline_output:bool=True):
    """ Negative marker purity aims to measure read leakeage between cells in spatial datasets. 
    For this, we calculate the increase in reads assigned in spatial datasets to pairs of genes-celltyes with no/very low expression in scRNAseq
    Parameters
    ----------
    adata_sp : AnnData
        annotated ``AnnData`` object with counts from spatial data
    adata_sc : AnnData
        annotated ``AnnData`` object with counts scRNAseq data
    pipeline_output : float, optional
        Boolean for whether to use the function in the pipeline or not
    Returns
    -------
    negative marker purity : float
       Increase in proportion of reads assigned in spatial data to pairs of genes-celltyes with no/very low expression in scRNAseq
    """   
    
    key='celltype'
    min_number_cells=10 #minimum number of cells belonging to a cluster to consider it in the analysis
    minimum_exp=0.005 #maximum relative expression allowed in a gene in a cluster to consider the gene-celltype pair the analysis 
    adata_sc = adata_sc[:,adata_sp.var_names]
    unique_celltypes=adata_sc.obs.loc[adata_sc.obs[key].isin(adata_sp.obs[key]),key].unique()
    adata_sc.obs['index']=adata_sc.obs.index
    present_celltypes=list(adata_sc.obs.groupby('celltype').count().loc[adata_sc.obs.groupby('celltype').count().iloc[:,0]>min_number_cells,:].index)
    unique_celltypes=list(set(unique_celltypes).intersection(present_celltypes))
    genes=adata_sc.var.index[adata_sc.var.index.isin(adata_sp.var.index)]
    exp_sc=pd.DataFrame(adata_sc.layers['raw'],columns=adata_sc.var.index)
    gene_means_sc=pd.DataFrame(np.mean(exp_sc,axis=0))
    gene_means_sc=gene_means_sc.loc[gene_means_sc.index.sort_values(),:]
    exp_sp=pd.DataFrame(adata_sp.layers['raw'],columns=adata_sp.var.index)
    gene_means_sp=pd.DataFrame(np.mean(exp_sp,axis=0))
    gene_means_sp=gene_means_sp.loc[gene_means_sp.index.sort_values(),:]
    exp_sc['celltype']=list(adata_sc.obs['celltype'])
    exp_sp['celltype']=list(adata_sp.obs['celltype'])
    exp_sc=exp_sc.loc[exp_sc['celltype'].isin(unique_celltypes),:]
    exp_sp=exp_sp.loc[exp_sp['celltype'].isin(unique_celltypes),:]
    mean_celltype_sp=exp_sp.groupby('celltype').mean()
    mean_celltype_sc=exp_sc.groupby('celltype').mean()
    mean_ct_sc_norm=mean_celltype_sc.div(mean_celltype_sc.sum(axis=0),axis=1)
    mean_ct_sp_norm=mean_celltype_sp.div(mean_celltype_sp.sum(axis=0),axis=1)
    mean_ct_sp_norm=mean_ct_sp_norm.loc[:,mean_ct_sc_norm.columns]
    lowvals_sc=np.array(mean_ct_sc_norm)[np.array(mean_ct_sc_norm<minimum_exp)]
    lowvals_sc_filt=[x for x in lowvals_sc if str(x) != 'nan']
    mean_sc_lowexp=np.mean(lowvals_sc_filt)
    lowvals_sp=np.array(mean_ct_sp_norm)[np.array(mean_ct_sc_norm<minimum_exp)]
    lowvals_sp_filt=[x for x in lowvals_sp if str(x) != 'nan']
    mean_sp_lowexp=np.mean(lowvals_sp_filt)
    negative_marker_purity=1-(mean_sp_lowexp-mean_sc_lowexp)
    if pipeline_output==True:
        return negative_marker_purity
    else:
        nmp_genes=[]
        nmp_values=[]
        for g in genes_with_noexp:
            sels=mean_ct_sc_norm.loc[:,g]<minimum_exp
            exi=mean_ct_sp_norm.loc[sels,g]-mean_ct_sc_norm.loc[sels,g]
            exi=[x for x in exi if str(x) != 'nan']
            if len(exi)>0:
                nmp_genes.append(g)
                nmp_values.append(np.mean(exi))
        nmp_by_gene=pd.DataFrame([nmp_values],columns=[nmp_genes],index=['negative_marker_purity']).transpose()
        return negative_marker_purity,nmp_by_gene
    
