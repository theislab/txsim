from anndata import AnnData
import numpy as np
import pandas as pd

def all_quality_metrics(
    spatial_data: AnnData,
) -> pd.DataFrame:

   
    #Generate metrics
    metrics = {}
  
    return pd.DataFrame.from_dict(metrics, orient='index')

