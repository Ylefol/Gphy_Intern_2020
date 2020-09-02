#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 08:17:18 2020

@author: yohan
"""

import numpy as np
import pyranges as pr
import scanpy as sc
import pandas as pd
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy
import logging as logg
import warnings
import velocyto as vcy
import seaborn as sns
import scipy.stats as stats
import re



from sklearn.neighbors import NearestNeighbors
from numpy_groupies import aggregate

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:90% !important; }</style>"))

import logging
# import sys
# logging.basicConfig(stream=sys.stdout, format='%(asctime)s - %(levelname)s - %(message)s', level=logging.DEBUG)


# Wrap implementation
import rpy2.robjects as robj
from rpy2.robjects.packages import importr

#%%Pre-processing

def filter_loom_barcode(adata,filter_list,new_file_name=None):
    """
    Filters cells based on provided list of barcodes
    Overwrites the AnnData object provided

    Parameters
    ----------
    adata : AnnData object
        The AnnData object to be filtered.
    filter_list : list
        A list of barcodes to filter out from the loom file.
    new_file_name : String, optional
        The name of the new loom file, if left to default, no file will be written. The default is None.

    Returns
    -------
    adata : AnnData object
        The filtered AnnData object.

    """
    
    velo_bc = []
    for i in adata.obs.index: 
        velo_bc.append(i.split(':')[1].rstrip()[:-1])
    
    #Get a list of indices to filter out
    ind_dict = dict((k,i) for i,k in enumerate(velo_bc))
    inter = set(ind_dict).intersection(filter_list)
    bad_indices = [ ind_dict[x] for x in inter ]
    
    #Creates and fills a list of indexes that are to be kept
    good_indices = []
    for i in ind_dict.values():
        if i in bad_indices:
            pass
        else:
            good_indices.append(i)
    
    
    #Creates the list to use in creating a anndata type
    keep_cells=[adata.obs.index[i] for i in good_indices]
    
    #Creates the new anndata by filtering the old anndata with the good_indices
    adata = adata[keep_cells, :]
    
    
    if new_file_name !=None:
        adata.write_loom(new_file_name+".loom", write_obsm_varm=False)
    
    return adata




def create_chrM_list(gtf_path):
    """
    Uses pyranges to extract all mitochondrial genes from a gtf file
    return the list of mitochondrial genes found
    Caution: loading gtf files can be a bit heavy on the RAM

    Parameters
    ----------
    gtf_path : string
        path to the Gene Transfer File.

    Returns
    -------
    MT_genes_list : list
        The list of mitochondrial genes found in the gtf file.

    """
    #Reads the gtf file using PyRanges
    gr=pr.read_gtf(gtf_path)
    
    #Subsets the pyrange file to only include the mitochondrial chromosome
    gr=gr["chrM"]
    
    MT_genes_list=[]
    #Removes the 'MT-' and ensures that the final list contains unique gene_names
    for i in gr.gene_name:
        x=i.split('-')
        if x[1] not in MT_genes_list:
            MT_genes_list.append(x[1])
    
    return MT_genes_list

def get_chrM_list(path):
    """
    Fetches an already generated list of mitochondrial genes

    Parameters
    ----------
    path : string
        path to the file containing mitochondrial gene names.

    Returns
    -------
    MT_list : list
        list of mitochondrial genes.

    """
    with open(path,'r') as f:
        MT_list=list(f.readlines())  
    MT_list = [x.strip() for x in MT_list] 
    return MT_list

def filter_adata_using_min_intron_count(adata,min_intron_count):  
    """
    Filters an AnnData object based on the number of unspliced reads in the
    unspliced layer for specific genes. Overwrites the provided AnnData file
    with it's filtered version

    Parameters
    ----------
    adata : AnnData
        The AnnData or loom file to be filtered.
    min_intron_count : int
        The minimum intron (unspliced) count per gene.

    Returns
    -------
    None.

    """
        
    X=adata.layers["unspliced"]
    
    #Counts everything on the zero axis
    nb_per_gene=np.sum(X>0,axis=0)
    
    #Flattens a matrix into an array
    nb_per_gene=nb_per_gene.A1
    #creates an array of True or False
    gene_subset=nb_per_gene>=min_intron_count
    #Subsets the adata file using the generated gene_subsets array
    adata._inplace_subset_var(gene_subset)
    
def filter_MT_percent(adata,MT_list,MT_percent_threshold):
    """
    Filters cells based on the percent of mitochondrial reads. The AnnData object
    will be overwritten by the resulting filter.
    
    Parameters
    ----------
    adata : AnnData
        The AnnData or loom object to be filtered.
    MT_list : list
        A list of mitochondrial genes, can be generated through the create_chrM_list() function.
    MT_percent_threshold : int
        An integer representing the percent threshold allowed.

    Returns
    -------
    None.

    """
    
    X=adata.X
    total_reads=np.sum(X,axis=1) #Sum of all reads in each cell_ID
    total_reads=total_reads.astype(np.int64) #Converts float to int
    total_reads=total_reads.A1 #Flattens matrix into array
    
    #Check quantity of MT genes in dataset
    #If as the solution is different based on if it is 1, >1 or 0
    MT_found=0
    the_one_found=''
    for gene in adata.var.index:
        if gene in MT_list:
            MT_found=MT_found+1
            the_one_found=gene

    #If only one MT gene was found in the dataset
    if MT_found==1: 
        number_mito_per_cell=adata.obs_vector(the_one_found)
        number_mito_per_cell=number_mito_per_cell.astype(np.int64)
        
    elif MT_found>1:
        #Create an adata subset with only mitochondrial genes
        mito_gene_indicator= np.in1d(adata.var_names,MT_list)
        adata_mito=adata[:,mito_gene_indicator]
        
        #Create an array with only mitochondrial counts per cell
        X_mito=adata_mito.X
        number_mito_per_cell = np.sum(X_mito,axis=1) #sum of all reads for cells
        total_reads=total_reads.astype(np.int64) #Converts float to int
        number_mito_per_cell = number_mito_per_cell.A1 #Flattens matrix into array
    else:
        print("No mitochondrial genes found in data set")
        return()
    
    #Instantiates an empty list to store MT_percentages per cell
    div_list=[]
    #Goes through each array and creates the percentages as necessary
    for idx,val in enumerate(total_reads):
        if number_mito_per_cell[idx] == 0 | val==0:
            div_list.append(0)
        else:
            div_list.append(number_mito_per_cell[idx]/val*100)
            
    #Converts the list of percentages to an array
    mito_percent=np.asarray(div_list)     
        
    #Creates a True/False mask to use in an adata subset
    cell_subset=mito_percent<=MT_percent_threshold        

    #Subsets the adata file using the generated cell_subset array
    adata._inplace_subset_obs(cell_subset)
    
    
    
def filter_based_on_spliced_unspliced_ratio(adata,layer_to_use,min_percent=None,max_percent=None):
    """
    Filters cells based on the percentage of reads of a layer in comparison with the main matrix.
    Only one percent can be inputted at a time.
    The AnnData object provided will be overwritten by the resulting filter

    Parameters
    ----------
    adata : AnnData
        The AnnData or loom object to be filtered.
    layer_to_use : string
        A string indicating the layer on which the filter will be performed.
    min_percent : int, optional
        The minimum percentage of layer reads per cell. The default is None.
    max_percent : int, optional
        The maximum percentage of layer reads per cell. The default is None.

    Returns
    -------
    None.

    """
    
    #Checks that only one of two thresholds has been inputed
    if min_percent is None and max_percent is None:
        print("A minimum or maximum percent threshold must be inputed")
        return()
    if min_percent is not None and max_percent is not None:
        print("Only one type of percent threshold can be inputed")
        return()
    
    #Generates total_reads
    X=adata.X
    total_reads=np.sum(X,axis=1) #Sum of all reads in each cell_ID
    total_reads=total_reads.astype(np.int64) #Converts float to int
    total_reads=total_reads.A1 #Flattens matrix into array

    #Checks that the layer entered exists
    if layer_to_use not in adata.layers.keys():
        print("The layer given was not found")
        return()
    
    #Generates layer reads
    X_layer=adata.layers[layer_to_use]
    total_layer_reads=np.sum(X_layer,axis=1) #Sum of all reads in each cell_ID
    total_layer_reads=total_layer_reads.astype(np.int64) #Converts float to int
    total_layer_reads=total_layer_reads.A1 #Flattens matrix into array
    
    #Creates the ratio of layer reads
    #Instantiates an empty list to store layer percentages per cell
    div_list=[]
    #Goes through each array and creates the percentages as necessary
    for idx,val in enumerate(total_reads):
        if total_layer_reads[idx] == 0 | val==0:
            div_list.append(0)
        else:
            div_list.append(total_layer_reads[idx]/val*100)
            
    #Converts the list of percentages to an array
    layer_percents=np.asarray(div_list)

    #Creates the mask array according to given threshold
    if min_percent is not None:
        layer_subset= layer_percents >= min_percent
    if max_percent is not None:
        layer_subset= layer_percents <= max_percent
    
    #Subsets the adata object
    adata._inplace_subset_obs(layer_subset)
    
    
def scanpy_pp_plots(
    adata,
    MT_list,
    std_plots,
    MT_plots,
    unspliced_plots):
    """
    Creation of some usefull pre-processing plots using scanpy functions

    Parameters
    ----------
    adata : AnnData object
        The AnnData object which will serve as the data for the plots.
    MT_list : list
        list containing mitochondrial genes.
    std_plots : Boolean
        Standard violin plots for n_genes, n_counts and a scatter plot  using both.
    MT_plots : Boolean
        violin plot for percent MT and scatter using percent MT and n_counts.
    unspliced_plots : Boolean
        violin plot for unspliceed and scatter using unspliced and n_counts.

    Returns
    -------
    None.

    """
    
    #Create standard plots
    adata.obs['n_counts'] = adata.X.sum(axis=1).A1
    
    if std_plots is True:
        sc.pl.violin(adata,['n_genes'], jitter=0.4)
        sc.pl.violin(adata,['n_counts'], jitter=0.4)
        sc.pl.scatter(adata,x='n_counts',y='n_genes')
    
    if MT_plots is True: #Create percent_MT plots
    
        MT_bool_list=[] #Bool list of MT_list in adata
        for val in adata.var_names:
            if val in MT_list:
                MT_bool_list.append(True)
            else:
                MT_bool_list.append(False)
        
        adata.obs["percent_MT"] = np.sum(adata[:,MT_bool_list].X,axis=1).A1/np.sum(adata.X,axis=1).A1
        
        sc.pl.violin(adata,['percent_MT'], jitter=0.4)
        sc.pl.scatter(adata,x='n_counts',y='percent_MT')
        
    if unspliced_plots is True:
        adata.obs["unspliced"] = np.sum(adata.layers['unspliced'],axis=1).A1/np.sum(adata.X,axis=1).A1
        sc.pl.violin(adata,['unspliced'], jitter=0.4)
        sc.pl.scatter(adata,x='n_counts',y='unspliced')
        
        

#%%Cell cycle scoring - Scanpy modified
def score_genes_in_layer(
        adata,
        gene_list,
        layer_choice='spliced',
        ctrl_size=50,
        gene_pool=None,
        n_bins=25,
        score_name='score',
        random_state=0,
        copy=False,
        use_raw=False):  # we use the scikit-learn convention of calling the seed "random_state"
    """Score a set of genes [Satija15]_.

    The score is the average expression of a set of genes subtracted with the
    average expression of a reference set of genes. The reference set is
    randomly sampled from the `gene_pool` for each binned expression value.

    This reproduces the approach in Seurat [Satija15]_ and has been implemented
    for Scanpy by Davide Cittaro.
    
    Modifications by Y.Lefol: Added the option of specifying the layer to work on

    Parameters
    ----------
    adata : :class:`~anndata.AnnData`
        The annotated data matrix.
    gene_list : iterable
        The list of gene names used for score calculation.
    layer_choice : `string`
        String containing the layer to be used as a matrix.
    ctrl_size : `int`, optional (default: 50)
        Number of reference genes to be sampled. If `len(gene_list)` is not too
        low, you can set `ctrl_size=len(gene_list)`.
    gene_pool : `list` or `None`, optional (default: `None`)
        Genes for sampling the reference set. Default is all genes.
    n_bins : `int`, optional (default: 25)
        Number of expression level bins for sampling.
    score_name : `str`, optional (default: `'score'`)
        Name of the field to be added in `.obs`.
    random_state : `int`, optional (default: 0)
        The random seed for sampling.
    copy : `bool`, optional (default: `False`)
        Copy `adata` or modify it inplace.
    use_raw : `bool`, optional (default: `False`)
        Use `raw` attribute of `adata` if present.
    Returns
    -------
    Depending on `copy`, returns or updates `adata` with an additional field
    `score_name`.

    Examples
    --------
    See this `notebook <https://github.com/theislab/scanpy_usage/tree/master/180209_cell_cycle>`__.
    """
    
    from scipy.sparse import issparse

    # start = logg.info(f'computing score {score_name!r}')
    adata = adata.copy() if copy else adata

    if random_state:
        np.random.seed(random_state)

    gene_list_in_var = []
    var_names = adata.raw.var_names if use_raw else adata.var_names
    for gene in gene_list:
        if gene in var_names:
            gene_list_in_var.append(gene)
        else:
            logg.warning(f'gene: {gene} is not in adata.var_names and will be ignored')
    gene_list = set(gene_list_in_var[:])

    if not gene_pool:
        gene_pool = list(var_names)
    else:
        gene_pool = [x for x in gene_pool if x in var_names]

    # Trying here to match the Seurat approach in scoring cells.
    # Basically we need to compare genes against random genes in a matched
    # interval of expression.

    _adata = adata.raw if use_raw else adata
    # TODO: this densifies the whole data matrix for `gene_pool`
    if issparse(_adata.layers[layer_choice]):
        obs_avg = pd.Series(
            np.nanmean(
                _adata[:, gene_pool].layers[layer_choice].toarray(), axis=0), index=gene_pool)  # average expression of genes
    else:
        obs_avg = pd.Series(
            np.nanmean(_adata[:, gene_pool].layers[layer_choice], axis=0), index=gene_pool)  # average expression of genes

    obs_avg = obs_avg[np.isfinite(obs_avg)] # Sometimes (and I don't know how) missing data may be there, with nansfor

    n_items = int(np.round(len(obs_avg) / (n_bins - 1)))
    obs_cut = obs_avg.rank(method='min') // n_items
    control_genes = set()

    # now pick `ctrl_size` genes from every cut
    for cut in np.unique(obs_cut.loc[gene_list]):
        r_genes = np.array(obs_cut[obs_cut == cut].index)
        np.random.shuffle(r_genes)
        control_genes.update(set(r_genes[:ctrl_size]))  # uses full r_genes if ctrl_size > len(r_genes)

    # To index, we need a list - indexing implies an order.
    control_genes = list(control_genes - gene_list)
    gene_list = list(gene_list)


    X_list = _adata[:, gene_list].layers[layer_choice]
    if issparse(X_list): X_list = X_list.toarray()
    X_control = _adata[:, control_genes].layers[layer_choice]
    if issparse(X_control): X_control = X_control.toarray()
    X_control = np.nanmean(X_control, axis=1)

    if len(gene_list) == 0:
        # We shouldn't even get here, but just in case
        logg.hint(
            f'could not add \n'
            f'    {score_name!r}, score of gene set (adata.obs)'
        )
        return adata if copy else None
    elif len(gene_list) == 1:
        score = _adata[:, gene_list].layers[layer_choice] - X_control
    else:
        score = np.nanmean(X_list, axis=1) - X_control

    adata.obs[score_name] = pd.Series(np.array(score).ravel(), index=adata.obs_names)

    # logg.info(
    #     '    finished',
    #     time=start,
    #     deep=(
    #         'added\n'
    #         f'    {score_name!r}, score of gene set (adata.obs)'
    #     ),
    # )
    return adata if copy else None



def my_score_genes_cell_cycle_improved(
        adata,
        layer_choice,
        CC_path,
        copy=False,
        **kwargs):
    """Score cell cycle genes.
    Given two lists of genes associated to S phase and G2M phase, calculates
    scores and assigns a cell cycle phase (G1,S or G2M). See 
    :func:`~scanpy.api.score_genes` for more information.

    Parameters
    ----------
    adata : :class:`~anndata.AnnData`
        The annotated data matrix.
    layer_choice : `string`
        String containing the layer to be used as a matrix.
    CC_path : `string'
        String for the path to the file containing the cell cycle genes and their phases
    copy : `bool`, optional (default:`False`)
        DESCRIPTION. The default is False.
    **kwargs : optional keyword arguments
        Are passed to :func:`~scanpy.api.score_genes`. `ctrl_size` is not
        possible, as it's set as `min(len(s_genes), len(g2m_genes))`.

    Returns
    -------
    Depending on `copy`, returns or updates `adata` with the following fields.
    **G1_score** : `adata.obs`, dtype `object`
        The score for G1 phase for each cell.
    **S_score** : `adata.obs`, dtype `object`
        The score for S phase for each cell.
    **G2M_score** : `adata.obs`, dtype `object`
        The score for G2M phase for each cell.
    **phase** : `adata.obs`, dtype `object`
        The cell cycle phase (`S`,`G2M` or `G1`) for each cell/
    
    See also
    -------
    score_genes
    Examples
    -------
    See this `notebook <https://github.com/theislab/scanpy_usage/tree/master/180209_cell_cycle>`__.
    """
    #logg.info('calculating cell cycle phase')    
    df = pd.read_csv(CC_path,delimiter=',')
    s_genes=[]
    g1_genes=[]
    g2m_genes=[]
    
    for idx,val in enumerate(df['gene']):
        phase=df['phase'][idx]
        s_genes.append(val) if phase=='S' else None
        g1_genes.append(val) if phase=='G1' else None
        g2m_genes.append(val) if phase=='G2/M' else None
    
    
    
    adata = adata.copy() if copy else adata
    ctrl_size = min(len(s_genes), len(g2m_genes), len(g1_genes))
    s_n_bins = round ((len(g1_genes)+len(g2m_genes))/ctrl_size)
    g1_n_bins = round ((len(s_genes)+len(g2m_genes))/ctrl_size)
    g2m_n_bins = round ((len(g1_genes)+len(s_genes))/ctrl_size)
    
    #add s-score
    score_genes_in_layer(adata, gene_list=s_genes, layer_choice=layer_choice, score_name='S_score', ctrl_size=ctrl_size, n_bins=s_n_bins, **kwargs)
    #add g2m-score
    score_genes_in_layer(adata, gene_list=g2m_genes, layer_choice=layer_choice, score_name='G2M_score', ctrl_size=ctrl_size, n_bins=g2m_n_bins, **kwargs)
    #add g1-score
    score_genes_in_layer(adata, gene_list=g1_genes, layer_choice=layer_choice, score_name='G1_score', ctrl_size=ctrl_size, n_bins=g1_n_bins, **kwargs)
    
    if not 'G1_score' in adata.obs.columns:
        print("WARNING: No G1-genes found in data set. Computing G1-score as -sum (S_score,G2M_score)")
        adata.obs['G1_score'] = -adata.obs[['S_score','G2M_score']].sum(1)
        
    scores = adata.obs[['S_score','G2M_score','G1_score']]
    
    #default phase is S
    phase = pd.Series('not_assigned', index=scores.index)
    
    #The = is to remove all `not assigned' as they cause issues downstream. if scores
    #are identical, then wether it is in one phase or the other does not matter much
    
    #if G2M is higher than S and G1, it's G2M
    phase[(scores.G2M_score >= scores.S_score) & (scores.G2M_score >= scores.G1_score)] = 'G2M'
    
    #if S is higher than G2M and G1, it's S
    phase[(scores.S_score >= scores.G2M_score) & (scores.S_score >= scores.G1_score)] = 'S'
    
    #if G1 is higher than G2M and S, it's G1
    phase[(scores.G1_score >= scores.G2M_score) & (scores.G1_score >= scores.S_score)] = 'G1'
    
    adata.obs['phase'] = phase
    #logg.hint(' \'phase\',cell cycle phase (adata.obs)') )
    return adata if copy else None

#%%Cell-cycle scoring - Geir
def compute_angles(points):
    """
    A function that calculates the angles of the cells based on the PCA coordinates
    Cells are first assigned from -pi to pi. Cells assigned to -pi are translated
    by 2pi so that each cell has an angle correspongin to one period if the cell cycle
    were to be shown as a circle.
    
    Function written by Geir Armun Svan Hasle

    Parameters
    ----------
    points : PCA data points
        Often given using adata.obsm['X_pca'][:,:2].

    Returns
    -------
    angles : numpy array
        An array containing the angles of each cell.

    """
    angles = np.arctan2(points[:,0], points[:,1])
    angles[angles < 0] += 2*np.pi
    return angles

def shift_data(data, n, direction = 'positive', reverse = False):
    """
    shifts the order of the AnnData object by a selected amount in the selected
    direction
    
    Function written by Geir Armun Svan Hasle

    Parameters
    ----------
    data : AnnData object
        The data that will be shifted.
    n : int
        The distance by which the data will be shifted.
    direction : string, optional
        positive or negative; determines the direction of the shift. The default is 'positive'.
    reverse : boolean, optional
        reverses the data if True. The default is False.

    Raises
    ------
    ValueError
           If the direction inputted is invalid

    Returns
    -------
    data : AnnData
        Return the shifted data.

    """
    
    if not direction in ['positive', 'negative']:
        raise ValueError('direction must be: positive,negative')
    #Need to find g2m g1 junction
    if direction == 'negative':
        data.obs['order'] -= n
        data.obs['order'][data.obs['order'] < 0] += len(data.obs)
    else:
        data.obs['order'] += n
        data.obs['order'][data.obs['order'] > len(data.obs)] -= len(data.obs)
    sort_order = data.obs['order'].argsort()[::-1] if reverse else data.obs['order'].argsort() 
    data = data[sort_order,:].copy()
    data.obs['order'] = np.arange(len(data.obs))
    return data

def plot_phase_bar_dist(data,bin_size,return_data=False,plot_path=None):
    """
    Function that plots the phase data into bar and line plots
    The data is put into `bins' for smoothing purposes
    The return of the data and saving of the figures is optional
    
    Function written by Geir Armun Svan Hasle

    Parameters
    ----------
    data : AnnData
        A AnnData object.
    bin_size : int
        The bin size used.
    return_data : Boolean, optional
        DESCRIPTION. The default is False.
    plot_path : string, optional
        The path to which the figures will be saved. The default is None.

    Returns
    -------
    count_df : pandas dataframe
        The count data used for the bar plot.
    ratio_df : pandas dataframe
        The ratio data (which phase is dominant) used for the line plot.

    """
    data_bb = data.copy()
    series_dict = {}
    colors_list=['royalblue','green','orange']
    for p in list(data_bb.obs['phase'].unique()):
        data_bb.obs['{}_count'.format(p)] = (data_bb.obs['phase'] == p).astype(int)
    data_bb.obs['phase_group'] = (np.arange(len(data_bb.obs)) // bin_size)*bin_size
    for p in list(data_bb.obs['phase'].unique()):
        series_dict['{}_counts'.format(p)] = data_bb.obs.groupby(['phase_group'])['{}_count'.format(p)].sum()
    count_df = pd.DataFrame({'G1': list(series_dict['G1_counts']), 
                             'S': list(series_dict['S_counts']), 
                             'G2M': list(series_dict['G2M_counts'])
                            }, 
                            index=np.arange(0,len(data_bb.obs),bin_size)
                    )
    if plot_path:
        if not os.path.exists(plot_path):
                os.makedirs(plot_path, exist_ok=True)
        #count_df.plot.bar(figsize=(30,10), xlabel='Pseudotime order', ylabel='Number of cells').get_figure().savefig(os.path.join(plot_path,"bar_dist.pdf"),bbox_inches = "tight")
        ax = count_df.plot.bar(figsize=(30,10),color=colors_list)
        ax.set_xlabel('Pseudotime order')
        ax.set_ylabel('Number of cells')
        plt.savefig(os.path.join(plot_path,"bar_dist.pdf"),bbox_inches='tight')
    else:
        count_df.plot.bar(figsize=(30,10),color=colors_list)
    #values = count_df.as_matrix(columns=data_bb.obs['phase'].unique())
    values = count_df.values
    for p in list(data_bb.obs['phase'].unique()):
        count_df['{}_ratio'.format(p)] = count_df[p]/values.sum(1)
    
    ratio_df = pd.DataFrame({'G1': list(count_df['G1_ratio']), 
                             'S': list(count_df['S_ratio']), 
                             'G2M': list(count_df['G2M_ratio'])
                            }, 
                            index=np.arange(0,len(data_bb.obs),bin_size)
                        )
    if plot_path:
        #count_df.plot.bar(figsize=(30,10)).get_figure().savefig(os.path.join(plot_path,"bar_dist.pdf"),bbox_inches = "tight")
        #ratio_df.plot(kind='line', figsize=(30,10), xlabel='Pseudotime order', ylabel='fractiosn', use_index=True).get_figure().savefig(os.path.join(plot_path,"line_dist.pdf"),bbox_inches = "tight")
        ax = ratio_df.plot(kind='line', figsize=(30,10), use_index=True,color=colors_list)
        ax.set_xlabel('Pseudotime order')
        ax.set_ylabel('Fraction')
        
        plt.savefig(os.path.join(plot_path,"line_dist.pdf"),bbox_inches='tight')
    else:
        ratio_df.plot(kind='line', figsize=(30,10), use_index=True,color=colors_list)
    
    if return_data:
        return count_df, ratio_df
    
def geir_QC_graph(adata):
    """
    Plots a kernel density plot for the different phases (G1, S, G2M)
    
    Function written by Geir Armun Svan Hasle

    Parameters
    ----------
    adata : AnnData Object

    Returns
    -------
    None.

    """
    
    
    phase_dict = {}
    for p in ['G1', 'S', 'G2M']:
        p_cells = adata[adata.obs['phase'] == p,:].copy()
        angles = p_cells.obs['angles']
        mean_angle = angles.mean()
        if angles.max() - angles.min() > 1.5*np.pi:
            angles[angles > np.pi] -= 2*np.pi
        var_angle = p_cells.obs['angles'].var()
        phase_dict[p] = {'mean': mean_angle, 'variance': var_angle, 'phase': p}
        # if mean_angle + 3.2*np.sqrt(var_angle) > 2*np.pi:
        #     phase_dict['{}-shift'.format(p)] = {'mean': mean_angle-2*np.pi, 'variance': var_angle, 'phase': p}
        # elif mean_angle - 3.2*np.sqrt(var_angle) < 0:
        #     phase_dict['{}-shift'.format(p)] = {'mean': mean_angle+2*np.pi, 'variance': var_angle, 'phase': p}
    
    plt.figure(figsize=(7,7))
    t = np.linspace(0,2*np.pi,1000)
    #t = np.linspace(0,len(data_bb.obs),len(data_bb.obs))
    for k,v in phase_dict.items():
        mean = v['mean']
        scale = np.sqrt(v['variance'])
        plt.plot(t,scipy.stats.norm.pdf(t, mean, scale),'-',label=k )
    plt.legend()
    

#%%Cell-cycle sorting methods

def selection_method(adata,highly_variable=True,CC_path=None):
    """
    Creates a subset of an AnnData object based on a selection method.
    Selecting for highly variable genes using Scanpy's method, or selecting for
    cell cycle regulating genes. Subsets stored as highly_variable for simplicity
    
    Function written by Yohan Lefol

    Parameters
    ----------
    adata : AnnData Object
        The AnnData/loom file.
    highly_variable : Boolean, optional
        if true, the selection method will be scanpy's highly variable gene selection
        otherwise it will select for cell cycle regulating genes. The default is True.

    Returns
    -------
    adata : AnnData object
        The overwritten AnnData object.

    """
    if highly_variable==True:
        sc.pp.highly_variable_genes(adata, min_mean=0.01235,max_mean=3,min_disp=0.5)
    else:
        g1_list,s_list,g2m_list=prepare_CC_lists(CC_path)
        select_for_CC_gene_markers(adata,g1_list,s_list,g2m_list)
    adata = adata[:, adata.var.highly_variable]
    return adata

def prepare_CC_lists(CC_path):
    """
    Function that reads a file and retrieves the genes associated to each phase
    
    Function written by Geir Armun Svan Hasle

    Parameters
    ----------
    CC_path : string
        path to file containing genes with associated cell cycle phases.

    Returns
    -------
    G1_list : list
        list of G1 genes.
    S_list : list
        list of S genes.
    G2M_list : list
        list of G2M genes.

    """
    df = pd.read_csv(CC_path,delimiter=',')
    S_list=[]
    G1_list=[]
    G2M_list=[]
    
    for idx,val in enumerate(df['gene']):
        phase=df['phase'][idx]
        S_list.append(val) if phase=='S' else None
        G1_list.append(val) if phase=='G1' else None
        G2M_list.append(val) if phase=='G2/M' else None

    return G1_list,S_list,G2M_list


def select_for_CC_gene_markers(adata, G1_list, S_list, G2M_list):
    """
    Used to subset AnnData object to only include cell cycle regulating genes
    
    Function written by Yohan Lefol

    Parameters
    ----------
    adata : AnnData
        The AnnData object.
    G1_list : list
        list of G1 genes.
    S_list : list
        list of S genes.
    G2M_list : list
        list of G2M genes.

    Returns
    -------
    None.

    """
    
    bool_df=pd.DataFrame(index=adata.var_names,columns=["CC_markers"])
    for idx,val in enumerate(bool_df.index):
        if val in S_list:
            bool_df["CC_markers"][idx]=True
        elif val in G1_list:
            bool_df["CC_markers"][idx]=True
        elif val in G2M_list:
            bool_df["CC_markers"][idx]=True
        else:
            bool_df["CC_markers"][idx]=False
    
    bool_df=bool_df.astype(bool)
    adata.var['highly_variable'] = bool_df["CC_markers"].values
    #return adata

def check_cols_and_rows(adata):
    """
    Simple function to check that there are no genes with 0 reads
    
    Function written by Yohan Lefol

    Parameters
    ----------
    adata : AnnData
        The AnnData object.

    Returns
    -------
    None.

    """
    
    if np.any(adata.X.sum(axis=0)==0)==True:
        start=len(adata.var_names)
        sc.pp.filter_genes(adata,min_counts=1)
        number_removed=abs(len(adata.var_names)-start)
        print("removed ",number_removed," genes with 0 reads")

def perform_scanpy_pca(adata,compute,exclude_gene_counts,exclude_CC):
    """
    Wrapper function for the scanpy PCA function. Used to create a diversity
    of PCAs.
    Mainly created for code legibility
    
    Function written by Geir Armun Svan Hasle

    Parameters
    ----------
    adata : AnnData
        The Anndata object.
    compute : boolean
        Wether PCA computation is to be done.
    exclude_gene_counts : boolean
        If n_genes and n_counts is to be plotted.
    exclude_CC : boolean
        If cell_cycle components (phase scores and phases) should be plotted.

    Returns
    -------
    None.

    """
    if compute==True:
        sc.tl.pca(adata,svd_solver='arpack')
    if exclude_gene_counts==False:
        sc.pl.pca(adata,color=['n_genes'])
        sc.pl.pca(adata,color=['n_counts'])
    if exclude_CC==False:
        sc.pl.pca(adata,color=['G1_score'])
        sc.pl.pca(adata,color=['S_score'])
        sc.pl.pca(adata,color=['G2M_score'])
        sc.pl.pca(adata,color=['phase'])

def auto_detect_orientation(adata):
    """
    Finds the orientation of the data based on a calculation of the median angle for each phase
    
    Function written by Yohan Lefol

    Parameters
    ----------
    adata : AnnData
        The AnnData object for which the orientation will be calculated.

    Returns
    -------
    orientation : string
        Either 'G1' or 'G2m' based on the orientation found.

    """    
    G1_median_angle=np.median(adata.obs.angles[adata.obs.phase=='G1'])
    S_median_angle=np.median(adata.obs.angles[adata.obs.phase=='S'])
    G2M_median_angle=np.median(adata.obs.angles[adata.obs.phase=='G2M'])
        
    if G1_median_angle>S_median_angle and G1_median_angle>G2M_median_angle:
        #G1 is highest angle, therefore next phase is smallest angle
        if S_median_angle<G2M_median_angle:
            orientation='G1'
        else:
            orientation='G2M'
    else:
        #Next angle is the one larger than G1
        if S_median_angle>G1_median_angle and S_median_angle>G2M_median_angle:
            if G2M_median_angle>G1_median_angle:
                orientation='G2M'
            else:
                orientation='G1'
        else:
            #G2M is max
            if S_median_angle>G1_median_angle:
                #S is in between G1 and G2M
                orientation='G1'
            else:
                orientation='G2M'
    return orientation

def find_angle_boundaries(adata):
    """
    Finds the angle boundaries based on the found order/phase boundaries
    
    Function written by Yohan Lefol
    
    Parameters
    ----------
    adata : AnnData
        AnnData object.

    Returns
    -------
    g1_ang_start : float
        The angle boundary for G1.
    s_ang_start : float
        The angle boundary for S.
    g2m_ang_start : float
        The angle boundary for G2M.

    """
    adata = adata[adata.obs['order'].argsort(),:]
    for idx,val in enumerate(adata.obs.order):
        if val == adata.uns['phase_boundaries']['g1_start']:
            g1_ang_start=adata.obs.angles[idx]
        elif val == adata.uns['phase_boundaries']['s_start']:
            s_ang_start=adata.obs.angles[idx]
        elif val == adata.uns['phase_boundaries']['g2m_start']:
            g2m_ang_start=adata.obs.angles[idx]
    
    print("G1 angle: ",g1_ang_start)
    print("S angle: ",s_ang_start)
    print("G2M angle: ",g2m_ang_start)
    return g1_ang_start, s_ang_start, g2m_ang_start


def phase_angle_assignment(adata,g1_limit,s_limit,g2m_limit):
    """
    Reassigns the phases of each cell based on their angles and the inputted
    phase angle boundaries.
    
    Function written by Yohan Lefol

    Parameters
    ----------
    adata : AnnData Object
        The AnnData object containing the cells and associated phases.
    g1_limit : float
        The g1 angle boundary.
    s_limit : float
        the s angle boundary.
    g2m_limit : float
        the G2M angle boundary.

    Returns
    -------
    None.

    """
    count_list=[]
    
    list_of_keys=[]
    list_of_values=[]
    
    angle_dict={}
    angle_dict['G1']=g1_limit
    angle_dict['G2M']=g2m_limit
    angle_dict['S']=s_limit
    angle_dict={k: v for k, v in sorted(angle_dict.items(), key=lambda item: item[1])}
    for i in angle_dict.keys():
        list_of_keys.append(i)
    for i in angle_dict.values():
        list_of_values.append(i)
    
    phase_series=pd.Series('not_assigned', index=adata.obs.index)
    for idx, val in enumerate(adata.obs.angles):
        if val >= list_of_values[0] and val<list_of_values[1]:
            phase_series[adata.obs.index[idx]]=list_of_keys[0]
            count_list.append(list_of_keys[0])
        elif val >=list_of_values[1] and val< list_of_values[2]:
            phase_series[adata.obs.index[idx]]=list_of_keys[1]
            count_list.append(list_of_keys[1])
        else:
            phase_series[adata.obs.index[idx]]=list_of_keys[2]
            count_list.append(list_of_keys[2])
            
    adata.obs['phase'] = phase_series
    g1_count = count_list.count('G1')
    s_count = count_list.count('S')
    g2m_count = count_list.count('G2M')
    print("new g1 count is ",g1_count)
    print("new s count is ",s_count)
    print("new g2m count is ",g2m_count)

def create_ratio_df_for_shift(data,bin_size):
    """
    A subset of Geir's function which only creates a ratio dataframe for subsetting purposes
    
    Function written by Yohan Lefol

    Parameters
    ----------
    data : AnnData
        AnnData object for whicht he ratio dataframe is calculated.
    bin_size : int
        The bin size for smoothing.

    Returns
    -------
    ratio_df : pandas dataframe
        A dataframe containing the ratios of each phase for each order of time.

    """
    data_bb = data.copy()
    series_dict = {}
    for p in list(data_bb.obs['phase'].unique()):
        data_bb.obs['{}_count'.format(p)] = (data_bb.obs['phase'] == p).astype(int)
    data_bb.obs['phase_group'] = (np.arange(len(data_bb.obs)) // bin_size)*bin_size
    for p in list(data_bb.obs['phase'].unique()):
        series_dict['{}_counts'.format(p)] = data_bb.obs.groupby(['phase_group'])['{}_count'.format(p)].sum()
    count_df = pd.DataFrame({'G1': list(series_dict['G1_counts']), 
                             'S': list(series_dict['S_counts']), 
                             'G2M': list(series_dict['G2M_counts'])
                            }, 
                            index=np.arange(0,len(data_bb.obs),bin_size)
                    )
    values = count_df.values
    for p in list(data_bb.obs['phase'].unique()):
        count_df['{}_ratio'.format(p)] = count_df[p]/values.sum(1)
    
    ratio_df = pd.DataFrame({'G1': list(count_df['G1_ratio']), 
                             'S': list(count_df['S_ratio']), 
                             'G2M': list(count_df['G2M_ratio'])
                            }, 
                            index=np.arange(0,len(data_bb.obs),bin_size)
                        )
    return ratio_df


def find_phase_boundaries_mod(data, ratio_df, orientation):
    """
    Function used to find the order which marks the beginning of each phase
    The order in which the phases are found are based on the given orientation
    
    Function written by Yohan Lefol

    Parameters
    ----------
    data : AnnData
        The AnnData object.
    ratio_df : pandas dataframe
        The calculated ratio for each phase at each given time point.
    orientation : string
        G1 or G2M to indicate the orientation in which the data is.

    Returns
    -------
    boundary_list : list
        a list containing integers of the three phase boundaries
        g1,s,g2m is the order.

    """
    
    G1_start_point=np.where(ratio_df['G1']==np.max(ratio_df['G1']))[0][0]
    G2M_start_point=np.where(ratio_df['G2M']==np.max(ratio_df['G2M']))[0][0]
    
    if orientation=='G1':
        start_point=G1_start_point
        var1='G1'
        var2='G2M'
    else:
        start_point=G2M_start_point
        var1='G2M'
        var2='G1'

    from scipy.interpolate import interp1d
    s_first = (np.where(ratio_df['S'][start_point+1:]>ratio_df[var1][start_point+1:])[0] + start_point+1)[0]
    s_start = interp1d(ratio_df[var1][start_point:s_first+1].values-ratio_df['S'][start_point:s_first+1].values,ratio_df[start_point:s_first+1].index)(0)
    var1_first = (np.where(ratio_df[var2][s_first+1:]>ratio_df['S'][s_first+1:])[0] + s_first+1)[0]
    var1_start = interp1d(ratio_df[var2][s_first+1:var1_first+1].values-ratio_df[var2][s_first+1:var1_first+1],ratio_df[s_first+1:var1_first+1].index)(0)
    
    var2_first = (np.where(ratio_df[var1][var1_first+1:]>ratio_df[var2][var1_first+1:])[0] + var1_first+1)
    #if g1_first is empty, check if they cross in [0,s_first]
    if var2_first.size == 0:
        var2_first = np.where(ratio_df[var1][:s_first]>ratio_df[var2][:s_first])[0][0]
        if var2_first == 0:
            var2_start = 0
        else:
            var2_start = interp1d(ratio_df[var2][:var2_first+1].values-ratio_df[var1][:var2_first+1],ratio_df[:var2_first+1].index)(0)
    else:
        var2_start = interp1d(ratio_df[var2][var1_first+1:].values-ratio_df[var1][var1_first+1:],ratio_df[var1_first+1:].index)(0)
    
    if orientation=='G1':
        g1_start=var2_start
        g2m_start=var1_start
    else:
        g1_start=var1_start
        g2m_start=var2_start
    
    boundary_list=[int(round(float(g1_start))), int(round(float(s_start))), int(round(float(g2m_start)))]
    
    return boundary_list

def compare_marker_genes_per_phase_mod(data,cc_path, phase_choice,do_plots=True,plot_path="./figures"):
    """
    A function which grades a AnnData file based on the expression of each gene for each cell
    and wether or not those genes have been associated to the expected phase
    The function gives the ability to plot, however it should be noted that this
    function can be quite RAM intensive and should be used accordingly
    
    Function initially written by Geir Armun Svan Hasle, adapted for lower RAM
    usage by Yohan Lefol

    Parameters
    ----------
    data : AnnData object
        The AnnData object containing the gene/cell and phase data.
    cc_path : string
        The file path to the cell cycle candidate list
    phase_choice : string
        G1, S, or G2M, used to select which phase to grade.
    do_plots : boolean, optional
        If plots are to be made. The default is True.
    plot_path : boolean, optional
        Where plots should be saved. The default is "./figures".

    Returns
    -------
    None.

    """
    
    
    g1_start=data.uns['phase_boundaries']['g1_start']
    s_start=data.uns['phase_boundaries']['s_start']
    g2m_start=data.uns['phase_boundaries']['g2m_start']
    
    known_df = pd.read_csv(cc_path,delimiter=',').dropna()
    if phase_choice=='G2M':
        p='G2/M'
    else:
        p=phase_choice
    expression_df = pd.DataFrame()
    p_df = known_df[known_df['phase'] == p]   #G2M == G2/M
    for i, r in p_df.iterrows():
        if not r['gene'] in list(data.var.index):
            continue
        gene = r['gene']  
        if do_plots==True:  #Adds total expression
            tot_expr = data[:,gene].X.mean()
            expression_df = expression_df.append({
                'known_phase': p,
                'gene_symbol': gene,
                'G1_expr': data[g1_start:s_start-1,:][:,gene].X.mean(),
                'S_expr': data[s_start:g2m_start-1,:][:,gene].X.mean(),
                'G2M_expr': data[g2m_start:,:][:,gene].X.mean(),
                'Total_expr': tot_expr,
            }, 
                ignore_index=True)
        else:   #Does not add total expression
            expression_df = expression_df.append({
                'known_phase': p,
                'gene_symbol': gene,
                'G1_expr': data[g1_start:s_start-1,:][:,gene].X.mean(),
                'S_expr': data[s_start:g2m_start-1,:][:,gene].X.mean(),
                'G2M_expr': data[g2m_start:,:][:,gene].X.mean(),
            }, 
                ignore_index=True)
    if expression_df.empty==True:
        print("No genes found for phase ",p)
        return
    expression_df.index = expression_df['gene_symbol']
    expression_df=expression_df.drop('gene_symbol',axis=1)
    #print(expression_df)
    if do_plots==True:
        if not os.path.exists(plot_path):
            os.makedirs(plot_path, exist_ok=True)
        for p in expression_df['known_phase'].unique():
            phase_df = expression_df[expression_df['known_phase'] == p].copy()
            phase_df=phase_df.astype({'G1_expr':float,'G2M_expr':float,'S_expr':float,'Total_expr':float})
            phase_df['G1_expr']=phase_df['G1_expr'].abs()
            phase_df['G2M_expr']=phase_df['G2M_expr'].abs()
            phase_df['S_expr']=phase_df['S_expr'].abs()
            phase_df['Total_expr']=phase_df['Total_expr'].abs()
            if phase_df.empty:
                print("No {} genes detected".format(p))
                continue
            ax = phase_df.plot(kind='bar', title='Mean expression of {} marker genes by modelled phase'.format(p), figsize=(10,10))
            ax.set_xlabel('Gene symbols')
            ax.set_ylabel('Reads Per Million')
            plt.savefig(os.path.join(plot_path,"mean_expression_{}.pdf".format(p.replace('/','-'))))
    
    score_ordering_mod(expression_df,p)
    del expression_df #To save on RAM
    # return expression_df

def score_ordering_mod(expression_df,phase):
    """
    Orders the scores for a specific phase for the 'compare_marker_genes_per_phase_mod' function
    
    Function initially written by Geir Armun Svan Hasle, adapted for
    lower RAM usageby Yohan Lefol
    

    Parameters
    ----------
    expression_df : pandas dataframe
        The expressions of the genes to be ordered by score.
    phase : string
        specifies the phase.

    Returns
    -------
    None.

    """
    p=phase
    score_dict = {}
    ind_dict = {'G1': 0, 'S': 1, 'G2/M': 2, 'G2': 2}
    total_score = 0
    # for p in expression_df['known_phase'].unique():
    expr_mat = expression_df[expression_df['known_phase'] == p]
    expr_mat= expr_mat[['G1_expr','S_expr','G2M_expr']]
    expr_mat = expr_mat.values
    score_list = []
    for row in expr_mat:
        score_list.append(int(np.where(row == np.max(row))[0][0] == ind_dict[p]))   
    phase_score = sum(score_list)
    score_dict[p] = phase_score
    total_score += phase_score
    
    print(p," phase: ","{}/{} genes classified correctly".format(phase_score,len(expression_df[expression_df['known_phase'] == p])))
        
    del score_dict#To save on RAM

def calculate_boundaries_post_shift(adata,boundary_list,orientation):
    """
    Recalculates the boundaries after the automated shift.
    In most cases the boundaries will not shift, but occasionnally
    the smoothed bin aspect of the shift will have caused a slight offset,
    this secondary calculations corrects for it
    
    Function written by Yohan Lefol

    Parameters
    ----------
    adata : AnnData object
        The AnnData object containing the data.
    boundary_list : list
        list of boundaries in g1,s,g2m order.
    orientation : string
        G1 or G2M to indicate the orientation of the dataset.

    Returns
    -------
    calculation_list : list
        The updated phase boundaries.

    """
    calculation_list=boundary_list
    
    G1_start=int(boundary_list[0])
    #S_start=int(boundary_list[1])
    G2M_start=int(boundary_list[2])
    
    if orientation=='G1':
        start_point=G1_start
    else:
        start_point=G2M_start
    
    for idx,val in enumerate(calculation_list):
        if val<start_point:
            calculation_list[idx]=len(adata.obs.index)-(abs(val-start_point))
        else:
            calculation_list[idx]=val-start_point
    return calculation_list
    
def automated_shift(adata,bin_size,shift_inc,orientation,recursion_limit=100,fail_safe=0):
    """
    A recursive function that shifts the data until it either manages to put the 
    G1/G2M crossover at 0 or until the maximum number of recursions is reached (recursion limit).
    It also calculates the new phase boundaries.
    
    
    Function written by Yohan Lefol

    Parameters
    ----------
    adata : AnnData object
        The AnnData object to be shifted.
    bin_size : int
        The bin sized used for data smoothing.
    shift_inc : int
        The number indication the shift amount.
    orientation : string
        Either G1 or G2M, indicates the orientation of the data.
    recursion_limit : int, optional
        The maximum number of recursions possible. The default is 10.
    fail_safe : int, optional
        Carry over variable incremented at each recursion, used to identify
        when the max amount of recursion (recursion_limit) has been
        reached. The default is 0.

    Returns
    -------
    AnnData object
        The updated/shifted AnnData object.

    """
    
    if fail_safe >=recursion_limit: #Reached the recursion limit, breaking out
        return False
    
    #create the ratio_df
    ratio_df=create_ratio_df_for_shift(adata,bin_size=bin_size)
    try:    #Attempt to find boundaries
        boundary_list=find_phase_boundaries_mod(adata,ratio_df,orientation=orientation)
    except: #Boundary function returned an error, do shift and try again
        adata=shift_data(adata, shift_inc, direction = 'negative', reverse=False)
        fail_safe+=1
        #Prevents the same warning from flooding the console based on recursion amount
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adata=automated_shift(adata,bin_size,shift_inc,orientation=orientation,recursion_limit=recursion_limit,fail_safe=fail_safe)
            return adata
    
    #Perform the shift by the found g1_start or g2m_start based on orientation
    if orientation=='G1':
        adata=shift_data(adata, boundary_list[0], direction = 'negative', reverse=False)
    else:
        adata=shift_data(adata, boundary_list[2], direction = 'negative', reverse=False)
    #Calculate the new starting amounts
    updated_boundary_list=calculate_boundaries_post_shift(adata,boundary_list,orientation=orientation)
    
    #input phase boundaries into adata
    adata.uns['phase_boundaries'] = {'g1_start': updated_boundary_list[0], 's_start': updated_boundary_list[1], 'g2m_start': updated_boundary_list[2]}
    return adata

def auto_shift_wrapper(adata,orientation):
    """
    A wrapper function for the automated shift which catches a potential error
    from the auto_detect orientation function. Sometimes when phase angles are
    very close together, the orientation isn't found accurately, this function 
    attempts with the found orientation, and if an error occurs, the opposite
    orientation is given instead.
    
    This wrapper was created for pre-phase reassignment data.
    
    Function written by Yohan Lefol

    Parameters
    ----------
    adata : AnnData object
        The AnnData object to be shifted.
    orientation : string
        Either G1 or G2M, indicates the orientation of the data.

    Raises
    ------
    ValueError
        If the function fails with both orientations, it means that the function 
        failed to shift the data regardless of orientation. This likely indicates
        that the score filter used should be changed (number of cells reduced).

    Returns
    -------
    adata : AnnData object
        The shifted AnnData object.
    orientation : string
        Either the same or corrected orientation depending on how the calculation
        went.

    """
    adata_check_bis=adata.copy()
    adata_check_bis=automated_shift(adata_check_bis,bin_size=20,orientation=orientation,recursion_limit=50,shift_inc=20)
    if type(adata_check_bis)==bool:
        adata_check_bis=adata.copy()
        if orientation=='G1':
            orientation='G2M'
        else:
            orientation='G1'
        adata_check_bis=automated_shift(adata_check_bis,bin_size=20,orientation=orientation,recursion_limit=50,shift_inc=20)
    adata_check=adata_check_bis
    if type(adata_check)==bool:
        raise ValueError ('Automated shift function failed using both orientation possibilities')
    return adata_check,orientation


def replace_obs_coordinates(adata_HV,adata_CC):
    """
    Creates a merged AnnData object that contains the variable (gene data) from  the
    adata_CC witht he observations (cell coordinates) from adata_HV
    
    Function written by Yohan Lefol

    Parameters
    ----------
    adata_HV : AnnData object
        The AnnData object created using the highly variable gene selection.
    adata_CC : AnnData object
        The AnnData object created using the cell cycle gene selection.

    Returns
    -------
    adata_merged : AnnData object
        The merged AnnData object.

    """
    adata_merged=adata_CC.copy()    #copy as to not overwrite
    
    for idx in adata_HV.obs.index:  #Loop through HV index
        if idx in adata_merged.obs.index:   #Check if index exists in CC (it should)
            for item in adata_merged[idx].obs:  #Loop through CC obs
                if item != "n_counts" and item != "n_genes":    #Overwrites all obs except n_counts and n_genes
                    adata_merged[idx].obs[item][0]=adata_HV[idx].obs[item][0]


    perform_scanpy_pca(adata_HV,compute=False,exclude_gene_counts=True,exclude_CC=False)
    
    perform_scanpy_pca(adata_CC,compute=False,exclude_gene_counts=True,exclude_CC=False)
    
    perform_scanpy_pca(adata_merged,compute=False,exclude_gene_counts=True,exclude_CC=False)
    
    sc.pl.pca_scatter(adata_merged, color=[ 'angles', 'phase'], components=['1,2'], save=False)
    
    return adata_merged


def find_angle_boundaries_post_reassignment(adata,orientation):
    """
    Finds the angle boundaries after having reassigned phases
    The found angle boundaries are saved into the observations of the 
    AnnData object in order to carry it over to velocyto
    
    Function written by Yohan Lefol

    Parameters
    ----------
    adata : AnnData object
        The AnnData obejct containing the data.
    orientation : string
        Either G1 or G2M to indicate the orientation of the data.

    Returns
    -------
    None.

    """
    if orientation=="G1":
        current_index=np.where(adata.obs.phase=="G1")[0][0]
        g1_angle=adata.obs.angles[current_index]
        remaining_phase=["S","G2M"]
    else:
        current_index=np.where(adata.obs.phase=="G2M")[0][0]
        g2m_angle=adata.obs.angles[current_index]
        remaining_phase=["S","G1"]
        
    for p in remaining_phase:
        P_list=np.where(adata.obs.phase==p)[0]
        for idx in P_list:
            if idx>current_index:
                if p=="S":
                    s_angle=adata.obs.angles[idx]
                elif p=="G1":
                    g1_angle=adata.obs.angles[idx]
                elif p=="G2M":
                    g2m_angle=adata.obs.angles[idx]
                current_index=idx
                break
    
    ang_boundaries=np.array([])
    for p in adata.obs.phase:
        if p == 'G1':
            ang_boundaries=np.append(ang_boundaries,g1_angle)
        elif p=='S':
            ang_boundaries=np.append(ang_boundaries,s_angle)
        else:   #it is G2M
            ang_boundaries=np.append(ang_boundaries,g2m_angle)
    
    adata.obs["angle_boundaries"]=ang_boundaries
    return(adata)



def score_filter_reassign(adata,cell_percent):
    """
    Calcualtes the necessary number of cells, then takes the the top x amount of cells
    for each phase score and associates that cell to the phase of that score.
    After having done the reassignment, it overwrites the AnnData object with
    the updated AnnData object.
    
    Function written by Yohan Lefol

    Parameters
    ----------
    adata : AnnData
        The AnnData object.
    cell_percent : float
        A percentage value in decimals.

    Returns
    -------
    None.

    """
    cell_num=int(len(adata.obs)*cell_percent)
    index_list=[]
    phase_series=pd.Series('not_assigned', index=adata.obs.index)
    for p in ['G1','S','G2M']:        
        if p == 'G1':
            score_series=adata.obs.G1_score.sort_values(ascending=False)
        elif p == 'S':
            score_series=adata.obs.S_score.sort_values(ascending=False)
        else:
            score_series=adata.obs.G2M_score.sort_values(ascending=False)
            
        for idx,val in enumerate(score_series):
            if idx<=cell_num:
                index_list.append(score_series.index[idx])
                cell_index=np.where(adata.obs.index==score_series.index[idx])[0][0]
                
                phase_series[cell_index]=p
                # adata.obs.phase[cell_index]=p  
    
    adata.obs['phase'] = phase_series
    bool_list=[]
    for index in adata.obs.index:
        bool_list.append(True) if index in index_list else bool_list.append(False)
            
    adata._inplace_subset_obs(bool_list)
    
    
#%% Calculation - Velocyto

def array_to_rmatrix(X):
    """
    
    Function taken from hgForebrainGlutamatergic velocyto notebook
    """
    nr, nc = X.shape
    xvec = robj.FloatVector(X.transpose().reshape((X.size)))
    xr = robj.r.matrix(xvec, nrow=nr, ncol=nc)
    return xr

def principal_curve(X, pca=True):
    """
    Function taken from hgForebrainGlutamatergic velocyto notebook
    
    Parameters
    ----------
    input : numpy.array
    returns:
    Result::Object
        Methods:
        projections - the matrix of the projection
        ixsort - the order ot the points (as in argsort)
        arclength - the lenght of the arc from the beginning to the point
    """
    # convert array to R matrix
    xr = array_to_rmatrix(X)
    
    if pca:
        #perform pca
        t = robj.r.prcomp(xr)
        #determine dimensionality reduction
        usedcomp = max( sum( np.array(t[t.names.index('sdev')]) > 1.1) , 4)
        usedcomp = min([usedcomp, sum( np.array(t[t.names.index('sdev')]) > 0.25), X.shape[0]])
        Xpc = np.array(t[t.names.index('x')])[:,:usedcomp]
        # convert array to R matrix
        xr = array_to_rmatrix(Xpc)

    #import the correct namespace
    princurve = importr("princurve",on_conflict="warn")
    
    #call the function
    fit1 = princurve.principal_curve(xr)

    
    #extract the outputs
    class Results:
        pass
    results = Results()
    results.projections = np.array( fit1[0] )
    results.ixsort = np.array( fit1[1] ) - 1 # R is 1 indexed
    results.arclength = np.array( fit1[2] )
    results.dist = np.array( fit1[3] )
    
    if pca:
        results.PCs = np.array(xr) #only the used components
        
    return results


def despline():
    """
    Function originates from DentateGyrus notebook for Velocyto

    Returns
    -------
    None.

    """
    ax1 = plt.gca()
    # Hide the right and top spines
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')




def minimal_xticks(start, end):
    """
    Determines the minimum number of x ticks possible
    
    Function originates from DentateGyrus notebook for Velocyto

    Parameters
    ----------
    start : int
        The first tick.
    end : int
        The last tick.

    Returns
    -------
    None.

    """
    end_ = np.around(end, -int(np.log10(end))+1)
    xlims = np.linspace(start, end_, 5)
    xlims_tx = [""]*len(xlims)
    xlims_tx[0], xlims_tx[-1] = f"{xlims[0]:.0f}", f"{xlims[-1]:.02f}"
    plt.xticks(xlims, xlims_tx,fontsize=15)

    
def minimal_yticks(start, end):
    """
    Determines the minimum number of y ticks possible
    
    Function originates from DentateGyrus notebook for Velocyto

    Parameters
    ----------
    start : int
        The first tick.
    end : int
        The last tick.

    Returns
    -------
    None.

    """
    end_ = np.around(end, -int(np.log10(end))+1)
    ylims = np.linspace(start, end_, 5)
    ylims_tx = [""]*len(ylims)
    ylims_tx[0], ylims_tx[-1] = f"{ylims[0]:.0f}", f"{ylims[-1]:.02f}"
    plt.yticks(ylims, ylims_tx,fontsize=15)

import math
def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return array[idx-1]
    else:
        return array[idx]

def create_boundary_dict(vlm):
    """
    Generates a boundary dictionnary containning the order point which marks the
    boundary between phases. The order associated to a phase is the first order
    of that specific phase.
    
    The function also accounts for orientation reversal
    
    Function written by Yohan Lefol

    Parameters
    ----------
    vlm : velocyto.analysis.VelocytoLoom
        The loom file as read by velocyto.

    Returns
    -------
    boundary_dict : dictionnary
        A dictionnary containing the order boundaries of each phase, it is later
        used in plotting functions.

    """
    all_angles=vlm.ca["angles"].copy()
    all_angles.sort()
    boundary_dict={}
    for ang in np.unique(vlm.ca["angle_boundaries"]):
        boundary_angle=find_nearest(all_angles,ang)
        boundary_order=np.where(vlm.ca["angles"]==boundary_angle)[0][0]
        ang_phase=vlm.ca["phase"][np.where(vlm.ca['angle_boundaries']==ang)[0][0]]
        boundary_dict[ang_phase]=[boundary_order,vlm.colorandum[np.where(vlm.ca['phase']==ang_phase)[0][0]]]
    cell_num_dict={}
    for k in ["G1","S","G2M"]:
        cell_num_dict[k]=len(np.where(vlm.ca['phase']==k)[0])
                        
    orientation=np.unique(vlm.ca['orientation'])[0]
    #new_boundaries
    if orientation=='G1':
        X=len(vlm.ca["CellID"])-boundary_dict['G1'][0]
        boundary_dict['G1'][0]=0
        boundary_dict['S'][0]=cell_num_dict["G1"]
        boundary_dict['G2M'][0]=cell_num_dict["G1"]+cell_num_dict["S"]
    else:
        X=len(vlm.ca["CellID"])-boundary_dict['G2M'][0]
        boundary_dict['G1'][0]=cell_num_dict["G2M"]+cell_num_dict["S"]
        boundary_dict['G2M'][0]=0
        boundary_dict['S'][0]=cell_num_dict["G2M"]
    
    arr_1=np.arange(start=0,stop=X)
    arr_2=np.arange(start=X,stop=len(vlm.ca["CellID"]))
    
    new_order=np.concatenate([arr_2,arr_1])
    vlm.ca["new_order"]=new_order
    
    return boundary_dict




def moving_average(array, window_size=200,orientation=None) :
    """
    A function which smoothes data using the moving average technique.
    In the case of G1 orientation, an offset will need to be corrected
    
    Function written by Yohan Lefol

    Parameters
    ----------
    array : numpy nd.array
        The array containing the data that will be smoothed.
    window_size : int, optional
        The amounf of values used at once for the smoothing. The default is 200.
    orientation : string, optional
        either G1 or G2M to indicate the orientation of the data. The default is None.

    Returns
    -------
    moving_averages : numpy nd.array
        The array containing the smoothed data.

    """
    # #for it to be circular, need to add the n first points to end
    array=np.concatenate([array,array[:window_size-1]])
    numbers_series=pd.Series(array)
    windows = numbers_series.rolling(window_size)
    moving_averages = windows.mean()
    moving_averages_list = moving_averages.tolist()
    without_nans = moving_averages_list[window_size - 1:]

    moving_averages=np.asarray(without_nans)
    if orientation=='G1':
        moving_averages=np.roll(moving_averages,int(window_size/2))

        
    return moving_averages

def smooth_layers(vlm,bin_size, window_size, spliced_array, unspliced_array, orientation):
    """
    The function first calculates the mean points for each data using the bin size,
    it then elongates the data to fit the number of order points, and finally
    runs the data throught the moving_average function in order to smooth it.
    
    Function written by Yohan Lefol

    Parameters
    ----------
    vlm : velocyto.analysis.VelocytoLoom
        The loom file as read by velocyto.
    bin_size : int
        The bin size used to calculate the mean points.
    window_size : int
        The window size used for the moving average smoothing.
    spliced_array : numpy nd.array
        The array containing the spliced data.
    unspliced_array : numpy nd.array
        The array containing the unspliced data.
    orientation : string
        Either G1 or G2M to indicate the orientation.

    Returns
    -------
    spli_mean_array : numpy nd.array
        The mean and smoothed spliced data.
    unspli_mean_array : numpy nd.array
        The mean and smoothed unspliced data .

    """
    num_bin=int(len(vlm.ca["new_order"])/bin_size)
    spli_mean_list=[]
    unspli_mean_list=[]
    for order in range(num_bin):
        if order==0:    #First iteration
            spli_mean_list.append(np.mean(spliced_array[0:bin_size]))
            unspli_mean_list.append(np.mean(unspliced_array[0:bin_size]))
        else:  
            spli_mean_list.append(np.mean(spliced_array[order*bin_size:(order+1)*bin_size]))              
            unspli_mean_list.append(np.mean(unspliced_array[order*bin_size:(order+1)*bin_size]))
    
    
    last_index_check=num_bin*bin_size
    if len(vlm.ca["new_order"])%bin_size==0:
        last_index_check=int(last_index_check-(bin_size/2))

    last_val_spli=np.mean(spliced_array[last_index_check:len(vlm.ca["new_order"])])
    last_val_unspli=np.mean(unspliced_array[last_index_check:len(vlm.ca["new_order"])])
    
    spli_mean_list.append(last_val_spli)
    unspli_mean_list.append(last_val_unspli)
    
    spli_mean_list.insert(0,last_val_spli)
    unspli_mean_list.insert(0,last_val_unspli)

    # These two for loops extend the means as desired.
    for idx,val in enumerate(spli_mean_list):
        if idx==0:#First iterattion:
            spli_mean_array=np.linspace(start=val,stop=spli_mean_list[idx+1],num=bin_size)
        else:
            if idx!=len(spli_mean_list)-1:
                if idx == len(spli_mean_list)-2:#Last iteration
                    spli_mean_array=np.concatenate([spli_mean_array,np.linspace(start=val,stop=spli_mean_list[idx+1],num=len(vlm.ca["new_order"])-(num_bin*bin_size))])
                else:
                    spli_mean_array=np.concatenate([spli_mean_array,np.linspace(start=val,stop=spli_mean_list[idx+1],num=bin_size)])
    for idx,val in enumerate(unspli_mean_list):
        if idx==0:#First iterattion:
            unspli_mean_array=np.linspace(start=val,stop=unspli_mean_list[idx+1],num=bin_size)
        else:
            if idx != len(unspli_mean_list)-1:
                if idx == len(unspli_mean_list)-2:
                    unspli_mean_array=np.concatenate([unspli_mean_array,np.linspace(start=val,stop=unspli_mean_list[idx+1],num=len(vlm.ca["new_order"])-(num_bin*bin_size))])
                else:
                    unspli_mean_array=np.concatenate([unspli_mean_array,np.linspace(start=val,stop=unspli_mean_list[idx+1],num=bin_size)])
    
    spli_mean_array=moving_average(spli_mean_array,window_size=window_size,orientation=orientation)
    unspli_mean_array=moving_average(unspli_mean_array,window_size=window_size,orientation=orientation)

    return spli_mean_array,unspli_mean_array

def align_yaxis(ax1, v1, ax2, v2):
    """adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    inv = ax2.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))
    miny, maxy = ax2.get_ylim()
    ax2.set_ylim(miny+dy, maxy+dy)

def count_symbols(array,orientation):
    """
    Function that creates an array of +1, -1, or 0 based on the values in the 
    array given.
    If the data is in G2M orientation, +1 and -1 are inverted
    
    Function written by Yohan Lefol

    Parameters
    ----------
    array : numpy nd.array
        Array containing data to be counted.
    orientation : string
        Either G1 or G2M to indicate the orientation of the data.

    Returns
    -------
    count_array : numpy nd.array
        The array containing the counted data.

    """
    count_list=[]
    if orientation=='G2M':#In the event of G2M orientation, symbols are reversed
        pos_num=-1
        neg_num=1
    else:
        pos_num=1
        neg_num=-1
    for idx,val in enumerate(array):
        if val>0:
            count_list.append(pos_num)
        elif val<0:
            count_list.append(neg_num)
        else:
            count_list.append(0)
    count_array=np.asarray(count_list)
    return count_array

def smooth_calcs(vlm, bin_size, window_size, spli_arr, unspli_arr, choice='mean'):
    """
    

    Parameters
    ----------
    vlm : velocyto.analysis.VelocytoLoom
        The loom file as read by velocyto.
    bin_size : int
        The size of the bin used for mean lines in smooth_layers function.
    window_size : int
        The size of the window used for the moving average function.
    spli_arr : numpy nd.array
        The numpy array containing the spliced data.
    unspli_arr : numpy nd.array
        The numpy array containing the unspliced data.
    choice : string, optional
        Used to either return the mean lines or velocity lines. The default is 'mean'.
        
    Returns
    -------
    spli_array : numpy nd.array
        The array containing the modified spliced data.
    unspli_array : numpy nd.array
        The array containing the modifyed data.
    """
    
    orientation=np.unique(vlm.ca['orientation'])[0]
    spli_mean_array,unspli_mean_array=smooth_layers(vlm,bin_size=bin_size,window_size=window_size,spliced_array=spli_arr,unspliced_array=unspli_arr,orientation=orientation)
    if choice=='mean':
        return spli_mean_array,unspli_mean_array
    else:   #Choice should be vel
        deriv_spli=np.diff(spli_mean_array)/np.diff(vlm.ca["new_order"])
        deriv_unspli=np.diff(unspli_mean_array)/np.diff(vlm.ca["new_order"])
        spli_array,unspli_array=smooth_layers(vlm,bin_size=bin_size,window_size=window_size,spliced_array=deriv_spli,unspliced_array=deriv_unspli,orientation=orientation)
    return spli_array,unspli_array




#%% Plotting - Velocyto
def plot_velocity_field(vlm):
    """
    Function which plots the velocity field along with a principal curve
    which shows the overall directionnality of the data.
    
    Function taken from the hgForebrainGlutamatergic notebook

    Parameters
    ----------
    vlm : velocyto.analysis.VelocytoLoom
        The loom file as read by velocyto.

    Returns
    -------
    None.

    """
    #Calls the principal curve function, used to produce a principal curve on the velocity field
    pc_obj =principal_curve(vlm.pcs[:,:4], False)
    pc_obj.arclength = np.max(pc_obj.arclength) - pc_obj.arclength
    
    #Plots the velocity field
    plt.figure(None,(9,9))
    #Plots the main field
    vlm.plot_grid_arrows(scatter_kwargs_dict={"alpha":0.7, "lw":0.7, "edgecolor":"0.4", "s":70, "rasterized":True},
                          min_mass=2.9, angles='xy', scale_units='xy',
                          headaxislength=2.75, headlength=5, headwidth=4.8, quiver_scale=0.35, scale_type="absolute")
    #Plot the arrows
    plt.plot(pc_obj.projections[pc_obj.ixsort,0], pc_obj.projections[pc_obj.ixsort,1], c="w", lw=6, zorder=1000000)
    plt.plot(pc_obj.projections[pc_obj.ixsort,0], pc_obj.projections[pc_obj.ixsort,1], c="k", lw=3, zorder=2000000)
    # plt.gca().invert_xaxis()
    plt.axis("off")
    plt.axis("equal");
    legend_elements = [mpl.lines.Line2D([0], [0], marker='o', color="w", label='G1', markerfacecolor=vlm.colorandum[np.where(vlm.ca['phase']=='G1')[0][0]], markersize=10),
                       mpl.lines.Line2D([0], [0], marker='o', color="w", label='S', markerfacecolor=vlm.colorandum[np.where(vlm.ca['phase']=='S')[0][0]], markersize=10),
                       mpl.lines.Line2D([0], [0], marker='o', color="w", label='G2M', markerfacecolor=vlm.colorandum[np.where(vlm.ca['phase']=='G2M')[0][0]], markersize=10)]    
    
    plt.legend(handles=legend_elements,loc='best')
    despline()
    # plt.legend(loc='best')
    plt.show()


def plot_markov(vlm):
    """
    Plots the standard PCA plot, and two markov plots, one showing the predicted
    points/cells for being the starting point,
    and another which shows the predicted end points/cells
    
    Function taken from the DentateGyrus velocyto notebook
    
    Parameters
    ----------
    vlm : velocyto.analysis.VelocytoLoom
        The loom file as read by velocyto.

    Returns
    -------
    None.

    """
    steps = 100, 100
    grs = []
    for dim_i in range(vlm.embedding.shape[1]):
        m, M = np.min(vlm.embedding[:, dim_i]), np.max(vlm.embedding[:, dim_i])
        m = m - 0.025 * np.abs(M - m)
        M = M + 0.025 * np.abs(M - m)
        gr = np.linspace(m, M, steps[dim_i])
        grs.append(gr)
    
    meshes_tuple = np.meshgrid(*grs)
    gridpoints_coordinates = np.vstack([i.flat for i in meshes_tuple]).T
    
    
    nn = NearestNeighbors()
    nn.fit(vlm.embedding)
    dist, ixs = nn.kneighbors(gridpoints_coordinates, 1)
    
    diag_step_dist = np.sqrt((meshes_tuple[0][0,0] - meshes_tuple[0][0,1])**2 + (meshes_tuple[1][0,0] - meshes_tuple[1][1,0])**2)
    min_dist = diag_step_dist / 2
    ixs = ixs[dist < min_dist]
    gridpoints_coordinates = gridpoints_coordinates[dist.flat[:]<min_dist,:]
    dist = dist[dist < min_dist]
    
    ixs = np.unique(ixs)
    
    plt.figure(None,(8,8))
    vcy.scatter_viz(vlm.embedding[ixs, 0], vlm.embedding[ixs, 1], c=vlm.colorandum[ixs], alpha=1, s=30, lw=0.4, edgecolor="0.4")
    plt.show()

    ###Start of markov end
    vlm.prepare_markov(sigma_D=diag_step_dist, sigma_W=diag_step_dist/2., direction='forward', cells_ixs=ixs)
    vlm.run_markov(starting_p=np.ones(len(ixs)), n_steps=2500)
    diffused_n = vlm.diffused - np.percentile(vlm.diffused, 3)
    diffused_n /= np.percentile(diffused_n, 97)
    diffused_n = np.clip(diffused_n, 0, 1)
    plt.figure(None,(7,7))
    plt.title(label="End points")
    vcy.scatter_viz(vlm.embedding[ixs, 0], vlm.embedding[ixs, 1], c=diffused_n, alpha=0.5, s=50, lw=0., edgecolor="", cmap="viridis_r", rasterized=True)
    plt.axis("off")
    plt.show()
    
    ##Start of Markov beginning
    vlm.prepare_markov(sigma_D=diag_step_dist, sigma_W=diag_step_dist/2., direction='backwards', cells_ixs=ixs)
    vlm.run_markov(starting_p=np.ones(len(ixs)), n_steps=2500)
    diffused_n = vlm.diffused - np.percentile(vlm.diffused, 3)
    diffused_n /= np.percentile(diffused_n, 97)
    diffused_n = np.clip(diffused_n, 0, 1)
    plt.figure(None,(7,7))
    plt.title(label="Beginning point")
    vcy.scatter_viz(vlm.embedding[ixs, 0], vlm.embedding[ixs, 1], c=diffused_n, alpha=0.5, s=50, lw=0., edgecolor="", cmap="viridis_r", rasterized=True)
    plt.axis("off")
    plt.show()


def create_bar_plots_custom(vlm,gene_list,second_gene_list=None,order=None,plot_title=None,y_label=True):
    """
    A bar plot which plots gene expression for each cell phase. Can take in two
    lists of gene names to allow plotting of numerous genes but only labelling
    of specific gene names. Warning, when plotting 100+ genes, labelling no longer
    shows accurate positinning in the bar plot.
    
    Function written by Yohan Lefol

    Parameters
    ----------
    vlm : velocyto.analysis.VelocytoLoom
        The loom file as read by velocyto.
    gene_list : list
        list of gene names that will be plotted and labelled if second_gene_list is None.
        if second_gene_list is not None, this list is used for labelling only
    second_gene_list : list, optional
        List of gene names to be plotted (not labelled) . The default is None.
    order : list, optional
        A list of three integers from 0-2 indicating
        the order (G1-S-G2M) to put the bars. The default is None.
    plot_title : string, optional
        The title of the plot. The default is None.
    y_label : Boolean, optional
        Add the y axis labels (gene names). The default is True.

    Returns
    -------
    None.

    """
    #aggregate will organize by ascending order of cluster_ix
    s = aggregate(vlm.cluster_ix, vlm.Sx_sz, func="mean", axis=1)
    u = aggregate(vlm.cluster_ix, vlm.Ux_sz, func="mean", axis=1)
    
    #reorganizes array based on custom user input
    if order is not None:
        s=s[:,order]
        u=u[:,order]
    else:
        order=np.unique(vlm.cluster_ix)  #reassign for labelling purposes
    
    label_array=np.array([])
    for i in order:
        idx=np.where(vlm.cluster_ix==i)
        label_array=np.append(label_array, vlm.ca["phase"][idx[0][0]])

    #Assign variables based on what the user wants
    if second_gene_list is not None:
        gene_list_plot=second_gene_list
        gene_list_label=gene_list
    else:
        gene_list_plot=gene_list
        gene_list_label=gene_list
    
    ix_filter=np.array([],int)
    for gene in gene_list_plot:
        if gene in vlm.ra["Gene"]:
            ix_filter=np.append(ix_filter,[np.where(vlm.ra["Gene"]==gene)[0][0]])
        
    Ssort = s[ix_filter,:] / s[ix_filter,:].sum(1)[:,None]
    Usort = u[ix_filter,:] / u[ix_filter,:].sum(1)[:,None]
    ix1 = np.argsort(Usort.argmax(1), kind='mergesort')
    ix2 = np.argsort(Ssort[ix1, :].argmax(1), kind='mergesort')
    ixa2b = ix1[ix2]
    
    ra_gene = np.array(vlm.ra["Gene"])[ix_filter][ixa2b]
    # gammas = np.array(vlm.gammas)[ix_filter][ixa2b]
    
    S = s[ix_filter,:][ixa2b,:]
    S_norm = np.array(S) - np.percentile(S, 1,1)[:,None]
    S_norm = S_norm / np.percentile(S_norm, 99,1)[:,None]
    S_norm = np.clip(S_norm, 0,1)
    
    U = u[ix_filter,:][ixa2b,:]
    U_norm = np.array(U) - np.percentile(U, 1,1)[:,None]
    U_norm = U_norm / np.percentile(U_norm, 99,1)[:,None]
    U_norm = np.clip(U_norm, 0,1)
    
    
    plt.figure(None, (7, 6))
    plt.subplot(131)
    plt.text(0.3,1.05,"spliced", fontdict={"size":12}, transform=plt.gca().transAxes )
    plt.xticks(ticks=np.arange(s.shape[1])+0.5, labels=label_array, fontsize=11, ha="center", va="center");
    plt.gca().tick_params(axis='x', labeltop=True, labelbottom=False, bottom=False )
    plt.gca().tick_params(axis='y', labelleft=True, left=False )
    
    
    plt.pcolormesh(S_norm, cmap=plt.cm.viridis, norm=mpl.colors.PowerNorm(gamma=1.5)) # aspect=0.1, interpolation="none"
    plt.gca().invert_yaxis()
    # for hl in np.where(np.diff(Ssort[ixa2b, :].argmax(1)))[0]:
    #     plt.axhline(hl, c="k", lw=2)
        
    plt.subplot(132)
    plt.text(0.27,1.05,"unspliced", fontdict={"size":12}, transform=plt.gca().transAxes )
    plt.xticks(ticks=np.arange(s.shape[1])+0.5, labels=label_array, fontsize=11, ha="center", va="center");
    plt.gca().tick_params(axis='x', labeltop=True, labelbottom=False, bottom=False)
    
    gene_loc_array=np.where(np.in1d(ra_gene,gene_list_label))[0]  #Create location array
    #Create the labels to go with the locations
    gene_labels=np.array([])
    if y_label is True:
        for loc in gene_loc_array:
            gene_labels=np.append(gene_labels, ra_gene[loc])
    
    #Genes, if present, are marked by small 'ticks' on right hand side    
    plt.yticks(ticks=gene_loc_array, labels=gene_labels, fontsize=11, ha="left", va="top");
    plt.gca().tick_params(axis='y', labelleft=False, left=False, labelright=True)
    
    
    cax = plt.pcolormesh(U_norm, cmap=plt.cm.viridis,norm=mpl.colors.PowerNorm(gamma=1.5),) #aspect=0.1, interpolation="none",   
    plt.gca().invert_yaxis()
    
    # for hl in np.where(np.diff(Ssort[ixa2b, :].argmax(1)))[0]:
    #     plt.axhline(hl, c="k", lw=2)    
    
    plt.subplot(133)
    plt.text(0.24,1.05,plot_title, fontdict={"size":12}, transform=plt.gca().transAxes )
    plt.axis("off")
    plt.colorbar(cax, ax=plt.gca(),ticks=[0,0.5,1.], orientation='vertical')
    # plot_path="my_figures/bar_plots"
    # name=plot_title
    # if not os.path.exists(plot_path):
    #     os.makedirs(plot_path, exist_ok=True)
    # plt.savefig(os.path.join(plot_path,name))
    plt.show() 



def my_portrait_plots(vlm,gene_index,gene_name,plot_choice,boundary_dict=None,bin_size=None,subplot_coordinates=None,window_size=200):
    """
    A function regrouping several plotting functions for convenience. Plots are selected
    via the plot_choice parameter and can be layer_plot, velocity_lines, velocity_counts,
    and phase_portrait.
    
    Function written by Yohan Lefol

    Parameters
    ----------
    vlm : velocyto.analysis.VelocytoLoom
        The loom file as read by velocyto.
    gene_index : int
        The index indicating the genes position in the velocyto object.
    gene_name : string
        name of gene to be used as plot title.
    plot_choice : string
        Either layer_plot,velocity_lines, velocity_counts, or phase_portrait.
        It dictates which plot will be created
    boundary_dict : Dictionnary, optional
        A dictionnary containing the order boundaries for celll cycle phases
        as well as the color code for each phase. The default is None.
    bin_size : int, optional
        The bin used for the smoothing of the data (mean lines, velocity lines). 
        The default is None.
    subplot_coordinates : matplotlib.axes._subplots.AxesSubplot, optional
        The matplotlib subplot/location fo the plot being created. The default is None.
    window_size: int, optional
        The size of the window used in the moving average for data smoothing.
        The default is 200.

    Returns
    -------
    None.

    """
    ix=gene_index
    gn=gene_name
    ax=subplot_coordinates
    if plot_choice=='layer_plot':
        ax.scatter(vlm.ca["new_order"], vlm.Ux_sz[ix, :], alpha=0.7, c="#b35806", s=5, label="unspliced")

        ax.set_ylim(0, np.max(vlm.Ux_sz[ix,:])*1.02)
        minimal_yticks(0, np.max(vlm.Ux_sz[ix,:])*1.02)
        ax_2 = ax.twinx()
        ax_2.scatter(vlm.ca["new_order"], vlm.Sx_sz[ix, :], alpha=0.7, c="#542788", s=5, label="spliced")

        ax_2.set_ylim(0, np.max(vlm.Sx_sz[ix,:])*1.02)
        minimal_yticks(0, np.max(vlm.Sx_sz[ix,:])*1.02)
        
        ax.set_ylabel("unspliced",labelpad=-20,fontsize=20)
        ax_2.set_ylabel("spliced",labelpad=-20,fontsize=20)
            
        plt.xlim(0,np.max(vlm.ca["new_order"]))
        plt.title(gn, fontsize=20)
       

        ax.set_xlabel("order",labelpad=-10,fontsize=20)
        p = np.min(vlm.ca["new_order"])
        P = np.max(vlm.ca["new_order"])
        # ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.xticks(np.linspace(p,P,5), [f"{p:.0f}", "","","", f"{P:.0f}"],fontsize=15)
        ax.tick_params(axis='x',labelsize=15)
        
        spli_mean_array,unspli_mean_array=smooth_calcs(vlm,bin_size=bin_size,window_size=window_size,spli_arr=vlm.Sx_sz[ix,:],unspli_arr=vlm.Ux_sz[ix,:],choice='mean')
        
        #Insurance
        bin_order_axis=np.arange(start=0,stop=len(vlm.ca["new_order"]))
        
        ax_2.plot(bin_order_axis, spli_mean_array, c="#2a1344",linewidth=5,label='mean_spliced')
        ax.plot(bin_order_axis, unspli_mean_array, c="#7d3d04",linewidth=5,label='mean_unspliced')


        plot_ax_lines_phase_portrait(vlm,boundary_dict)
        
        lines,labels=ax.get_legend_handles_labels()
        lines_2,labels_2=ax_2.get_legend_handles_labels()
        ax_2.legend(lines + lines_2, labels + labels_2, loc='best')
                
        if np.unique(vlm.ca['orientation'])[0] == 'G2M':
            plt.gca().invert_xaxis()
            ax.set_xlabel("order (reversed)",labelpad=-10,fontsize=20)
    
    if plot_choice=="velocity_lines" or plot_choice=="velocity_counts":
        spli_array,unspli_array=smooth_calcs(vlm,bin_size=bin_size,window_size=window_size,spli_arr=vlm.Sx_sz[ix,:],unspli_arr=vlm.Ux_sz[ix,:],choice='vel')
        bin_order_axis=np.arange(start=0,stop=len(vlm.ca["new_order"]))

    if plot_choice=="velocity_lines":
        plt.title(gn, fontsize=20)
        plot_ax_lines_phase_portrait(vlm,boundary_dict)

        # ax.plot(bin_order_axis,unspli_array,c="#7d3d04",linewidth=5,label='velocity_unspliced')
        ax.plot(bin_order_axis,unspli_array,c="#b35806",linewidth=5,label='velocity_unspliced')

        ax.set_ylim(np.min(unspli_array)*1.5, np.max(unspli_array)*1.5)
        ax.get_yaxis().set_visible(False)        
        
        ax_2 = ax.twinx()
        # ax_2.plot(bin_order_axis,spli_array,c="#2a1344",linewidth=5,label='velocity_spliced')
        ax_2.plot(bin_order_axis,spli_array,c="#542788",linewidth=5,label='velocity_spliced')
        ax_2.set_ylim(np.min(spli_array)*1.5, np.max(spli_array)*1.5)
        ax_2.get_yaxis().set_visible(False) 
        plt.xlim(0,np.max(vlm.ca["new_order"]))
        
        align_yaxis(ax, 0, ax_2, 0)
        lines,labels=ax.get_legend_handles_labels()
        lines_2,labels_2=ax_2.get_legend_handles_labels()
        ax_2.legend(lines + lines_2, labels + labels_2, loc='best',prop={"size":20})
        
        ax.set_xlabel("order",labelpad=-10,fontsize=20)
        p = np.min(vlm.ca["new_order"])
        P = np.max(vlm.ca["new_order"])
        # ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.xticks(np.linspace(p,P,5), [f"{p:.0f}", "","","", f"{P:.0f}"],fontsize=15)
        ax.tick_params(axis='x',labelsize=15)
        
        if np.unique(vlm.ca['orientation'])[0] == 'G2M':
            plt.gca().invert_xaxis()
            ax.invert_yaxis()
            ax_2.invert_yaxis()
            ax.set_xlabel("order (reversed)",fontsize=20)
            
    if plot_choice=="velocity_counts":
        spli_vel_list=count_symbols(spli_array,orientation=np.unique(vlm.ca['orientation'])[0])
        unspli_vel_list=count_symbols(unspli_array,orientation=np.unique(vlm.ca['orientation'])[0])

        #plot symbols
        #########
        plt.title(gn, fontsize=20)
        plot_ax_lines_phase_portrait(vlm,boundary_dict)
        
        # plt.plot(bin_order_axis,unspli_vel_list,c="#7d3d04",linewidth=5,label='velocity_unspliced')
        # plt.plot(bin_order_axis,spli_vel_list,c="#2a1344",linewidth=5,label='velocity_spliced')
        plt.plot(bin_order_axis,unspli_vel_list,c="#b35806",linewidth=5,label='velocity_unspliced')
        plt.plot(bin_order_axis,spli_vel_list,c="#542788",linewidth=5,label='velocity_spliced')

        plt.ylim(-1.25, 1.25)
        minimal_yticks(-1*1.02, 1*1.02)
        
        plt.xlim(0,np.max(vlm.ca["new_order"]))
        
        
        p = np.min(vlm.ca["new_order"])
        P = np.max(vlm.ca["new_order"])
        # ax.spines['right'].set_visible(False)
        # plt.spines['top'].set_visible(False)
        plt.xticks(np.linspace(p,P,5), [f"{p:.0f}", "","","", f"{P:.0f}"],fontsize=15)
        ax.tick_params(axis='x',labelsize=15)
        plt.xlabel("order",labelpad=-10,fontsize=20)
        plt.legend(loc='best',prop={"size":20})
        
        if np.unique(vlm.ca['orientation'])[0] == 'G2M':
            plt.gca().invert_xaxis()
            # plt.gca().invert_yaxis()
            ax.set_xlabel("order (reversed)",fontsize=20)

    if plot_choice=="phase_portrait":
        vcy.scatter_viz(vlm.Sx_sz[ix,:], vlm.Ux_sz[ix,:], c=vlm.colorandum, s=5, alpha=0.7, rasterized=True)
        plt.title(gn, fontsize=20)
        xnew = np.linspace(0,vlm.Sx[ix,:].max())
        plt.plot(xnew, vlm.gammas[ix] * xnew + vlm.q[ix], c="k")
        plt.ylim(0, np.max(vlm.Ux_sz[ix,:]))
        plt.xlim(0, np.max(vlm.Sx_sz[ix,:]))
        plt.ylabel("unspliced",labelpad=-20,fontsize=20)
        plt.xlabel("spliced",labelpad=-10,fontsize=20)
        minimal_yticks(0, np.max(vlm.Ux_sz[ix,:])*1.02)
        minimal_xticks(0, np.max(vlm.Sx_sz[ix,:])*1.02)
        despline()


def plot_grid_of_figures(vlm,gene_list,boundary_dict,bin_size=100,window_size=200):
    """
    Function which creates a grid of plots for a list of gene names
    
    Function written by Yohan Lefol

    Parameters
    ----------
    vlm : velocyto.analysis.VelocytoLoom
        The loom file as read by velocyto.
    gene_list : list
        list of gene names.
    boundary_dict : dictionnary
        Dictionnary containing the order boundaries and color codes for each phase.
    bin_size : int, optional
        The bin sized used for the data smoothing. The default is 100
    window_size: int, optional
        The size of the window used in the moving average for data smoothing.
        The default is 200.

    Returns
    -------
    None.

    """
    plt.rcParams.update({'figure.max_open_warning': 0})
    mpl.rc('figure', max_open_warning = 0)    

    for i, gn in enumerate(gene_list):
        check_gene(vlm,gn)
        
        plt.figure(None, (24.5,5.5), dpi=80)
        gs = plt.GridSpec(1,3)
        ax = plt.subplot(gs[i*3])
        ix=np.where(vlm.ra["Gene"] == gn)[0][0]
        
        my_portrait_plots(vlm,ix,gn,plot_choice="layer_plot",boundary_dict=boundary_dict,bin_size=bin_size,subplot_coordinates=ax,window_size=window_size)

        ax = plt.subplot(gs[i*3+1])
        my_portrait_plots(vlm,ix,gn,plot_choice="velocity_lines",boundary_dict=boundary_dict,bin_size=bin_size,subplot_coordinates=ax,window_size=window_size)
        
        ax = plt.subplot(gs[i*3+2])
        my_portrait_plots(vlm,ix,gn,plot_choice="velocity_counts",boundary_dict=boundary_dict,bin_size=bin_size,subplot_coordinates=ax,window_size=window_size)
        
        # ax = plt.subplot(gs[i*6+3])
        # my_portrait_plots(vlm,ix,gn,plot_choice="phase_portrait")
        
        # plot_velocity_as_color_mod(vlm,gene_name=gn, which_tsne="Pcs" ,gs=gs[i*6+4])#, s=3, rasterized=True)
        # plot_expression_as_color_mod(vlm,gene_name=gn, which_tsne="Pcs", gs=gs[i*6+5])#, s=3, rasterized=True)

    plt.gcf().subplots_adjust(bottom=0.70)        
    plt.tight_layout()
    # for k in control_dict:
    #     if gn in control_dict[k]:
    #         break
    # plot_path="my_figures/temp_folder"
    # name=gn
    # if not os.path.exists(plot_path):
    #     os.makedirs(plot_path, exist_ok=True)
    # plt.savefig(os.path.join(plot_path,name))
    plt.show()        
    
    # return spli_mean_array,unspli_mean_array
    
def plot_layers(vlm,gene_list,boundary_dict,bin_size=100,window_size=200):
    """
    Function that checks if each gene in the list is present in the velocyto
    object and then plots theyr layer_plot separately
    
    Function written by Yohan Lefol

    Parameters
    ----------
    vlm : velocyto.analysis.VelocytoLoom
        The loom file as read by velocyto.
    gene_list : list
        list of gene names.
    boundary_dict : dictionnary
        Dictionnary containing the order boundaries and color codes for each phase.
    bin_size : int, optional
        The bin sized used for the data smoothing. The default is 100
    window_size: int, optional
        The size of the window used in the moving average for data smoothing.
        The default is 200.

    Returns
    -------
    None.

    """

    for i, gn in enumerate(gene_list):
        check_gene(vlm,gn)
        plt.figure(None,(7.5,5),dpi=80)
        ax = plt.subplot()
        try:
            ix=np.where(vlm.ra["Gene"] == gn)[0][0]
        except:
            continue 
        my_portrait_plots(vlm,ix,gn,plot_choice="layer_plot",boundary_dict=boundary_dict,bin_size=bin_size,subplot_coordinates=ax)

def plot_velocity(vlm,gene_list,boundary_dict,bin_size=100,window_size=200,do_lines=True,do_counts=True):
    """
    Function which plots either the smoothed velocity line plot or the velocity
    count plot, or both. The function ensures that the genes names in the list
    are present in the velocyto object
    
    Function written by Yohan Lefol

    Parameters
    ----------
    vlm : velocyto.analysis.VelocytoLoom
        The loom file as read by velocyto.
    gene_list : list
        list of gene names.
    boundary_dict : dictionnary
        Dictionnary containing the order boundaries and color codes for each phase.
    bin_size : int
        The bin sized used for the data smoothing.
    do_lines : Boolean, optional
        If the smoothed line plot should be done. The default is True.
    do_counts : Boolean, optional
        If the count plot should be done. The default is True.
    bin_size : int, optional
        The bin sized used for the data smoothing. The default is 100
    window_size: int, optional
        The size of the window used in the moving average for data smoothing.
        The default is 200.

    Returns
    -------
    None.

    """

    for i, gn in enumerate(gene_list):
        check_gene(vlm,gn)
        if do_lines==True and do_counts==True:
            plt.figure(None,(16.5,7.5),dpi=80)
            gs = plt.GridSpec(1,2)
        else:
            plt.figure(None,(7.5,6.5),dpi=80)
        ix=np.where(vlm.ra["Gene"] == gn)[0][0]
        if do_lines==True and do_counts==True:
            ax=plt.subplot(gs[i*2])
            my_portrait_plots(vlm,ix,gn,plot_choice="velocity_lines",boundary_dict=boundary_dict,bin_size=bin_size,subplot_coordinates=ax, window_size=window_size)
            ax2=plt.subplot(gs[i*2+1])
            my_portrait_plots(vlm,ix,gn,plot_choice="velocity_counts",boundary_dict=boundary_dict,bin_size=bin_size,subplot_coordinates=ax, window_size=window_size)
        elif do_counts==True:
            ax=plt.subplot()
            my_portrait_plots(vlm,ix,gn,plot_choice="velocity_counts",boundary_dict=boundary_dict,bin_size=bin_size,subplot_coordinates=ax, window_size=window_size)
        elif do_lines==True:
            ax=plt.subplot()
            my_portrait_plots(vlm,ix,gn,plot_choice="velocity_lines",boundary_dict=boundary_dict,bin_size=bin_size,subplot_coordinates=ax, window_size=window_size)

def plot_phase_portrait(vlm,gene_list):  
    """
    Checks if gene names in list are in velocyto object and plots phase portait
    
    Function written by Yohan Lefol

    Parameters
    ----------
    vlm : velocyto.analysis.VelocytoLoom
        The loom file as read by velocyto.
    gene_list : list
        list of gene names.

    Returns
    -------
    None.

    """
    for i, gn in enumerate(gene_list):
        check_gene(vlm,gn)
        plt.figure(None,(7.5,5),dpi=80)
        ix=np.where(vlm.ra["Gene"] == gn)[0][0]
        my_portrait_plots(vlm,ix,gn,plot_choice="phase_portrait")
        
        
def plot_ax_lines_phase_portrait(vlm,boundary_dict):
    """
    Function which plots colored horizontal lines to represent the cell cycle
    phases at specific time points. The phases are always shown in the G1-S-G2M
    order. Function also plots vertical black axis lines to clearly show the 
    transition between one cell cycle phase to the next.
    The lines are based on the boundary order.

    Function written by Yohan Lefol
    
    Parameters
    ----------
    vlm : velocyto.analysis.VelocytoLoom
        The loom file as read by velocyto.
    boundary_dict : dictionnary
        Dictionnary containing the order boundaries and color codes for each phase.

    Returns
    -------
    None.

    """
    for key,order in boundary_dict.items():
        plt.axvline(order[0],c='k',lw=2)
    
    orientation=np.unique(vlm.ca['orientation'])[0]
    #Sorts based on the order
    phase_order=sorted(boundary_dict,key=lambda k: boundary_dict[k][0])
    for inc,p in enumerate(phase_order):
        if orientation == 'G1':
            if p=='G1':
                color_used=boundary_dict['G2M'][1]
            elif p== 'S':
                color_used=boundary_dict['G1'][1]
            else:
                color_used=boundary_dict['S'][1]
        if orientation=='G2M':
            if p=='G1':
                color_used=boundary_dict['S'][1]
            elif p== 'S':
                color_used=boundary_dict['G2M'][1]
            else:
                color_used=boundary_dict['G1'][1] 
        if inc==0:
            plt.hlines(0,0,boundary_dict[p][0], colors=color_used, linestyles='solid',lw=8)
        else:
            plt.hlines(0,boundary_dict[phase_order[inc-1]][0],boundary_dict[p][0], colors=color_used, linestyles='solid',lw=8)
        
        if inc==2 and boundary_dict[p][0]<np.max(vlm.ca['new_order']):
            # if orientation == 'G1':
            plt.hlines(0,boundary_dict[phase_order[inc]][0],np.max(vlm.ca['new_order']), colors=boundary_dict[p][1], linestyles='solid',lw=8)
            # if orientation == 'G2M':
            #     plt.hlines(0,boundary_dict[phase_order[inc]][0],np.max(vlm.ca['new_order']), colors=boundary_dict[phase_order[0]][1], linestyles='solid',lw=8)


def plot_velocity_as_color_mod(vlm, gene_name: str=None, cmap= plt.cm.RdBu_r, gs=None, which_tsne: str="ts") -> None:
    """Plot velocity as color on the Tsne embedding
    
    Native function to Velocyto, modified by Yohan Lefol 
    
    Arguments
    ---------
    vlm : velocyto.analysis.VelocytoLoom
        The loom file as read by velocyto.
    gene_name: str
        The name of the gene, should be present in self.S
    cmap: maplotlib.cm.Colormap, default=maplotlib.cm.RdBu_r
        Colormap to use, divergent ones are better, RdBu_r is default
        Notice that 0 will be always set as the center of the colormap. (e.g. white in RdBu_r)
    gs: Gridspec subplot
        Gridspec subplot to plot on.
    which_tsne: str, default="ts"
        the name of the attributed where the desired embedding is stored
    **kwargs: dict
        other keywords arguments will be passed to the plt.scatter call
    
    Returns
    -------
    Nothing
    """
    check_gene(vlm,gene_name)
    ix = np.where(vlm.ra["Gene"] == gene_name)[0][0]
    kwarg_plot = {"alpha": 1, "s": 30, "edgecolor": "0.4", "lw": 0.4,"rasterized":True}
    kwarg_plot_2 = {"alpha": 1, "s": 30, "edgecolor": "0.4", "lw": 0.4,"rasterized":True}
    # kwarg_plot.update(kwargs)
    if gs is None:
        plt.figure(figsize=(10, 10))
        plt.subplot(111)
    else:
        plt.subplot(gs)
    
    tsne = getattr(vlm, which_tsne)
    if vlm.which_S_for_pred == "Sx_sz":
        tmp_colorandum = vlm.Sx_sz_t[ix, :] - vlm.Sx_sz[ix, :]
    else:
        tmp_colorandum = vlm.Sx_t[ix, :] - vlm.Sx[ix, :]
    if (np.abs(tmp_colorandum) > 0.00005).sum() < 10:  # If S vs U scatterplot it is flat
        print("S vs U scatterplot it is flat")
        return
    limit = np.max(np.abs(np.percentile(tmp_colorandum, [1, 99])))  # upper and lowe limit / saturation
    tmp_colorandum = tmp_colorandum + limit  # that is: tmp_colorandum - (-limit)
    tmp_colorandum = tmp_colorandum / (2 * limit)  # that is: tmp_colorandum / (limit - (-limit))
    tmp_colorandum = np.clip(tmp_colorandum, 0, 1)
    vcy.scatter_viz(tsne[:, 0], tsne[:, 1], c=vlm.colorandum, **kwarg_plot)
    vcy.scatter_viz(tsne[:, 0], tsne[:, 1], c=cmap(tmp_colorandum), **kwarg_plot_2)
    plt.axis("off")
    plt.title(f"{gene_name} Velocity")

def plot_expression_as_color_mod(vlm, gene_name: str=None, imputed: bool= True, cmap = plt.cm.Greens,gs=None, which_tsne: str="ts") -> None:
    """Plot expression as color on the Tsne embedding

    Native function to Velocyto, modified by Yohan Lefol     

    Arguments
    ---------
    vlm : velocyto.analysis.VelocytoLoom
        The loom file as read by velocyto.
    gene_name: str
        The name of the gene, should be present in self.S
    imputed: bool, default=True
        whether to plot the smoothed or the raw data
    cmap: maplotlib.cm.Colormap, default=maplotlib.cm.Greens
        Colormap to use.
    gs: Gridspec subplot
        Gridspec subplot to plot on.
    which_tsne: str, default="ts"
        the name of the attributed where the desired embedding is stored
    **kwargs: dict
        other keywords arguments will be passed to the plt.scatter call

    Returns
    -------
    Nothing
    """
    check_gene(vlm,gene_name)
    ix = np.where(vlm.ra["Gene"] == gene_name)[0][0]
    kwarg_plot = {"alpha": 1, "s": 30, "edgecolor": "0.4", "lw": 0.4,"rasterized":True}
    kwarg_plot_2 = {"alpha": 1, "s": 30, "edgecolor": "0.4", "lw": 0.4,"rasterized":True}
    if gs is None:
        plt.figure(figsize=(10, 10))
        plt.subplot(111)
    else:
        plt.subplot(gs)

    tsne = getattr(vlm, which_tsne)
    if imputed:
        if vlm.which_S_for_pred == "Sx_sz":
            tmp_colorandum = vlm.Sx_sz[ix, :]
        else:
            tmp_colorandum = vlm.Sx[ix, :]
    else:
        tmp_colorandum = vlm.S_sz[ix, :]
        
    tmp_colorandum = tmp_colorandum / np.percentile(tmp_colorandum, 99)
    # tmp_colorandum = np.log2(tmp_colorandum+1)
    tmp_colorandum = np.clip(tmp_colorandum, 0, 1)

    vcy.scatter_viz(tsne[:, 0], tsne[:, 1], c=vlm.colorandum, **kwarg_plot)
    vcy.scatter_viz(tsne[:, 0], tsne[:, 1], c=cmap(tmp_colorandum), **kwarg_plot_2)
    plt.axis("off")
    plt.title(f"{gene_name} Expression")


def check_gene(vlm,gene_name):
    """
    Simple function to check if gene is in the dataset, written as function
    to avoid code duplication within the plotting functions.
    Plotting functions called automatically will not require this function, it
    exists for when users may call plotting functions themselves.
    
    Function written by Yohan Lefol

    Parameters
    ----------
    vlm : velocyto.analysis.VelocytoLoom
        The loom file as read by velocyto.
    gene_name : string
        Name of the gene which needs to be verified.

    Raises
    ------
    KeyError
        Error indicating that the key (Gene) isn't in the dataset.

    Returns
    -------
    None.

    """
    if gene_name not in vlm.ra['Gene']:
        raise KeyError("Gene '"+ str(gene_name)+"' not in dataset")

#%%Sorting algorithms


def load_gene_control(path):
    """
    Function created to load a specific file into a dictionnary
    The file is a control file created using plots to sort genes into 
    categories
    
    Function created by Yohan Lefol

    Parameters
    ----------
    path : string
        the file path to the gene control file.

    Returns
    -------
    my_dict : dictionnary
        A dictionnary with categories as keys and gene names as values.

    """
    gene_file=pd.read_csv(path)
    my_dict={}
    for col in gene_file.columns:
        for idx,gene in enumerate(gene_file[col].notna()):
            if gene == True:
                if col in my_dict:
                    my_dict[col].append(gene_file[col][idx])
                else:
                    my_dict[col]=[gene_file[col][idx]]
    return my_dict


def find_gaps(vlm,index_array_1,inc,index_array_2=None):
    """
    Function that creates a dictionnary of gap spacings. The dictionnary contains
    two keys (start and end). It looks through index_array_1 and records when
    the sequence of index no longer matches (ex: 1,2,3,7,8,9) would result
    in a gap with 'start' as 4 and 'end' as 6. 
    
    index_list_2 is used a precaution in case the gap overlaps with the start point
    Assume a list of 0 to 10, index_list_1 may be (6,7,8,9) which would indicate
    a gap starting at 10 and going to 5. This is possible as we assume the
    dataset to be circular.
    
    This event is unlikely to occur but it is possible
    
    Function written by Yohan Lefol

    Parameters
    ----------
    vlm : velocyto.analysis.VelocytoLoom
        The loom file as read by velocyto.
    index_array_1 : np.ndarray
        A list of indexes that will be analyzed for gaps.
    inc : int
        The number of allowed differences before a gap is 'started'.
    index_array_2 : np.ndarray, optional
        Used to check if the gap goes beyond the start/end point. The default is None.

    Returns
    -------
    gap_dict : dictionnary
        A dictionnary containing two keys (start and end) which represent the
        start and end values of all gaps found in index_list_1.

    """
    index_array_1=sorted(index_array_1)
    gap_dict={}
    gap_dict['start']=[]
    gap_dict['end']=[]
    good_val_found=False
    for idx,val in enumerate(index_array_1):
        if idx==len(index_array_1)-1:#end of list
            if val-inc!=index_array_1[idx-inc]:#Gap on last num
                if index_array_1[0]==1 and index_array_1[-1]==np.max(vlm.ca["new_order"]):
                    gap_dict['start'].append(index_array_1[idx])
                    for dex,item in enumerate(index_array_1):
                        if val+inc!=index_array_1[idx+inc]:#Gap found
                            gap_dict['end'].append(index_array_1[idx])
            else:
                if index_array_1[0]==1 and index_array_1[-1]==np.max(vlm.ca['new_order']):
                    for dex,item in enumerate(index_array_1):
                        if val+inc!=index_array_1[idx+inc]:#Gap found
                            gap_dict['end'].append(index_array_1[idx])
                else:
                    gap_dict['end'].append(index_array_1[idx])
                            
            break
        if val+inc==index_array_1[idx+inc]:
            if good_val_found==False:#First one of series detected
                gap_dict['start'].append(index_array_1[idx])
            good_val_found=True
        else:
            if good_val_found==True:#Else it isn't a gap since there was no good val before hand
                gap_dict['end'].append(index_array_1[idx])
                good_val_found=False
    if index_array_2 !=None:
        for idx,val in enumerate(gap_dict['start']):
            if val-1 in index_array_2[0]: #Means the start point was spliced, no bueno
                #Need to remove it
                del gap_dict['start'][idx]
                del gap_dict['end'][idx]
    
    return gap_dict
    
def create_delay_dataframe(vlm,bin_size=100,window_size=200):
    """
    Function which sorts genes based on any difference detected. If either a
    difference between unspliced and spliced during their increase and/or decrease
    it will be sorted as such

    Parameters
    ----------
    vlm : velocyto.analysis.VelocytoLoom
        The loom file as read by velocyto.
    bin_size : int, optional
        The bin size used for data smoothing. The default is 100.
    window_size: int, optional
        The size of the window used in the moving average for data smoothing.
        The default is 200.

    Returns
    -------
    delay_df : pandas.DataFrame
        A dataframe which contains all the categories and associated gene names.

    """
    
    orientation=np.unique(vlm.ca['orientation'])[0]
    delay_dict={}
    delay_dict['inc_gene_names']=[]
    delay_dict['inc_delay']=[]
    delay_dict['dec_gene_names']=[]
    delay_dict['dec_delay']=[]
    # delay_dict['no_delay_inc_gene_names']=[]
    # delay_dict['no_delay_inc_values']=[]
    # delay_dict['no_delay_dec_gene_names']=[]
    # delay_dict['no_delay_dec_values']=[]
    #Basis for each gene
    for gene_name in vlm.ra["Gene"]:
        # gene_name='TRABD'
        ix=np.where(vlm.ra["Gene"]==gene_name)[0][0]
        # print(ix)
        #smooth velocity lines
        deriv_spli,deriv_unspli=smooth_calcs(vlm,bin_size=bin_size,window_size=window_size,spli_arr=vlm.Sx_sz[ix,:],unspli_arr=vlm.Ux_sz[ix,:],choice='vel')

        #count symbols
        symbols_spli=count_symbols(deriv_spli,orientation=orientation)
        symbols_unspli=count_symbols(deriv_unspli,orientation=orientation)
        
        if orientation=='G2M':#Need to account for x axis reversal in G2M orientation
            symbols_spli=np.asarray(list(reversed(symbols_spli)))
            symbols_unspli=np.asarray(list(reversed(symbols_unspli)))
        
        #Detect difference in increase
        index_check_spli=np.where(symbols_spli<=0)
        index_check_unspli=np.where(symbols_unspli>0)
        
        good_indices=[]
        for idx in index_check_spli[0]:
            if idx in index_check_unspli[0]:
                good_indices.append(idx)
        gap_dict=find_gaps(vlm,good_indices,1,index_check_unspli)
        # print(gap_dict)
        if len(gap_dict['start'])>0:
            start_val_list=[]
            index_check_positive_spli=np.where(symbols_spli>0)
            for idx,start_val in enumerate(gap_dict['start']):
                if gap_dict['end'][idx]+1 in index_check_positive_spli[0]:
                    if gap_dict["end"][idx]<start_val:#It means we went full circle
                        end_val=np.max(good_indices)+gap_dict["end"][idx]
                    else:
                        end_val=gap_dict['end'][idx]
                    start_val_list.append(abs(end_val-start_val))
                    inc_delay_check=True
                else:
                    inc_delay_check=False
        else:
            inc_delay_check=False
        if inc_delay_check==True:
            delay_dict['inc_delay'].append(max(start_val_list))
            delay_dict['inc_gene_names'].append(gene_name)
        else:
            delay_dict['inc_delay'].append(0)
            delay_dict['inc_gene_names'].append(gene_name)
            
            
        #Detect difference in decrease
        index_check_spli=np.where(symbols_spli>=0)
        index_check_unspli=np.where(symbols_unspli<0)

        good_indices=[]
        for idx in index_check_spli[0]:
            if idx in index_check_unspli[0]:
                good_indices.append(idx)
        
        gap_dict=find_gaps(vlm,good_indices,1,index_check_unspli)
        # print(gap_dict)
        if len(gap_dict['start'])>0:
            end_val_list=[]
            index_check_negative_spli=np.where(symbols_spli<0)
            for idx,end_val in enumerate(gap_dict['end']):
                if gap_dict['end'][idx]+1 in index_check_negative_spli[0]:
                    if gap_dict["start"][idx]>end_val:#It means we went full circle
                        start_val=np.max(good_indices)+gap_dict["start"][idx]
                    else:
                        start_val=gap_dict['start'][idx]
                    end_val_list.append(abs(start_val-end_val))
                    dec_delay_check=True
                else:
                    dec_delay_check=False
        else:
            dec_delay_check=False
        if dec_delay_check==True:
            delay_dict['dec_delay'].append(max(end_val_list))
            delay_dict['dec_gene_names'].append(gene_name)
        else:
            delay_dict['dec_delay'].append(0)
            delay_dict['dec_gene_names'].append(gene_name)
        # break#break for testing
    delay_df=pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in delay_dict.items() ]))
    return delay_df

#%%TargetScan functions
def kernel_density_plot(my_dict,xlabel='x'):
    """
    Function that creates kernel density plots based on a given dictionnary

    Parameters
    ----------
    my_dict : dictionnary
        dictionnary expected to contain category names as keys and values should
        be numbers/floats.
    xlabel : string, optional
        A string that will be the x axis label. The default is 'x'.

    Returns
    -------
    None.

    """

    plt.figure(figsize=(7,7))
    for k in my_dict.keys():
        sns.kdeplot(np.asarray(my_dict[k]), label=k+' group')
     
    plt.xlabel(xlabel)
    plt.ylabel("Density function")
    plt.legend()
    plt.show()
    
    
def target_scan_analysis(TS_path,gene_dict,miRNA_list=None):
    """
    Function that uses a targetscan prediction file and creates three dictionnaries
    based on the number of target sites observed, the weight context score of each gene
    and the sum of weighted context for each gene.
    The function automatically filters for miRNAs which are specific to humans, 
    the function can also filter based on a set list of miRNAs.
    
    Function written by Yohan Lefol

    Parameters
    ----------
    TS_path : string
        The file patht to the TargetScan prediction file.
    gene_dict : dictionnary
        A dictionnary containing categoris and gene names.
    miRNA_list : list, optional
        A list of miRNAs that will specify which miRNA target sites will be
        searched for in the TargetScan file. The default is None.

    Returns
    -------
    output_dict : dictionnary
        A dictionnary with the number of target sites per gene.
    output_weight_dict : dictionnary
        A dictionnary with all the weighted context score for each gene.
    sum_weight_dict : dictionnary
        A dictionnary with the sum of the weighted context score for each gene.

    """
    
    TargetScan_file=pd.read_csv(TS_path,sep='\t')
    
    TargetScan_file=TargetScan_file[TargetScan_file['Gene Tax ID']==9606]
    
    if miRNA_list is not None:
        TargetScan_file=TargetScan_file[TargetScan_file['miRNA'].isin(miRNA_list)]

    output_dict={}
    output_weight_dict={}
    sum_weight_dict={}
    for key in gene_dict.keys():
        gene_list=gene_dict[key]
        output_dict[str(key)]={}
        output_weight_dict[str(key)+'_weighted']={}
        sum_weight_dict[key]=[]
        list_no_miRNA=[]
        total_num=0
        total_good_num=0
        for gene in gene_list:
            num_found=len(np.where(TargetScan_file['Gene Symbol']==gene)[0])
            if num_found==0:
                list_no_miRNA.append(gene)
            else:
                total_num=total_num+num_found
                
            if num_found in output_dict[str(key)]:#Key exists
                output_dict[str(key)][num_found].append(gene)
            else:#create key and add gene
                output_dict[str(key)][num_found]=[gene]
            
            subset=TargetScan_file[TargetScan_file['Gene Symbol']==gene]
            sum_weight=sum(subset['weighted context++ score'])
            sum_weight_dict[key].append(sum_weight)
            
            good_targets=len(np.where(subset['weighted context++ score']<=-0.2))
            
            if good_targets in output_weight_dict[str(key)+'_weighted']:#Key existsts
                output_weight_dict[str(key)+'_weighted'][good_targets].append(gene)
            else:#create key and add gene
                output_weight_dict[str(key)+'_weighted'][good_targets]=[gene]
            
            total_good_num=total_good_num+good_targets
        print("########################################################")
        print("For this dict: "+str(key))
        print('Total number of genes: '+str(len(gene_list)))
        print('Number of genes that have targets: '+str(abs(len(gene_list)-len(list_no_miRNA))))
        print('Number of genes that have no targets: '+str(len(list_no_miRNA)))
        print('Total number of targets found for the group: '+str(total_num))
        print('Total number of targets with a weighted context <-0.2: '+str(total_good_num))
        
    return output_dict,output_weight_dict,sum_weight_dict


def target_scan_analysis_version_2(TS_file,gene_dict):
    """
    Function that uses a targetscan prediction file and creates three dictionnaries
    based on the number of target sites observed, the weight context score of each gene
    and the sum of weighted context for each gene.
    The function automatically filters for miRNAs which are specific to humans, 
    the function can also filter based on a set list of miRNAs.
    
    Function written by Yohan Lefol

    Parameters
    ----------
    TS_file : pandas.core.frame.DataFram
        The loaded TargetScan prediction file
    gene_dict : dictionnary
        A dictionnary containing categoris and gene names.
    Returns
    -------
    output_dict : dictionnary
        A dictionnary with the number of target sites per gene.
    output_weight_dict : dictionnary
        A dictionnary with all the weighted context score for each gene.
    sum_weight_dict : dictionnary
        A dictionnary with the sum of the weighted context score for each gene.

    """
    
    TargetScan_file=TS_file

    output_dict={}
    output_weight_dict={}
    sum_weight_dict={}
    for key in gene_dict.keys():
        gene_list=gene_dict[key]
        output_dict[str(key)]={}
        output_weight_dict[str(key)+'_weighted']={}
        sum_weight_dict[key]=[]
        list_no_miRNA=[]
        total_num=0
        total_good_num=0
        for gene in gene_list:
            num_found=len(np.where(TargetScan_file['Gene Symbol']==gene)[0])
            if num_found==0:
                list_no_miRNA.append(gene)
            else:
                total_num=total_num+num_found
                
            if num_found in output_dict[str(key)]:#Key exists
                output_dict[str(key)][num_found].append(gene)
            else:#create key and add gene
                output_dict[str(key)][num_found]=[gene]
            
            subset=TargetScan_file[TargetScan_file['Gene Symbol']==gene]
            sum_weight=sum(subset['weighted context++ score'])
            sum_weight_dict[key].append(sum_weight)
            
            good_targets=len(np.where(subset['weighted context++ score']<=-0.2))
            
            if good_targets in output_weight_dict[str(key)+'_weighted']:#Key existsts
                output_weight_dict[str(key)+'_weighted'][good_targets].append(gene)
            else:#create key and add gene
                output_weight_dict[str(key)+'_weighted'][good_targets]=[gene]
            
            total_good_num=total_good_num+good_targets
        
    return output_dict,output_weight_dict,sum_weight_dict

def convert_gene_to_numbers(the_dict):
    """
    Function that converts the occurence of gene_names to numbers
    
    Function written by Yohan Lefol
    
    Parameters
    ----------
    the_dict : dictionnary
        dictionnary containing gene_names and converts them to numbers. The
        dictionnary is a result of the 'target_scan_analysis' function.

    Returns
    -------
    new_dict : dictionnary
        A dictionnary with gene_name occurence converted to a number.

    """
    new_dict={}
    for key,val in the_dict.items():
        new_dict[key]=[]
        for k,v in val.items():
            for i in range(len(v)):
                new_dict[key].append(k)
    return new_dict


def spearman_analysis(delay_df,TS_path,path_miRNA=None):
    """
    Function that uses the delay dataframe to perform a spearman statistical
    analysis
    
    Function written by Yohan Lefol

    Parameters
    ----------
    delay_df : pandas.core.frame.DataFrame
        The dataframe containing all delay values, obtained from the 
        'create_delay_dataframe" function.
    TS_path : string
        file path to the TargetScan prediction file.
    path_miRNA : string, optional
        file path to the miRNA file. The default is None.

    Returns
    -------
    None.

    """
    
    gene_dict={}
    gene_dict['increase_delay']=[x for x in list(delay_df.inc_gene_names) if str(x) != 'nan']
    gene_dict['decrease_delay']=[x for x in list(delay_df.dec_gene_names) if str(x) != 'nan']
    # gene_dict['no_delay_increase']=[x for x in list(my_delay_df.no_delay_inc_gene_names) if str(x) != 'nan']
    # gene_dict['no_delay_decrease']=[x for x in list(my_delay_df.no_delay_dec_gene_names) if str(x) != 'nan']
    
    if path_miRNA is None:
        miRNA_list=None
    else:    
        miRNA_pd=pd.read_csv(path_miRNA)
        miRNA_list=list(miRNA_pd.found)
        
    TS_dict,TS_weight_dict,sum_weight_dict=target_scan_analysis(TS_path,gene_dict=gene_dict,miRNA_list=miRNA_list)
    #Ensure that the order is the same
    increase_list=[]
    decrease_list=[]
    for key,val in gene_dict.items():
        for gene in val:
            for k,v in TS_dict[key].items():
                if gene in v:
                    if key=='increase_delay':
                        increase_list.append(k)
                    else:
                        decrease_list.append(k)
    
    
    decrease_val=[x for x in list(delay_df.dec_delay) if str(x) != 'nan']
    increase_val=[x for x in list(delay_df.inc_delay) if str(x) != 'nan']
    
    
    import scipy.stats as stats
    print('#########')
    print('increase delay vs number of miRNA target sites')
    print(stats.spearmanr(increase_list,increase_val))
    
    print('#########')
    print('increase delay vs sum of weighted context score ++')
    print(stats.spearmanr(sum_weight_dict['increase_delay'],increase_val))
    
    
    print('#########')
    print('decrease number of miRNA target sites')
    print(stats.spearmanr(decrease_list,decrease_val))
    
    print('#########')
    print('decrease sum of weighted context score ++')
    print(stats.spearmanr(sum_weight_dict['decrease_delay'],decrease_val))
    
    # Below is code for scatter plots
    ##################
    # plt.figure(None,(7.5,5),dpi=80)
    # plt.scatter(increase_list,increase_val)
    # plt.xlabel('Number of Target Sites')
    # plt.ylabel('Delay (increase)')
    # plt.show()
    
    # plt.figure(None,(7.5,5),dpi=80)
    # plt.scatter(sum_weight_dict['increase_delay'],increase_val)
    # plt.xlabel('sum of weights')
    # plt.ylabel('Delay (increase)')
    # plt.gca().invert_xaxis()
    # plt.show()
    
    # plt.figure(None,(7.5,5),dpi=80)
    # plt.scatter(decrease_list,decrease_val)
    # plt.xlabel('Number of Target Sites')
    # plt.ylabel('Delay (decrease)')
    # plt.show()
    
    # plt.figure(None,(7.5,5),dpi=80)
    # plt.scatter(sum_weight_dict['decrease_delay'],decrease_val)
    # plt.xlabel('sum of weights')
    # plt.ylabel('Delay (decrease)')
    # plt.gca().invert_xaxis()
    # plt.show()


def merge_dicts(dict_list):
    """
    Simple function to merge dictionnary on keys

    Parameters
    ----------
    dict_list : list
        a list of dictionnaries to merge.

    Returns
    -------
    merged_dict : dictionnary
        The merged dictionnary.

    """
    merged_dict={}
    for the_dict in dict_list:
        for key,val in the_dict.items():
            if key not in merged_dict.keys():
                merged_dict[key]=[val]
            else:
                merged_dict[key].append(val)
    return merged_dict
    
#%%miRNA list preparation

def rpm_normalization(df_path):
    """
    Perform rpm normalization and subsequent average calculation on dataframe.
    This function selects for columns that are specific to our file.
    May require modification on a case by case basis.

    Parameters
    ----------
    df_path : string
        path to the dataframe that will be rpm normalized.

    Returns
    -------
    my_means : pandas.core.series.Series
        The average rpm of each miRNA in the dataframe.

    """
    my_matrix=pd.read_csv(df_path, sep='\t')
    #Start by keeping only S columns
    filter_col = [col for col in my_matrix if col.startswith('s') or col.endswith('S')]
    my_matrix=my_matrix[filter_col]
    
    my_means=my_matrix
    for col in my_matrix.columns:
        scaling_factor=sum(my_matrix[col])
        for idx,val in enumerate(my_matrix[col]):
            my_means[col][idx]=val*1000000/scaling_factor
    

    my_means=my_means.mean(axis=1)
    return my_means

def categorize_findings(TS_path,rpm_df,miRNA_thresh=None):
    """
    Categorizes the list of miRNAs based on the results of the TargetScan file
    Sorted based on if it was found, the ones not found are split into dedicated
    categories for further sorting.
    The miRNA list can be filtered according to an rpm

    Parameters
    ----------
    TS_path : string
        file path to the TargetScan prediction file
    rpm_df : pandas.core.series.Series
        The average rpm of each miRNA in the dataframe.
    miRNA_thresh : int, optional
        value for which miRNA rpms are filtered. Only miRNA with a value equal
        or superior to the threshold are kept. The default is None.

    Returns
    -------
    main_dict : dictionnary
        A dictionnary containing all miRNAs of the list in their respective category.

    """
    main_dict={}
    main_dict['found']=[]
    main_dict['not_found_all']=[]
    main_dict['not_found_accounted_for']=[]
    main_dict['not_found_not_p']=[]
    main_dict['not_found_single_p']=[]
    main_dict['not_found_both_p']=[]
    
    TS_file=pd.read_csv(TS_path,sep='\t')
    TS_file=TS_file[TS_file['Gene Tax ID']==9606]
    
    #Filter for RPM desired
    miRNA_list=[]
    for idx,val in rpm_df.iteritems():
        if val>=miRNA_thresh:
            miRNA_list.append(idx)
    
    
    for i in miRNA_list:
        if i in TS_file.miRNA.unique():
            main_dict['found'].append(i)
        else:
            main_dict['not_found_all'].append(i)
    found_remove_p=[] 
    for i in main_dict['found']:
        if i.endswith('p'):
            found_remove_p.append(i[:-3]) 
    
    not_found_remove_p=[]
    for i in main_dict['not_found_all']:
        if i.endswith('p'):
            if i[:-3] not in found_remove_p:
                not_found_remove_p.append(i[:-3])
                main_dict['not_found_single_p'].append(i)
        else:
            main_dict['not_found_not_p'].append(i)
        
    from collections import Counter
    for k,v in Counter(not_found_remove_p).items():
        if v>1:
            main_dict['not_found_both_p'].append(k+'-3p')
            main_dict['not_found_both_p'].append(k+'-5p')
            main_dict['not_found_single_p'].remove(k+'-3p')
            main_dict['not_found_single_p'].remove(k+'-5p')
    
    
    for i in main_dict['not_found_single_p']:
        if i[-3:]=='-3p':
            if i[:-3]+'-5p' not in TS_file.miRNA.unique():
                continue
            else:
                main_dict['not_found_single_p'].remove(i)
        else:
            if i[:-3]+'-3p' not in TS_file.miRNA.unique():
                continue
            else:
                main_dict['not_found_single_p'].remove(i)
    
    
    for i in main_dict['not_found_all']:
        if i not in main_dict['not_found_not_p'] and i not in main_dict['not_found_single_p'] and i not in main_dict['not_found_both_p']:
            main_dict['not_found_accounted_for'].append(i)
    
    return main_dict


def check_miRNA_family(miRNA_dict,miR_family_path, TS_path):
    """
    Verifies if the not found miRNAs have a subscript format.
    If this is the case, the will be sorted back into the accounted for category.

    Parameters
    ----------
    miRNA_dict : dictionnary
        the list of sorted miRNAs from the categorize_findings function.
    miR_family_path : string
        The path to the miRNA family document.
    TS_path : string
        The path to the TargetScan prediction file

    Returns
    -------
    miRNA_dict : dictionnary
        a newly sorted version of the miRNAs.

    """
    miR_family=pd.read_csv(miR_family_path)
    TS_file=pd.read_csv(TS_path,sep='\t')
    TS_file=TS_file[TS_file['Gene Tax ID']==9606]
    for key in ['not_found_not_p', 'not_found_single_p', 'not_found_both_p']:
        for miR in miRNA_dict[key]:
            if type(miR)==float:
                continue
            miR_split=miR.split('.')[0]
            miR_split=miR_split.split('hsa-')[1]
            found_indices=np.where(miR_family['Human microRNA family'].str.contains(miR_split)==True)
            if len(found_indices[0])>0:
                for idx in found_indices[0]:
                    if type(miR_family.miRNAs[idx])==float:
                        continue
                    for val in miR_family.miRNAs[idx].replace('\xa0',' ').split():
                        if miR in val:
                            if re.search(r'\.\d',val) is not None:
                                if val in list(TS_file.miRNA):
                                    miRNA_dict['found'].append(val)
                                    miRNA_dict[key].remove(miR)
                                    miRNA_dict['not_found_all'].remove(miR)
                                    if key=='not_found_both_p':
                                        if miR.endswith('-3p')==True:
                                            no_p_miR=miR.split('-3p')[0]
                                            miRNA_dict['not_found_both_p'].remove(no_p_miR+'-5p')
                                            miRNA_dict['not_found_accounted_for'].append(no_p_miR+'-5p')
                                        else:
                                            no_p_miR=miR.split('-5p')[0]
                                            miRNA_dict['not_found_both_p'].remove(no_p_miR+'-3p')
                                            miRNA_dict['not_found_accounted_for'].append(no_p_miR+'-3p')                            
    return miRNA_dict





def check_non_conserved(miRNA_dict, non_conserved_path):
    """
    Checks if missing miRNAs are found in the non_conserved target scan file

    Parameters
    ----------
    miRNA_dict : Dictionnary
        A sorted dictionnary of miRNAs.
    non_conserved_path : string
        path to the non-conserved target scan file.

    Returns
    -------
    my_df : pandas.core.frame.DataFrame
        A dataframe with the finalized categorization.

    """
    TS_file_non_conserved=pd.read_csv(non_conserved_path,sep='\t')
    
    TS_file_non_conserved=TS_file_non_conserved[TS_file_non_conserved['Gene Tax ID']==9606]
    
    
    new_dict={}
    new_dict['found']=miRNA_dict['found']
    new_dict['not_found_all']=miRNA_dict['not_found_all']
    new_dict['not_found_accounted_for']=miRNA_dict['not_found_accounted_for']
    new_dict['not_found_not_accounted_for']=[]
    for key in ['not_found_not_p', 'not_found_single_p', 'not_found_both_p']:
        for miR in miRNA_dict[key]:
            found=False
            if miR not in TS_file_non_conserved.miRNA.unique():
                if miR+'.1' not in TS_file_non_conserved.miRNA.unique() or miR+'.2' not in TS_file_non_conserved.miRNA.unique():
    
                    new_dict['not_found_not_accounted_for'].append(miR)
                    found=True
            if found==False:
                new_dict['not_found_accounted_for'].append(miR)
                
        
    my_df=pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in new_dict.items() ]))
    return my_df



