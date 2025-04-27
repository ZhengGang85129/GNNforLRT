from pathlib import Path
import numpy as np
import pandas as pd
from ExaTrkXDataIO import DataReader
import matplotlib.pyplot as plt
import sys, os 
import torch

def calc_total_edges(hits = None, edge_cut: float = 0):
    def edges_in_radius(hits, r_range,z_range):
        '''
        Consider the edge is single direction.
        '''
        x_idxs = np.nonzero(np.logical_and.reduce((
        hits.r/1000 > r_range[0],
        hits.r/1000 <= r_range[1],
        hits.z/1000 > z_range[0],
        hits.z/1000 <= z_range[1])))
        return len(x_idxs[0])

    r_ranges = []
    z_ranges = []

    # Pixel
    for z_range in [(-2.0, -0.5), (-0.5, 0.5), (0.5, 2.0)]:
        r_ranges.append((0.0, 0.2))
        z_ranges.append(z_range)
    results = dict()
    results['constructed'] = [] 
    results['truth'] = []
    results['intersect'] = []
    results['total'] = []
     
    for r_range, z_range in zip(r_ranges, z_ranges):
        n_constructed_edges_in_graph = edges_in_radius(hits[hits['score'] >  edge_cut], r_range, z_range)
        n_truth_edges_in_graph = edges_in_radius(hits[hits['truth']], r_range, z_range)
        n_intersection_edges_in_graph = edges_in_radius(hits[hits['truth']][hits['score'] >  edge_cut], r_range, z_range)
        n_total_edges_in_graph = edges_in_radius(hits, r_range, z_range)
        results['constructed'].append(n_constructed_edges_in_graph)
        results['truth'].append(n_truth_edges_in_graph)
        results['intersect'].append(n_intersection_edges_in_graph)
        
        results['total'].append(n_total_edges_in_graph) 
    return results    




if __name__ == '__main__':
    
    if len(sys.argv) != 3:
        raise RuntimeError('usage: python3 ./tracks/check_rawhits.py <configuration file> <gnn_output>')
    
    
    edge_cut =  0.333
    path = Path(f'{sys.argv[1]}')
    base_dir = Path(f'{os.path.basename(sys.argv[1])}')
    gnn_output = Path(f'{sys.argv[2]}') #gnn_output: XXX/train/0000 , for example
    event_ids = range(0, 5000)
    
    reader = DataReader(
        config_path=path,
        base_dir=base_dir
    )
    
    #results = dict()
    #results['constructed'] = [] 
    #results['truth'] = []
    #results['intersect'] = []
    constructed = dict()
    truth = dict()
    intersect = dict() 
    total = dict()
    nhits = dict()
    for name in ['left', 'middle', 'right']:
        constructed[name] = []
        truth[name] = []
        intersect[name] = []
        nhits[name] = []
        total[name] = []
    
    fig, axs = plt.subplots(2, 2, figsize = (10, 8))
    #fig, axs = plt.subplots(figsize = (10, 8))
     
    run = 500
    cnt = 0 
    for event_id in event_ids:
        data = reader.read_one(evtid = event_id)
        if data is None: continue
        cnt += 1
        hits = data['hits'].reset_index().rename(columns = {'index': 'hit_id'})
        edges = data['edges']
        
        A = torch.load(gnn_output).to('cpu')
        
        if cnt == run: 
            break
        edges.columns = ['hit_id', 'receiver', 'truth', 'score']
        
        hits = edges.merge(hits, how = 'left', on = 'hit_id')

        result = calc_total_edges(hits = hits,
                                   edge_cut = edge_cut) 
        constructed['left'].append(result['constructed'][0])
        constructed['middle'].append(result['constructed'][1])
        constructed['right'].append(result['constructed'][2])
        
        truth['left'].append(result['truth'][0])
        truth['middle'].append(result['truth'][1])
        truth['right'].append(result['truth'][2])
        
        intersect['left'].append(result['intersect'][0])
        intersect['middle'].append(result['intersect'][1])
        intersect['right'].append(result['intersect'][2])
        total['left'].append(result['total'][0])
        total['middle'].append(result['total'][1])
        total['right'].append(result['total'][2])
        if cnt == run: 
            break
    plt.grid(True, alpha=0.3) 
    bin_width = 100
    bin_end = 1200
    bin_beg = 0
    n_bins = int(np.ceil((bin_end - bin_beg) / bin_width))
    bins = np.linspace(bin_beg, bin_end, n_bins + 1)
    axs[0, 0].hist(intersect['left'], bins = bins, edgecolor = 'black', label = 'left', alpha = 0.3) 
    axs[0, 0].hist(intersect['right'], bins = bins, edgecolor = 'black', label = 'right', alpha = 0.3)
    axs[0, 0].set_title('Intersection') 
    axs[0, 0].legend(loc = 'upper right')
    axs[0, 0].set_xlim([bin_beg, bin_end])
    axs[0, 1].hist(constructed['left'], bins = bins, edgecolor = 'black', label = 'left', alpha = 0.3) 
    axs[0, 1].hist(constructed['right'], bins = bins, edgecolor = 'black', label = 'right', alpha = 0.3) 
    axs[0, 1].legend(loc = 'upper right')
    axs[0, 1].set_title('Constructed') 
    axs[0, 1].set_xlim([bin_beg, bin_end])
    axs[1, 0].hist(truth['left'], bins = bins, edgecolor = 'black', label = 'left', alpha = 0.3) 
    axs[1, 0].hist(truth['right'], bins = bins, edgecolor = 'black', label = 'right', alpha = 0.3) 
    axs[1, 0].set_title('Truth') 
    axs[1, 0].legend(loc = 'upper right')
    axs[1, 0].set_xlim([bin_beg, bin_end])
    #axs[0, 0].hist(intersect['middle'], bins = 30, edgecolor = 'black') 
    bin_width = 100
    bin_end = 12000
    bin_beg = 0
    n_bins = int(np.ceil((bin_end - bin_beg) / bin_width))
    bins = np.linspace(bin_beg, bin_end, n_bins + 1)
    axs[1, 1].hist(total['left'], bins = bins, edgecolor = 'black', label = 'left', alpha = 0.3) 
    axs[1, 1].hist(total['right'], bins = bins, edgecolor = 'black', label = 'right', alpha = 0.3) 
    axs[1, 1].set_title('Total') 
    axs[1, 1].legend(loc = 'upper right')
    axs[1, 1].set_xlim([bin_beg, bin_end])
    fig.savefig('check-hits.png')