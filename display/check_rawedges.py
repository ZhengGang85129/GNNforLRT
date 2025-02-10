#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import torch




def calc_total_edges(hits = None):
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
    result = []
    for z_range in [(-2.0, -0.5), (-0.5, 0.5), (0.5, 2.0)]:
        r_ranges.append((0.0, 0.2))
        z_ranges.append(z_range)
    for r_range, z_range in zip(r_ranges, z_ranges):
        result.append(edges_in_radius(hits, r_range, z_range))
        
    return result   


if __name__ == '__main__':

    e_ids = 1000
    raw_data_file = '/global/cfs/cdirs/m3443/data/GNNforLRT/trackPt1GeV-smeared/TTbar_DiLep_output_PU200_NPZ/REPLACE.npz'
    fig, axs = plt.subplots(figsize = (10, 8))
    nhits = {
        'left': [],
        'middle': [],
        'right': []
    }
    
     
    for e_id in range(e_ids): 
        result = torch.load(raw_data_file.replace('REPLACE', str(e_id)))
        true_edge = 'layerless_true_edges'
        true_graph = torch.cat((result[true_edge].T, result[true_edge].flip(0).T)).T
        true_graph = true_graph.permute(1, 0)
        true_graph = pd.DataFrame(true_graph.tolist(), columns = ['hid', 'Reciever']).sort_values(by = 'hid', ascending = True)
        spatial = pd.DataFrame(result['x'].tolist(), columns = ['r', 'phi', 'z']).reset_index().rename(columns = {'index': 'hid'})
        hits = true_graph.merge(spatial, on = 'hid', how = 'left')
        result = calc_total_edges(hits)
        nhits['left'].append(result[0])
        nhits['middle'].append(result[1])
        nhits['right'].append(result[2])
        print(e_id)
    plt.grid(True, alpha=0.3) 
    bin_width = 100
    bin_end = 2000
    bin_beg = 0
    n_bins = int(np.ceil((bin_end - bin_beg) / bin_width))
    bins = np.linspace(bin_beg, bin_end, n_bins + 1)
    axs.hist(nhits['left'], bins = bins, edgecolor = 'black', label = 'left', alpha = 0.3) 
    axs.hist(nhits['right'], bins = bins, edgecolor = 'black', label = 'right', alpha = 0.3)
    axs.set_title('Total layerless_true_edges') 
    axs.legend(loc = 'upper right')
    axs.set_xlim([bin_beg, bin_end])
    fig.savefig('check-hits.png')
#print(true_graph.shape)

#print(torch.cat((test_tensor.T, test_tensor.flip(0).T)).T)

