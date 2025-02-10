#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import torch
import os



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

    e_ids = 5000
    stage = 'GNN'
    raw_data_file = f'/global/cfs/cdirs/m3443/data/GNNforLRT/results/TTbar_PU200_{stage}_output-smeared/train/REPLACE'
    fig, axs = plt.subplots(figsize = (10, 8))
    nhits = {
        'left': [],
        'middle': [],
        'right': []
    }
    run = 500
    cnt = 0 
    for e_id in range(e_ids): 
        if not os.path.isfile(raw_data_file.replace('REPLACE', str(e_id))): continue 
        result = torch.load(raw_data_file.replace('REPLACE', str(e_id))).to('cpu')
        print(f'Read({cnt+1})-> ', raw_data_file.replace('REPLACE', str(e_id))) 

        edge = 'edge_index'
        graph = result[edge]
        raise ValueError(result['hid'])
        #graph = torch.cat((result[edge].T, result[edge].flip(0).T)).T
        truth = result['truth']
        score = result['score']
        #truth = torch.cat([truth.T, truth.T]) 
        #raise ValueError(torch.cat([truth.T, truth.T]).shape) 
        graph = graph.permute(1, 0)
        graph = pd.DataFrame(graph.tolist(), columns = ['hid', 'Reciever']).sort_values(by = 'hid', ascending = True)
        spatial = pd.DataFrame(result['x'].tolist(), columns = ['r', 'phi', 'z']).reset_index().rename(columns = {'index': 'hid'})
        truth = pd.DataFrame(truth, columns = ['truth'])
        score = pd.DataFrame(score, columns = ['score'])
        #graph = pd.merge(graph, truth, how = 'left') 
        graph = pd.concat([graph, truth, score], axis = 1, ignore_index = True)
        graph.columns =  ['hid', 'Reciever', 'truth', 'score']
        hits = graph[graph['truth']].merge(spatial, on = 'hid', how = 'left')
        result = calc_total_edges(hits)
        nhits['left'].append(result[0])
        nhits['middle'].append(result[1])
        nhits['right'].append(result[2])
        #print(nhits)
        cnt += 1
        if cnt == run:break 

    plt.grid(True, alpha=0.3) 
    bin_width = 100#1000#5000
    bin_end = 1200#15000#150000
    bin_beg = 0
    n_bins = int(np.ceil((bin_end - bin_beg) / bin_width))
    bins = np.linspace(bin_beg, bin_end, n_bins + 1)
    axs.hist(nhits['left'], bins = bins, edgecolor = 'black', label = 'left', alpha = 0.3) 
    axs.hist(nhits['right'], bins = bins, edgecolor = 'black', label = 'right', alpha = 0.3)
    axs.set_title(f'Truth edges({stage})') 
    axs.legend(loc = 'upper right')
    axs.set_xlim([bin_beg, bin_end])
    fig.savefig(f'check-edges_{stage}.png')

