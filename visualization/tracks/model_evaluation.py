import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd
from typing import List, Union, Tuple
from ExaTrkXDataIO import DataReader

import matplotlib.pyplot as plt

from workflows import reconstruct_and_match_tracks as DBSCAN
from tracks.track_reconstruction.algorithm.Wrangler import reconstruct_and_match_tracks_visualization as Wrangler

from plot_configurations import (
    particle_filters,
)
from functools import partial
import os
import argparse
def myParse() -> argparse.ArgumentParser:
    '''
    usage: 
    (1) Extract the generated/reconstructable/matched particles from GNN and DBSCAN stage
        -> python3 ./tracks/evaluate_on_eff.py --config <PathToYourConfig> --output metrics/final<or Your specified Path> --mode extract --lepton <prompt/displaced/all> --fname <Specify Whatever you like>
    (2) Evaluate on the model
        -> python3 ./tracks/evaluate_on_eff.py --config <PathToYourConfig> --output metrics/final<or Your specified Path> --mode evaluate --lepton <prompt/displaced/all> --fname <Specify Whatever you like>
    '''
    
    parser = argparse.ArgumentParser(description = 'parser for evaluation_of_eff')
    parser.add_argument('--config', type = str, default = None)
    parser.add_argument('-d', '--output', type = str, default = 'metrics/final')
    parser.add_argument('-o', '--fname', type = str, default = 'test')
    parser.add_argument('-m', '--mode', type = str, default = 'extract', choices = ['extract', 'evaluate', 'merge']) 
    parser.add_argument('--lepton', choices = ['prompt', 'displaced', 'all', 'HSS'], default = 'prompt')
    parser.add_argument('--merge_list' )
    parser.add_argument('-a', '--algorithm', choices = ['DBSCAN', 'Wrangler'], default = 'Wrangler')
    args = parser.parse_args()
    #assert args.config is not None
    return args

def put_text(ax, sample_name = 'ttbar', pileup = 'flat'):
    ax.text(0.05, 0.77, r'ExatrkX', transform = ax.transAxes, fontsize = 25 )
    #if pileup == 'flat':
    #    ax.text(0.05, 0.72, r'Evalutated on $t\bar{t}$ events', transform = ax.transAxes, fontsize = 15)
    #else:
    if pileup != 'flat':
        ax.text(0.05, 0.72, rf"Evalutated on process with $\langle \mu \rangle = {pileup}$", transform = ax.transAxes, fontsize = 16)
    ax.text(0.05, 0.72, r'$\mathbf{Open \  data \ detector} (Fastras sim.)$', transform = ax.transAxes, fontsize = 18)
    ax.text(0.72, 0.77, r'$\sqrt{s} = 14 \  \mathbf{TeV}$', transform = ax.transAxes, fontsize = 20) 
    ax.text(0.05, 0.68, r'Walkthrough algo. with $\omega_{1} = 0.3$ and $\omega_{2} = 0.8.$', transform = ax.transAxes, fontsize = 18 )

    #.legend(title='abc xyz')

def plot_eff_2D(data):
    fig, ax = plt.subplots(1, 1, figsize = (8, 8), tight_layout = True)
    
    X_reco = data['reco']['eta']
    Y_reco = data['reco']['pt']
    X_gen = data['gen']['eta']
    Y_gen = data['gen']['pt']
    
    plt.hist2d(X_gen, Y_gen, bins = 10)
    
    #plt.savefig('gen.png') 
     

def plot_eff(data, plt_config, var_name = 'pt')->None:
    var_names = ['pt', 'eta', 'd0', 'z0', 'vr', 'npileup']
    if var_name not in var_names: raise RuntimeError(f'No corresponding plot setting for this variable name: {var_name}') 
    
    fig, ax = plt.subplots(1, 1, figsize = (8, 8), tight_layout = True)
    #ax.hist(data['gen'][var_name], bins = plt_config['bins'], edgecolor = 'black')
    
    Types = ['gen', 'reco', 'matched']
    counts = {}
    bin_edges = {}
    bin_centers = {}
    bin_width = {}  
    
    for Type in Types:
        counts[Type], bin_edges[Type], _ = ax.hist(data[Type][var_name], bins = plt_config['bins'])
        bin_centers[Type] = (bin_edges[Type][:-1] + bin_edges[Type][1:])/2
        bin_width[Type] = (bin_edges[Type][1:] - bin_edges[Type][:-1])/2
    ratio = {'tech': None, 'phys': None}
    error = {'tech': None, 'phys': None}
    
    ratio['phys'] = np.divide(counts['matched'], counts['gen'],  where = (counts['gen']!=0))
    ratio['tech'] = np.divide(counts['matched'], counts['reco'],  where = (counts['reco']!=0))
    ratio['tech'][counts['reco'] == 0] = 0
    ratio['phys'][counts['gen'] == 0] = 0
    
    print(sum(counts['reco']), sum(counts['gen']))
     
    error['phys'] = np.sqrt(ratio['phys'] * (1 - ratio['phys'])/counts['gen'])
    error['phys'] = np.nan_to_num(error['phys'], nan = 0, posinf = 0, neginf = 0)
    error['tech'] = np.sqrt(ratio['tech'] * (1 - ratio['tech'])/counts['reco']) 
    error['tech'] = np.nan_to_num(error['tech'], nan = 0, posinf = 0, neginf = 0)
    
    ax.clear()
    _, ymax = ax.get_ylim()
    ax.set_xscale(plt_config['ax_opts']['xscale'])
    ax.set_ylim(0, 1.5 * ymax)
    ax.legend() 
    put_text(ax)
    #ax.errorbar(bin_centers['gen'], ratio['phys'], yerr = error['phys'], fmt = 'o', label = 'Physical efficacy', xerr = bin_width['gen'])
    
    ax.errorbar(bin_centers['reco'], ratio['tech'], yerr = error['tech'], fmt = 'o', label = 'Techincal efficacy', xerr = bin_width['reco'])
    ax.legend() 
    ax.set_xlabel(plt_config['x_label'])
    ax.grid(True, linestyle = '--', alpha = 0.7)
    
    plt_name = f'{args.output}/{var_name}-{args.lepton}_{args.fname}_eff.png' 
    fig.savefig(plt_name) 
    fig.savefig(plt_name.replace('png', 'pdf')) 
    print(f'Check: {plt_name}') 
    
    return 

def create_log_bins(min_val, max_val, threshold, n_bins_below, n_bins_above):
    log_min = np.log10(min_val)
    log_max = np.log10(max_val)
    log_threshold = np.log10(threshold)
    bins_below = np.linspace(log_min, log_threshold, n_bins_below)
    
    # Create dense bins above threshold in log space
    bins_above = np.linspace(log_threshold, log_max, n_bins_above + 1)[1:]
    
    # Combine the bins
    bins = np.unique(np.concatenate([bins_below, bins_above]))
    log_bins = np.concatenate([bins_below, bins_above])
    bins = 10**log_bins 
    return bins

def plot(data, plt_config, var_name = 'pt')->None:
    var_names = ['pt', 'eta', 'd0', 'z0', 'vr', 'npileup']
    if var_name not in var_names: raise RuntimeError(f'No corresponding plot setting for this variable name: {var_name}') 
    
    fig, ax = plt.subplots(1, 1, figsize = (8, 8), tight_layout = True)
    #ax.hist(data['gen'][var_name], bins = plt_config['bins'], edgecolor = 'black')
    
    Types = ['gen', 'reco', 'matched']
    counts = {}
    bin_edges = {}
    bin_centers = {}
    bin_width = {}  
    label_names = ['Generated', 'Reconstructable', 'Matched']
    #<tab>
    transparencies = [0.0, 0., 0.]
    colors = ['red', 'blue', 'green'] 
    #plt.margins(y=0.25) 
    for Type, label_name, alpha, color in zip(Types, label_names, transparencies, colors):
        ax.hist(data[Type][var_name], bins = plt_config['bins'], label = label_name, facecolor = 'none', edgecolor = color)

    _, ymax = ax.get_ylim()
    #ax.set_ylim(0, 1.5 * ymax)
    ax.set_xscale(plt_config['ax_opts']['xscale'])
    ax.set_yscale(plt_config['ax_opts']['yscale'])
    ax.legend() 
    ax.set_xlabel(plt_config['x_label'])
    ax.grid(True, linestyle = '--', alpha = 0.7)
    
    plt_name = f'{args.output}/{var_name}-{args.lepton}_{args.fname}.png' 
    fig.savefig(plt_name) 
    print(f'Check: {plt_name}') 
    
    return 

    

def extract():
    with multiprocessing.Pool(processes = 8) as pool:
        reader = DataReader(
            config_path = path,
            base_dir = base_dir
        )
        if args.algorithm == 'DBSCAN':
            algo = partial(eval(args.algorithm), epsilon = 0.14, statistics = True)
        
            
        elif args.algorithm == 'Wrangler':
            algo = partial(eval(args.algorithm), filter_cut = 0.1, walk_min = 0.4, walk_max = 0.95, statistics = True)
        else:
            raise RuntimeError(f'No such algorithm: {args.algorithm}')
        results = pool.map(algo, reader.read())
    particles, hits_df, constructed_tracks_df, edges_df, matched_tracks_df = aggregate_results(results)
    particles = particles[particle_filters[args.lepton](particles)]
    particles.to_csv(os.path.join(args.output, args.fname+f'_gen-{args.lepton}.csv'), index = False)
    particles[particles.is_trackable].to_csv(os.path.join(args.output, args.fname+f'_reco-{args.lepton}.csv'), index = False)
    particles[particles.is_trackable & particles.is_matched].to_csv(os.path.join(args.output, args.fname+f'_match-{args.lepton}.csv'), index = False)
    #print(particles)
    #raise ValueError
    return

def aggregate_results(results: List[Union[pd.DataFrame, float]]) -> Tuple[pd.DataFrame, float, float]:
    
    dataframes = []
    hits = []
    constructed_tracks = []
    matched_tracks = []
    edges = [] 
    for df, statistics in results:
        dataframes.append(df)
        hits.append(statistics['hits'])
        constructed_tracks.append(statistics['constructed_tracks'])
        if statistics['edges'] is not None:
            edges.append(statistics['edges'])
        matched_tracks.append(statistics['matched_track'])
    dataframes = pd.concat(dataframes, ignore_index=True)
    hits = pd.concat(hits, ignore_index=True)
    constructed_tracks = pd.concat(constructed_tracks, ignore_index=True)
    if edges != []:
        edges = pd.concat(edges, ignore_index=True)
    matched_tracks = pd.concat(matched_tracks, ignore_index = True)
    return dataframes, hits, constructed_tracks, edges, matched_tracks 
def merge():
    plt_configs = {
        'pt':{
        'x_label': r'$p_{T}$ [GeV]',
        'bins': np.array([0,15, 30, 60, 90, 120, 150]),
        'ax_opts': {
            'xscale': 'linear',
            'yscale': 'log'
        }
    },
        'eta':{
            'x_label': r'$\eta$',
            'bins': [-3, -2.7, -2.4, -2.1, -1.8, -1.5, -1.2, -0.9, -0.6, -0.3, 0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0],
            'ax_opts': {
                'xscale': 'linear',
                'yscale': 'linear'
            }
        },
        'd0':{
            'x_label': r'$d0 [mm]$',
            'bins': create_log_bins(0.01, 10**3, 1, 2, 10),
            'ax_opts': {
                'xscale': 'log',
                'yscale': 'log'
            }
            
        },
        'z0':{
            'x_label': r'$z0 [mm]$',
            'bins': np.concatenate([-np.logspace(np.log10(1), np.log10(1000), 5)[::-1], [0 ], np.logspace(np.log10(1), np.log10(1000), 5)]),
            'ax_opts': {
                'xscale': 'symlog',
                'yscale': 'linear'
            }
            
        },
        'vr':{
            'x_label': r'$v_{r} [cm]$',
            'bins': np.array([0, 12, 25, 37, 50, 65, 100, 150, 250]),
            'ax_opts': {
                'xscale': 'linear',
                'yscale': 'linear'
            }
            
        },
        'npileup':{
            'x_label': r'$N_{PU}$',
            'bins': 10,
            'ax_opts': {
                'xscale': 'linear',
                'yscale': 'linear'
            }
            
        }
    }
    
    var_names = ['npileup']
    
    dataframe = pd.read_csv(args.merge_list, header = None)
    dataframe.columns = ['prefix', 'lepton_type', 'label', 'dataset']
    
    
    particles = dict()
    particles['reco'] = dict()
    particles['match'] = dict()
    particles['gen'] = dict()
     
    
    for index in range(len(dataframe)):
        prefix = dataframe.iloc[index]['prefix']
        lepton_type = dataframe.iloc[index]['lepton_type']
        gen_path = os.path.join('metrics', f'{prefix}_gen-{lepton_type}.csv')
        match_path = os.path.join('metrics', f'{prefix}_match-{lepton_type}.csv') 
        reco_path = os.path.join('metrics', f'{prefix}_reco-{lepton_type}.csv') 
        if not os.path.isfile(gen_path) or not os.path.join(reco_path) or not os.path.isfile(match_path):
            print(f'check: {reco_path}\n'
                  f'{gen_path}\n'
                  f'{match_path}')
            continue 
        
        particles['reco'][prefix] = pd.read_csv(os.path.join('metrics', f'{prefix}_reco-{lepton_type}.csv')) 
        particles['match'][prefix] = pd.read_csv(os.path.join('metrics', f'{prefix}_match-{lepton_type}.csv')) 
        particles['gen'][prefix] = pd.read_csv(os.path.join('metrics', f'{prefix}_gen-{lepton_type}.csv')) 
         
    var_names = ['npileup', 'pt', 'eta', 'vr', 'z0', 'd0']
    
    for var_name in var_names:    
        fig, ax = plt.subplots(1, 1, figsize = (8, 8), tight_layout = True) 
        fig1, ax1 = plt.subplots(1, 1, figsize = (8, 8), tight_layout = True) 
        
        for index in range(len(dataframe)):
            prefix = dataframe.iloc[index]['prefix']
            label = dataframe.iloc[index]['label']
            dataset = dataframe.iloc[index]['dataset'] 
            if particles['reco'].get(prefix) is None:
                print(f'Skip: {prefix}')
                continue
            if particles['reco'].get(prefix) is None:continue
            if particles['match'].get(prefix) is None:continue
            if particles['reco'][prefix].get(var_name) is None: break
             
            counts = {}
            bin_edges = {}
            bin_centers = {}
            bin_width = {}
            #print(plt_configs[var_name]['bins'])
            for Type in particles.keys():
                counts[Type], bin_edges[Type], _ = ax.hist(particles[Type][prefix][var_name], bins = plt_configs[var_name]['bins'])
                bin_centers[Type] = (bin_edges[Type][:-1] + bin_edges[Type][1:])/2
                bin_width[Type] = (bin_edges[Type][1:] - bin_edges[Type][:-1])/2
                #print(Type, sum(counts[Type]), var_name)
            ratio = {'tech': None, 'phys': None}
            error = {'tech': None, 'phys': None}
            
            ratio['tech'] = np.divide(counts['match'], counts['reco'],  where = (counts['reco']!=0))
            ratio['tech'][counts['reco'] == 0] = 0
            
            error['tech'] = np.sqrt(ratio['tech'] * (1 - ratio['tech'])/counts['reco']) 
            error['tech'] = np.nan_to_num(error['tech'], nan = 0, posinf = 0, neginf = 0)
            
            ratio['phys'] = np.divide(counts['match'], counts['gen'],  where = (counts['gen']!=0))
            ratio['phys'][counts['gen'] == 0] = 0
            
            error['phys'] = np.sqrt(ratio['phys'] * (1 - ratio['phys'])/counts['gen']) 
            error['phys'] = np.nan_to_num(error['phys'], nan = 0, posinf = 0, neginf = 0)
            label_name = label
            if 'ttbar' in dataset:
                label_name += r' evaluated on $t\bar{t}$ dataset'
            elif 'HSS' in dataset:
                label_name += r' evaluated on $HSS$ dataset'
            else:
                label_name += r' evaluated on HNL dataset'
            mask = counts['match'] > 0  
            ax1.errorbar(bin_centers['reco'][mask], ratio['tech'][mask], 
                         yerr = error['tech'][mask], fmt = 'o', label = label_name, xerr = bin_width['reco'][mask])
            
        #ax.clear()
        _, ymax = ax1.get_ylim()
        ax1.set_xscale(plt_configs[var_name]['ax_opts']['xscale'])
        ax1.set_ylim(0, 2 * ymax)
        put_text(ax1)
        ax1.legend(loc = 'upper right') 
        
        ax1.set_xlabel(plt_configs[var_name]['x_label'], loc = 'right', fontsize = 18)
        ax1.set_ylabel('Technical Efficiency', loc = 'top', fontsize = 18)
        ax1.grid(True, linestyle = '--', alpha = 0.7)
        
        plt_name = f'{args.output}/{var_name}-merge.png' 
        fig1.savefig(plt_name) 
        fig1.savefig(plt_name.replace('png', 'pdf')) 
        print(f'Check: {plt_name}') 
    
    return 
        
        
 
def evaluate():
    gen_particles = pd.read_csv(os.path.join(args.output, args.fname+f'_gen-{args.lepton}.csv'))
    reco_particles = pd.read_csv(os.path.join(args.output, args.fname+f'_reco-{args.lepton}.csv'))
    match_particles = pd.read_csv(os.path.join(args.output, args.fname+f'_match-{args.lepton}.csv'))
    
    data = {
        'gen': gen_particles,
        'reco': reco_particles,
        'matched': match_particles
    }
    #print(data['gen']) 
    plt_configs = {
        'pt':{
        'x_label': r'$p_{T}$ [GeV]',
        'bins': np.array([0,15, 30, 60, 90, 120, 150]),
        'ax_opts': {
            'xscale': 'linear',
            'yscale': 'log'
        }
    },
        'eta':{
            'x_label': r'$\eta$',
            'bins': [-3, -2.7, -2.4, -2.1, -1.8, -1.5, -1.2, -0.9, -0.6, -0.3, 0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0],
            'ax_opts': {
                'xscale': 'linear',
                'yscale': 'linear'
            }
        },
        'd0':{
            'x_label': r'$d0 [m]$',
            'bins': np.array([0, 0.005, 0.01,0.015, 0.02, 0.025, 0.03, 0.04, 0.05]),
            'ax_opts': {
                'xscale': 'linear',
                'yscale': 'log'
            }
            
        },
        'z0':{
            'x_label': r'$z0 [mm]$',
            'bins': np.array([-150,-120,-75,-50, -25, -12, 12, 25, 50, 75, 120, 150]),
            'ax_opts': {
                'xscale': 'linear',
                'yscale': 'linear'
            }
            
        },
        'vr':{
            'x_label': r'$v_{r} [cm]$',
            'bins': np.array([0, 12, 25, 37, 50, 65, 100, 150]),
            'ax_opts': {
                'xscale': 'linear',
                'yscale': 'linear'
            }
            
        },
        'npileup':{
            'x_label': r'$N_{PU}$',
            'bins': 10,
            'ax_opts': {
                'xscale': 'linear',
                'yscale': 'linear'
            }
            
        }
    }
     
    print(f"#generated: {len(data['gen'])}")
    print(f"#reconstructable: {len(data['reco'])}")
    print(f"#matched: {len(data['matched'])}")
    plot_eff_2D(data)
    for plt_key in plt_configs.keys():
        print(plt_key)
        plot(data, var_name = plt_key, plt_config = plt_configs[plt_key] )
        plot_eff(data, var_name = plt_key, plt_config = plt_configs[plt_key] )
    return  
    
     
def main():
    global path, base_dir, args 
    args = myParse()
    if args.mode == 'extract':
        path = Path(args.config)
        base_dir = Path(os.path.basename(args.config))
    save = Path(args.output)
    save.mkdir(parents = True, exist_ok = True)
    
        #raise ValueError  

    exec(args.mode+'()')
    
    
    
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 15})
    
    import matplotlib.pyplot as plt

    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12
    AXES_SIZE = 17
    TICK_SIZE = 18 
    LEGEND_SIZE = 14
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=AXES_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=AXES_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=TICK_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=TICK_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=LEGEND_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
     
    main()

 