#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import pandas as pd

from ExaTrkXDataIO import DataReader

import matplotlib
import matplotlib.pyplot as plt
from ExaTrkXPlotting import plotter, PlotConfig
from ExaTrkXPlots import hits, pairs, particles

if __name__ == '__main__':
    data_dir = Path("./")

    reader = DataReader(
        data_dir/"particle.yaml",
        data_dir
    )

    output = Path('./results')
    output.mkdir(exist_ok=True, parents=True)
    
    all = pd.DataFrame()

    for data in reader.read():
        particles = data['particles']

        # Compute some necessary parameters.
        pt = np.sqrt(particles['px'] ** 2 + particles['py'] ** 2)
        pz = particles['pz']
        z0 = particles['vz']
        d0 = np.sqrt(particles['vx']**2 + particles['vy']**2)
        p3 = np.sqrt(pt ** 2 + pz ** 2)
        p_theta = np.arccos(pz / p3)
        eta = -np.log(np.tan(0.5 * p_theta))
        vr = np.sqrt(d0**2 + particles['vz']**2)

        particles = particles.assign(
            pt=pt,
            eta=eta,
            z0=z0,
            d0=d0,
            vr=vr
        )

        hits = data['hits']
        pids = pd.DataFrame(data={
            'particle_id': hits['particle_id'].unique()
        })
        pids = pids.merge(particles, on="particle_id")

        all = pd.concat([all, pids])

    # vr
    matplotlib.rcParams.update({'font.size': 16})

    fig, ax = plt.subplots(figsize=(8, 8), tight_layout=True)
    
    n, bins, patches = ax.hist(
        all['vr'],
        bins=np.arange(0.0, 1001, 25),
        histtype='step',
        density=True,
        label="All",
        lw=2,
    )
    
    n, bins, patches = ax.hist(
        all[
            all.parent_ptype.abs() == 24
        ]['vr'],
        bins=np.arange(0.0, 1001, 25),
        histtype='step',
        density=True,
        label="Prompt",
        lw=2,
    )

    n, bins, patches = ax.hist(
        all[
            all.parent_ptype.abs() == 50
        ]['vr'],
        bins=np.arange(0.0, 1001, 25),
        histtype='step',
        density=True,
        label="Displaced",
        lw=2,
    )
    # ax.set_xlim(0, 500)
    ax.set_xlabel('Production vertex radius [mm]')
    ax.set_ylabel('Normalized Scale')
    ax.legend()
    fig.savefig(output/'dist_vr.pdf')

    # z0
    matplotlib.rcParams.update({'font.size': 16})

    fig, ax = plt.subplots(figsize=(8, 8), tight_layout=True)

    n, bins, patches = ax.hist(
        all['z0'],
        bins=np.arange(-2000, 2001, step=200),
        histtype='step',
        density=True,
        label="All",
        lw=2,
    )

    n, bins, patches = ax.hist(
        all[
            all.parent_ptype.abs() == 24
            ]['z0'],
        bins=np.arange(-2000, 2001, step=200),
        histtype='step',
        density=True,
        label="Prompt",
        lw=2,
    )

    n, bins, patches = ax.hist(
        all[
            all.parent_ptype.abs() == 50
            ]['z0'],
        bins=np.arange(-2000, 2001, step=200),
        histtype='step',
        density=True,
        label="Displaced",
        lw=2,
    )
    # ax.set_xlim(0, 500)
    ax.set_xlabel('Truth Longitudinal Impact Parameter $z_0$[mm]')
    ax.set_ylabel('Normalized Scale')
    ax.legend()
    fig.savefig(output/'dist_z0.pdf')

    #d0
    matplotlib.rcParams.update({'font.size': 16})

    fig, ax = plt.subplots(figsize=(8, 8), tight_layout=True)

    n, bins, patches = ax.hist(
        all['d0'],
        bins=np.arange(0, 1, step=0.01),
        histtype='step',
        density=True,
        label="All",
        lw=2,
    )

    n, bins, patches = ax.hist(
        all[
            all.parent_ptype.abs() == 24
            ]['d0'],
        bins=np.arange(0, 1, step=0.01),
        histtype='step',
        density=True,
        label="Prompt",
        lw=2,
    )

    n, bins, patches = ax.hist(
        all[
            all.parent_ptype.abs() == 50
            ]['d0'],
        bins=np.arange(0, 1, step=0.01),
        histtype='step',
        density=True,
        label="Displaced",
        lw=2,
    )
    # ax.set_xlim(0, 500)
    ax.set_xlabel('Truth Transverse Impact Parameter $|d_0|$[mm]')
    ax.set_ylabel('Normalized Scale')
    ax.legend()
    fig.savefig(output/'dist_d0.pdf')

    ###pt
    matplotlib.rcParams.update({'font.size': 16})

    fig, ax = plt.subplots(figsize=(8, 8), tight_layout=True)

    n, bins, patches = ax.hist(
        all['pt'],
        bins=np.arange(0.0, 90.5, 3.0),
        histtype='step',
        density=True,
        label="All",
        lw=2,
    )

    n, bins, patches = ax.hist(
        all[
            all.parent_ptype.abs() == 24
            ]['pt'],
        bins=np.arange(0.0, 90.5, 3.0),
        histtype='step',
        density=True,
        label="Prompt",
        lw=2,
    )

    n, bins, patches = ax.hist(
        all[
            all.parent_ptype.abs() == 50
            ]['pt'],
        bins=np.arange(0.0, 90.5, 3.0),
        histtype='step',
        density=True,
        label="Displaced",
        lw=2,
    )
    # ax.set_xlim(0, 500)
    ax.set_xlabel('Truth Transverse Momentum $p_T$ [GeV]')
    ax.set_ylabel('Normalized Scale')
    ax.legend()
    fig.savefig(output/'dist_pt.pdf')

