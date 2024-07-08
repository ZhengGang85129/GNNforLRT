#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file define series of configuration while plotting.
"""

import numpy as np

# Different track types which will be considered.
track_types = [
    'prompt',
    'displaced'
]

# Different parameter to consider.
particle_parameters = [
    'pt', 'd0', 'z0', 'vr'
]

# Plotting configurations.
plot_configs = {
    'prompt': {
        'pt': {
            'args': {
                'x_variable': 'pt',
                'x_label': 'Truth Transverse Momentum $p_T$ [GeV]',
                'bins': np.arange(1.0, 30.5, 3.0)
            }
        },
        'd0': {
            'args': {
                'x_variable': 'd0',
                'x_label': r'Truth Transverse Impact Parameter $|d_0|$[mm]',
                'bins': np.arange(0, 0.1, step=0.005)
            }
        },
        'z0': {
            'args': {
                'x_variable': 'z0',
                'x_label': r'Truth Longitudinal Impact Parameter $z_0$[mm]',
                'bins': np.arange(-150.0, 150.0, step=10.0)
            }
        },
        'vr': {
            'args': {
                'x_variable': 'vr',
                'x_label': 'Production vertex radius [mm]',
                'bins': np.arange(0.0, 301, 25)
            }
        }
    },
    'displaced': {
        'pt': {
            'args': {
                'x_variable': 'pt',
                'x_label': 'Truth Transverse Momentum $p_T$ [GeV]',
                'bins': np.arange(1.0, 30.5, 3.0)
            }
        },
        'd0': {
            'args': {
                'x_variable': 'd0',
                'x_label': r'Truth Transverse Impact Parameter $|d_0|$[mm]',
                'bins': np.arange(0, 800, step=50)
            }
        },
        'z0': {
            'args': {
                'x_variable': 'z0',
                'x_label': r'Truth Longitudinal Impact Parameter $z_0$[mm]',
                'bins': np.arange(-2000, 2001, step=200)
            }
        },
        'vr': {
            'args': {
                'x_variable': 'vr',
                'x_label': 'Production vertex radius [mm]',
                'bins': np.arange(0.0, 301, 25)
            }
        }
    }
}

particle_filters = {
    'displaced': lambda particles: (
        (particles['parent_ptype'] == 50) |
        (particles['parent_ptype'] == -50)
    ),
    'prompt': lambda particles: (
        (particles['parent_ptype'] == 24) |
        (particles['parent_ptype'] == -24)
    )
}
