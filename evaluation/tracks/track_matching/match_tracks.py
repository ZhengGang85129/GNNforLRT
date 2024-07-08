#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np


def analyze_tracks(truth, submission):
    """
    Taken from https://github.com/LAL/trackml-library/blob/master/trackml/score.py
    Compute the majority particle, hit counts, and weight for each track.
    Parameters
    ----------
    truth : pandas.DataFrame
        Truth information. Must have hit_id, particle_id, and weight columns.
    submission : pandas.DataFrame
        Proposed hit/track association. Must have hit_id and track_id columns.
    Returns
    -------
    pandas.DataFrame
        Contains track_id, nhits, major_particle_id, major_particle_nhits,
        major_nhits, and major_weight columns.
    """
    # true number of hits for each particle_id
    particles_nhits = truth['particle_id'].value_counts(sort=False)

    # combined event with minimal reconstructed and truth information
    event = pd.merge(
        truth[['hit_id', 'particle_id']],
        submission[['hit_id', 'track_id']],
        on=['hit_id'],
        how='left',
        validate='one_to_one'
    )
    event.drop(
        'hit_id',
        axis=1,
        inplace=True
    )
    event.sort_values(
        by=['track_id', 'particle_id'],
        inplace=True
    )

    # Assumptions: 0 <= track_id, 0 <= particle_id

    tracks = []
    # running sum for the reconstructed track we are currently in
    rec_track_id = -1
    rec_nhits = 0
    # running sum for the particle we are currently in (in this track_id)
    cur_particle_id = -1
    cur_nhits = 0
    # majority particle with most hits up to now (in this track_id)
    maj_particle_id = -1
    maj_nhits = 0

    for hit in event.itertuples(index=False):
        # we reached the next track so we need to finish the current one
        if (rec_track_id != -1) and (rec_track_id != hit.track_id):
            # could be that the current particle is the majority one
            if maj_nhits < cur_nhits:
                maj_particle_id = cur_particle_id
                maj_nhits = cur_nhits
            # store values for this track
            tracks.append((rec_track_id, rec_nhits, maj_particle_id,
                           particles_nhits[maj_particle_id], maj_nhits))

        # setup running values for next track (or first)
        if rec_track_id != hit.track_id:
            rec_track_id = hit.track_id
            rec_nhits = 1
            cur_particle_id = hit.particle_id
            cur_nhits = 1
            maj_particle_id = -1
            maj_nhits = 0
            continue

        # hit is part of the current reconstructed track
        rec_nhits += 1

        # reached new particle within the same reconstructed track
        if cur_particle_id != hit.particle_id:
            # check if last particle has more hits than the majority one
            # if yes, set the last particle as the new majority particle
            if maj_nhits < cur_nhits:
                maj_particle_id = cur_particle_id
                maj_nhits = cur_nhits
            # reset runnig values for current particle
            cur_particle_id = hit.particle_id
            cur_nhits = 1
        # hit belongs to the same particle within the same reconstructed track
        else:
            cur_nhits += 1

    # last track is not handled inside the loop
    if maj_nhits < cur_nhits:
        maj_particle_id = cur_particle_id
        maj_nhits = cur_nhits

    if rec_track_id != -1:
        # store values for the last track
        tracks.append((
            rec_track_id,
            rec_nhits,
            maj_particle_id,
            particles_nhits[maj_particle_id],
            maj_nhits
        ))

    cols = [
        'track_id',
        'nhits',
        'major_particle_id',
        'major_particle_nhits',
        'major_nhits'
    ]

    return pd.DataFrame.from_records(tracks, columns=cols)


def match_tracks(
    truth: pd.DataFrame,
    reconstructed: pd.DataFrame,
    particles: pd.DataFrame,
    min_hits_truth=9,
    min_hits_reco=5,
    min_pt=1.,
    frac_reco_matched=0.5,
    frac_truth_matched=0.5,
    particle_filter = None
):
    """
    Match reconstructed tracks to particles.

    Args:
        truth: a dataframe with columns of ['hit_id', 'particle_id']
        reconstructed: a dataframe with columns of ['hit_id', 'track_id']
        particles: a dataframe with columns of
            ['particle_id', 'pt', 'eta', 'radius', 'vz', 'charge'].
            radius = sqrt(vx**2 + vy**2),
            ['vx', 'vy', 'vz'] are the production vertex of the particle
        min_hits_truth: minimum number of hits for truth tracks
        min_hits_reco:  minimum number of hits for reconstructed tracks

    Returns:
        A tuple of (
            n_true_tracks: int, number of true tracks
            n_reco_tracks: int, number of reconstructed tracks
            n_matched_reco_tracks: int, number of reconstructed tracks
                matched to true tracks
            matched_pids: np.narray, a list of particle IDs matched
                by reconstructed tracks
        )
    """
    # just in case particle_id == 0 included in truth.
    truth = truth[truth.particle_id > 0]

    # Associate hits with particle data.
    hits = truth.merge(
        particles,
        on='particle_id',
        how='left'
    )

    # Count number of hits for each particle.
    n_hits_per_particle = hits.groupby("particle_id")['hit_id'].count()
    n_hits_per_particle = n_hits_per_particle.reset_index().rename(columns={
        "hit_id": "nhits"
    })

    # Associate number of hits to hits.
    hits = hits.merge(n_hits_per_particle, on='particle_id', how='left')

    if min_pt > 0:
        hits = hits[
            (hits.nhits >= min_hits_truth) & (hits.pt >= min_pt)
        ]
    else:
        hits = hits[hits.nhits >= min_hits_truth]

    hits = hits[hits.eta.abs() <= 4]

    # Extract trackable particles.
    trackable_pids = np.unique(
        hits.particle_id.values
    )

    pruned_sub = reconstructed[
        reconstructed.hit_id.isin(hits.hit_id)
    ]

    # some hits do not exist in the reconstructed tracks,
    # fill their track id with -1.
    if pruned_sub.shape[0] != hits.shape[0]:
        extended_sub = hits[['hit_id']].merge(
            pruned_sub,
            on='hit_id',
            how='left'
        ).fillna(-1)
    else:
        extended_sub = pruned_sub

    # Compute track properties.
    tracks = analyze_tracks(hits, extended_sub)

    # double-majority matching criteria.
    purity_rec = np.true_divide(
        tracks['major_nhits'],
        tracks['nhits']
    )
    purity_maj = np.true_divide(
        tracks['major_nhits'],
        tracks['major_particle_nhits']
    )
    matched_reco_track = (
        (tracks.nhits >= min_hits_reco) &
        (frac_reco_matched < purity_rec) &
        (frac_truth_matched < purity_maj)
    )

    reco_tracks = tracks[tracks.nhits >= min_hits_reco]
    matched_pids = tracks[matched_reco_track].major_particle_id.values

    n_true_tracks = np.unique(hits.particle_id).shape[0]
    n_reco_tracks = reco_tracks.shape[0]
    n_matched_reco_tracks = np.sum(matched_reco_track)

    is_matched = particles.particle_id.isin(matched_pids).values
    is_trackable = particles.particle_id.isin(trackable_pids).values

    particles = particles.assign(
        is_matched=is_matched,
        is_trackable=is_trackable
    )

    # Only track particle with charge.
    # This should be common requirement.
    composed_filter = (particles.charge.abs() > 0)

    # Custom filter.
    if particle_filter is not None:
        composed_filter &= particle_filter

    # pT cut.
    if min_pt > 0:
        composed_filter &= (particles.pt >= min_pt)

    particles = particles[
        composed_filter
    ]

    return (
        n_true_tracks,
        n_reco_tracks,
        n_matched_reco_tracks,
        particles
    )
