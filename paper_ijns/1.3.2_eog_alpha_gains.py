import numpy as np
import pandas as pd
import mne
from tqdm.auto import tqdm
import eoglearn

from eoglearn.io.eegeyenet import pixels_to_radians
from eoglearn.models.utils import optimal_alpha

"""
    ## Step 8: Compute alpha factors
"""


def prep_data(subject="EP10", run=1):
    fpath = eoglearn.datasets.fetch_eegeyenet(subject=subject, run=run)
    raw = eoglearn.io.read_raw_eegeyenet(fpath)

    raw.set_montage("GSN-HydroCel-129")
    #raw.filter(1, 30, picks="eeg").resample(100)  # DO NOT filter eyetrack channels
    raw.set_eeg_reference("average")
    return raw


def get_angles(subject, run, return_raw=False):
    raw = prep_data(subject, run)
    raw = mne.preprocessing.eyetracking.interpolate_blinks(raw, interpolate_gaze=True, buffer=0.1)

    gaze_x_px, gaze_y_px = raw.get_data(picks=['L-GAZE-X', 'L-GAZE-Y'])

    theta, phi = pixels_to_radians(gaze_x_px,
                                    gaze_y_px,
                                    width_px=800,
                                    height_px=600,
                                    screen_diag_in=24.0,
                                    viewing_dist_cm=68.0,
                                    origin='top-left')
    # TODO: Fix
    theta[np.isnan(theta)] = 0
    phi[np.isnan(phi)] = 0
    if return_raw:
        return theta, phi, raw
    return theta, phi


def get_dipoles(theta, phi):
    # Computing moments as per the eye diagram.
    # Moment
    moment_x_left = np.cos(phi)*np.cos(theta) 
    moment_y_left = np.cos(phi)*np.sin(theta) 
    moment_z_left = np.sin(phi)
    # Same moments for both eyes
    moment_x_right = np.cos(phi)*np.cos(theta) 
    moment_y_right = np.cos(phi)*np.sin(theta) 
    moment_z_right = np.sin(phi)

    # Mapping the eye coordinate system with the head coordinate system 
    # (left-to-righ, back-to-front, bottom-to-top)
    # x-head == y-eye; y-head == x-eye; x-head == x-eye
    return np.vstack([moment_y_left, moment_x_left, moment_z_left,
                      moment_y_right, moment_x_right, moment_z_right]).T


def main():
    sim_elect_only = np.load('leadfield_elect_only.npy')

    runs_dict = eoglearn.datasets.eegeyenet.get_subjects_runs()
    subject_run = np.concatenate([[(subject, run) 
                                   for run in runs_dict[subject]]
                                  for subject in runs_dict if "EP" in subject])

    dfs = []
    for subject, run in tqdm(subject_run):
        theta, phi, raw = get_angles(subject=subject, run=run, return_raw=True)
        dipole_moments = get_dipoles(theta, phi)

        eog_sims = (dipole_moments @ sim_elect_only).T
        raw_sim = mne.io.RawArray(eog_sims, raw.copy().pick("eeg").info, verbose=False)
        raw_sim.filter(1, 100)  # Filter the simulated data as the original data have been filtered

        x_sim = raw_sim.copy().get_data(picks="eeg")
        x_raw = raw.copy().get_data(picks="eeg")

        gains = (np.ones((1, 6)) @ sim_elect_only).squeeze() 
        gains_x = (np.array([[1, 0, 0, 1, 0, 0]]) @ sim_elect_only).squeeze() 
        gains_y = (np.array([[0, 1, 0, 0, 1, 0]]) @ sim_elect_only).squeeze() 
        gains_z = (np.array([[0, 0, 1, 0, 0, 1]]) @ sim_elect_only).squeeze() 
        global_gains = gains * optimal_alpha(x_sim, x_raw)
        global_gains_x = gains_x * optimal_alpha(x_sim, x_raw)
        global_gains_y = gains_y * optimal_alpha(x_sim, x_raw)
        global_gains_z = gains_z * optimal_alpha(x_sim, x_raw)

        ind_gains = gains * optimal_alpha(x_sim, x_raw, axis=1)
        ind_gains_x = gains_x * optimal_alpha(x_sim, x_raw, axis=1)
        ind_gains_y = gains_y * optimal_alpha(x_sim, x_raw, axis=1)
        ind_gains_z = gains_z * optimal_alpha(x_sim, x_raw, axis=1)

        df = pd.DataFrame({"gains": global_gains, "ind_gains": ind_gains, 
                            "gains_x": global_gains_x, "ind_gains_x": ind_gains_x, 
                            "gains_y": global_gains_y, "ind_gains_y": ind_gains_y, 
                            "gains_z": global_gains_z, "ind_gains_z": ind_gains_z, 
                            "ch_names": raw.copy().pick("eeg").ch_names})
        df["subject"] = subject
        df["run"] = run
        dfs.append(df)

    df = pd.concat(dfs)
    df.to_csv("gains.csv")


if __name__ == "__main__":
    main()
