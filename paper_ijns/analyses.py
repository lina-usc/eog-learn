import mne
from mne.io import BaseRaw
from mne.preprocessing.eyetracking import set_channel_types_eyetrack
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from pathlib import Path
from itertools import product
import eoglearn
import xarray as xr

from filter import filter_kwargs
from eoglearn.io.eegeyenet import pixels_to_radians, get_annotations_diff
from eoglearn.models.utils import optimal_alpha


ET_MAPPING = {"L-GAZE-X": ('eyegaze', 'px', 'left', 'x'),
              "L-GAZE-Y": ("eyegaze", "px", "left", "y"),
              "L-AREA": ("pupil", "au", "left")}


def get_epochs(raw, events, event_id, baseline=(None, 0), verbose=False):
    # this applies a baseline to the eyegaze channels which we dont generally
    # want but okay for now i think
    return mne.Epochs(raw, events, event_id=event_id, tmin=-.2,
                      tmax=1, preload=True, baseline=baseline, verbose=verbose,
                      on_missing='ignore', reject_by_annotation=False)


def get_evoked(raw, picks, events, event_id, event_list=("2", "5", "15"),
               baseline=(None, 0), format="mne"):
    # see https://eoglearn.readthedocs.io/en/latest/auto_examples/
    # plot_eegeyenet.html#sphx-glr-auto-examples-plot-eegeyenet-py
    # for stimuli locations
    if isinstance(raw, BaseRaw):
        message = "Must pass in an events array to get evoked from Raw."
        assert events is not None, message
        epochs = get_epochs(raw, events, event_id=event_id, baseline=baseline)

    if format == "mne":
        epochs.pick(picks)
        if event_list is None:
            return_val = {}
            for id_ in epochs.event_id:
                sel_epochs = epochs[id_]
                if len(sel_epochs):
                    return_val[id_] = sel_epochs.average()
            return return_val
        return epochs[event_list].average()

    if format == "pandas":
        dat = epochs.to_data_frame(picks="eeg")
        if event_list is None:
            nave = dat.groupby(["epoch", "condition"]).count().reset_index()[["condition", "epoch"]].groupby("condition").count()["epoch"]
            return dat.drop(columns="epoch").groupby(["condition", "time"]).mean(), nave
        else:
            return dat[np.in1d(dat.conditions, event_list)].drop(columns="epoch").groupby(["time"]).mean()


def save_ica_noise(fname, raw):
    if fname.exists():
        return
    tmax = min(raw["original"].times[-1], raw["ica"].times[-1])
    raw_icanoise = (mne.io.RawArray(raw["original"].get_data(picks="eeg", tmax=tmax) -
                    raw["ica"].get_data(picks="eeg", tmax=tmax), raw["ica"].info))
    raw_icanoise.resample(100).export(fname)


def get_insts(fname_clean, event_list=None, diff=False, format="mne",
              adjust_for_RT=False, picks="eeg"):
    fname = Path(str(fname_clean).replace("_clean.", "_sim."))
    if not fname.exists():
        print("not fname.exists()", fname)
        return None

    evoked = {}
    raw = {}
    nave = {}

    for kind in ["original", "clean", "noise", "ica", "sim", "noiseica",
                 "noisesim", "simlocal", "noisesimlocal"]:
        fname = Path(str(fname_clean).replace("clean.", kind + "."))

        raw[kind] = mne.io.read_raw_edf(fname, verbose=False)

        if kind == "original":
            if diff:
                annotations = get_annotations_diff(raw["original"])
                regexp = None
            else:
                annotations = (raw["original"].copy()
                                              .drop_channels(['L-GAZE-X',
                                                              'L-GAZE-Y',
                                                              'L-AREA'],
                                                             on_missing="ignore")
                                              .annotations)
                regexp = "^[0-9]*$"

            events, event_id = mne.events_from_annotations(
                raw["original"].copy().set_annotations(annotations),
                regexp=regexp,
                verbose=False)

            if adjust_for_RT:
                events = adjust_events_for_RT(events, event_id, raw["original"])

        if kind == "original":
            set_channel_types_eyetrack(raw["original"], ET_MAPPING)

        if "TIME" in raw[kind].ch_names:
            raw[kind].drop_channels(["TIME"])

        ch_names = [ch_name
                    for ch_name in raw[kind].ch_names
                    if ch_name[0] == "E"]
        raw[kind].set_channel_types(dict(zip(ch_names,
                                    ["eeg"]*len(ch_names))))

        if format == "mne":
            evoked[kind] = get_evoked(raw[kind], picks, events,
                                      event_id, event_list, format=format)
            nave[kind] = None
        else:
            evoked[kind], nave[kind] = get_evoked(raw[kind], picks, events,
                                                  event_id, event_list, format=format)

    return raw, evoked, event_id, nave


def get_evoked_eye(fname_clean, event_id, event_list=None, diff=False,
                   adjust_for_RT=False):
    fname = Path(str(fname_clean).replace("clean.", "original."))
    if not fname.exists():
        return None

    raw = mne.io.read_raw_edf(fname, verbose=False)
    if diff:
        annotations = get_annotations_diff(raw)
    else:
        annotations = raw.copy().drop_channels(['L-GAZE-X',
                                                'L-GAZE-Y',
                                                'L-AREA']).annotations

    raw.set_annotations(annotations)
    if diff:
        events, event_id = mne.events_from_annotations(raw, verbose=False)
    else:
        events, event_id = mne.events_from_annotations(raw, regexp="^[0-9]*$",
                                                    verbose=False)
    set_channel_types_eyetrack(raw, ET_MAPPING)

    if adjust_for_RT:
        events = adjust_events_for_RT(events, event_id, raw)

    return get_evoked(raw, "eyegaze", events, event_id, event_list,
                      baseline=None)


def get_rt(epochs):
    eye_x, eye_y = epochs.get_data(picks=['L-GAZE-X', 'L-GAZE-Y']).transpose([1, 0, 2])
    r = np.sqrt(eye_x**2 + eye_y**2)
    times = epochs.times
    return [times[(row > 0.05*row.max()) & (times > 0.1)][0] for row in r]


def adjust_events_for_RT(events, event_id, raw):
    epochs = get_epochs(raw, events, event_id=event_id)
    rt = get_rt(epochs)
    rt_sample_offset = (np.array(rt)*raw.info["sfreq"]).astype(int)
    valid_mask = [len(e) == 0 for e in epochs.drop_log]
    events[valid_mask, 0] += rt_sample_offset
    return events


def rms(x):
    return np.sqrt(np.mean(x**2, axis=1))


def get_snrs(evoked, event_id,
             kinds=("clean", "ica", "sim", "simlocal")):
    t = evoked["original"][event_id].times
    ev = {kind: evoked[kind][event_id].get_data() for kind in evoked}
    rt_masks = {"pre": t < 0,
                "post": (t > 0) & (t < 0.2)}

    dfs = []
    dfs_topo = {(time, kind): []
                for time, kind in product(["pre", "post"], kinds)}
    eeg_names = evoked[kinds[0]][event_id].ch_names
    for condition, mask in rt_masks.items():
        signal = rms(ev["original"][:, mask])

        for kind in kinds:
            noise = rms((ev["original"] - ev[kind])[:, mask])
            df = pd.DataFrame([10*np.log10(signal/noise)], columns=eeg_names)
            df["approach"] = kind
            df["condition"] = condition
            df["event_id"] = event_id
            dfs.append(df)

            nsr = rms(ev[kind][:, mask]) / rms(ev["original"][:, mask])
            percent_noise = 1 - nsr
            percent_noise *= 100

            data_dict = dict(list(zip(eeg_names, percent_noise)))
            df = pd.DataFrame([data_dict])
            df["event_id"] = event_id
            dfs_topo[(condition, kind)].append(df)

    return pd.concat(dfs), {key: pd.concat(dfs_topo[key]) for key in dfs_topo}


def compute_et_xarrays(path="processed", diff=False, nb_files=None, dryrun=False):

    kinds = ["clean", "ica", "sim", "simlocal"]
    snr_dfs = []
    et_signals_dfs = []
    topo_ev_dfs = {(time, kind): []
                   for time, kind in product(["pre", "post"], kinds)}

    files = list(Path(path).glob("*_clean.edf"))
    if nb_files:
        files = files[:nb_files]
    for fname_clean in tqdm(files):
        run, subject = fname_clean.name.split("_")[:2]
        insts = get_insts(fname_clean, diff=diff, adjust_for_RT=True)
        if insts is None:
            continue
        evoked, event_id = insts[1:3]
        evoked_eye = get_evoked_eye(fname_clean, event_id, event_list=None,
                                    diff=diff, adjust_for_RT=True)

        for event_id in evoked_eye:
            df, topo_ev = get_snrs(evoked, event_id)
            df["subject"], df["run"] = run, subject
            snr_dfs.append(df)

            for key in topo_ev_dfs:
                topo_ev[key]["subject"], topo_ev[key]["run"] = run, subject
                topo_ev_dfs[key].append(topo_ev[key])

            ev_eye = evoked_eye[event_id].get_data()
            df = pd.DataFrame({
                'times': evoked_eye[event_id].times,
                'eye-x': ev_eye[0],
                'eye-y': ev_eye[1]
            })
            df = df.melt(id_vars="times", value_name="amp", var_name="ch_name")
            df["subject"], df["run"] = run, subject
            df["event_id"] = event_id
            et_signals_dfs.append(df)

    if not snr_dfs:
        raise RuntimeError(
            f"No valid recordings found in '{path}'. "
            "Ensure Steps 1–4 have been run successfully."
        )
    snr_df = pd.concat(snr_dfs)
    topo_ev_df = {key: pd.concat(topo_ev_dfs[key]) for key in topo_ev_dfs}

    snr_xr = snr_df.melt(id_vars=["approach", "condition", "event_id",
                                  "subject", "run"],
                         var_name="ch_name", value_name="snr")\
                   .set_index(["approach", "condition", "event_id",
                               "subject", "run", "ch_name"])\
                   .to_xarray()
    if not dryrun:
        if diff:
            snr_xr.to_netcdf("snr_diff.netcdf")
        else:
            snr_xr.to_netcdf("snr.netcdf")

    for condition, kind in topo_ev_df:
        topo_ev_df[(condition, kind)]["condition"] = condition
        topo_ev_df[(condition, kind)]["kind"] = kind

    topo_ev_xr = pd.concat(topo_ev_df.values())\
                   .melt(id_vars=["subject", "run", "kind",
                                  "event_id", "condition"],
                         var_name="ch_name", value_name="percent")\
                   .set_index(["subject", "run", "ch_name", "kind",
                               "event_id", "condition"])\
                   .to_xarray()

    if not dryrun:
        if diff:
            topo_ev_xr.to_netcdf("topo_erp_diff.netcdf")
        else:
            topo_ev_xr.to_netcdf("topo_erp.netcdf")

    et_signals_df = pd.concat(et_signals_dfs)
    et_signals_xr = et_signals_df.set_index(["times", "ch_name", "subject",
                                             "run", "event_id"]).to_xarray()

    if not dryrun:
        if diff:
            et_signals_xr.to_netcdf("et_signals_diff.netcdf")
        else:
            et_signals_xr.to_netcdf("et_signals.netcdf")

    return snr_xr, topo_ev_xr, et_signals_xr


def compute_erp_xarrays(path="processed", diff=False, nb_files=None, dryrun=False):
    eeg_signals_dfs = []
    kinds = ["clean", "ica", "sim", "simlocal",
             "noise", "noiseica", "noisesim", "noisesimlocal"]

    topo_dfs = {kind: [] for kind in kinds}
    nave_xrs = []

    files = list(Path(path).glob("*_clean.edf"))
    if nb_files:
        files = files[:nb_files]

    for fname_clean in tqdm(files):
        subject, run = fname_clean.name.split("_")[:2]
        insts = get_insts(fname_clean, diff=diff, format="pandas", adjust_for_RT=True)
        if insts is None:
            continue
        raw, evoked, _, nave = insts

        eeg_names = raw["original"].copy().pick("eeg").ch_names

        nsample = min(len(raw["original"].times), len(raw["ica"]))
        signal = rms(raw["original"].get_data(picks=eeg_names)[:, :nsample])

        for kind in topo_dfs:
            noise = raw["original"].get_data(picks=eeg_names)[:, :nsample]
            noise -= raw[kind].get_data(picks=eeg_names)[:, :nsample]
            noise = rms(noise)

            nsr = noise / signal
            percent_noise = nsr
            percent_noise *= 100

            data_dict = dict(list(zip(eeg_names, percent_noise)))
            df = pd.DataFrame([data_dict])
            df["subject"] = subject
            df["run"] = run
            topo_dfs[kind].append(df)

        for kind in ["original"] + kinds:
            ch_names = evoked[kind].columns.values
            n_ch = len(ch_names)
            N = len(evoked[kind].index)
            condition, time = evoked[kind].reset_index()[["condition", "time"]].T.values
            df = pd.DataFrame({
                    'times': np.concatenate([time]*n_ch),
                    "amp": np.concatenate(evoked[kind].values.T),
                    "subject": [subject]*N*n_ch,
                    "run": [run]*N*n_ch,
                    "kind": [kind]*N*n_ch,
                    "event_id": np.concatenate([condition]*n_ch),
                    "ch_name": np.concatenate([[ch_name]*N for ch_name in ch_names]),
                })
            eeg_signals_dfs.append(df)

        nave_xrs.append(xr.DataArray(
            [[nave["original"].values]],
            coords={"subject": [subject],
                    "run": [run],
                    "event_id": nave["original"].index.values},
            dims=["subject", "run", "event_id"]))

    if not eeg_signals_dfs:
        raise RuntimeError(
            f"No valid recordings found in '{path}'. "
            "Ensure Steps 1–4 have been run successfully."
        )
    eeg_signals_df = pd.concat(eeg_signals_dfs)
    topo_df = {kind: pd.concat(topo_dfs[kind]) for kind in topo_dfs}

    cols = ["times", "kind", "event_id", "ch_name", "subject", "run"]
    eeg_signals_df.set_index(cols, inplace=True)
    eeg_signals_xr = eeg_signals_df.to_xarray()

    eeg_signals_xr["nave"] = xr.combine_by_coords(nave_xrs)

    if not dryrun:
        if diff:
            eeg_signals_xr.to_netcdf("eeg_signals_diff.netcdf")
        else:
            eeg_signals_xr.to_netcdf("eeg_signals.netcdf")

    for kind in kinds:
        topo_df[kind]["kind"] = kind
    topo_df = pd.concat(topo_df.values())
    topo_xr = topo_df.melt(id_vars=["subject", "run", "kind"],
                           var_name="ch_name",
                           value_name="percent")\
                     .set_index(["subject", "run", "ch_name", "kind"])\
                     .to_xarray()

    if not dryrun:
        if diff:
            topo_xr.to_netcdf("topo_raw_diff.netcdf")
        else:
            topo_xr.to_netcdf("topo_raw.netcdf")

    return eeg_signals_xr, topo_xr


def get_sim_eog(subject, run, return_raw=False):
    fpath = eoglearn.datasets.fetch_eegeyenet(subject=subject, run=run)
    raw = eoglearn.io.read_raw_eegeyenet(fpath)

    raw.set_montage("GSN-HydroCel-129")
    raw.set_eeg_reference("average")
    raw.set_annotations(None)
    raw.filter(picks="eeg", **filter_kwargs)

    tmax = int(raw.times[-1])
    raw.crop(tmax=tmax, include_tmax=False)

    raw = mne.preprocessing.eyetracking.interpolate_blinks(raw, interpolate_gaze=True, buffer=0.1)

    gaze_x_px, gaze_y_px = raw.get_data(picks=['L-GAZE-X', 'L-GAZE-Y'])

    theta, phi = pixels_to_radians(gaze_x_px, gaze_y_px,
                                    width_px=800, height_px=600,
                                    screen_diag_in=24.0, viewing_dist_cm=68.0,
                                    origin='top-left')

    theta[np.isnan(theta)] = 0
    phi[np.isnan(phi)] = 0

    moment_x_left = np.cos(phi)*np.cos(theta)
    moment_y_left = np.cos(phi)*np.sin(theta)
    moment_z_left = np.sin(phi)
    moment_x_right = np.cos(phi)*np.cos(theta)
    moment_y_right = np.cos(phi)*np.sin(theta)
    moment_z_right = np.sin(phi)

    dipole_moments = np.vstack([moment_y_left, moment_x_left,
                                moment_z_left, moment_y_right,
                                moment_x_right, moment_z_right]).T

    sim_elect_only = np.load('leadfield_elect_only.npy')

    eog_sims = (dipole_moments @ sim_elect_only).T

    raw_sim = mne.io.RawArray(eog_sims, raw.copy().pick("eeg").info)

    if return_raw:
        return raw_sim, raw

    return raw_sim
