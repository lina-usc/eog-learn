import mne
from mne.io import BaseRaw
from mne.preprocessing.eyetracking import set_channel_types_eyetrack
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
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

# LSTM clean/noise kind names per training condition
_LSTM_KINDS = {
    "perrecording":  ("clean",),
    "persubject":    ("clean_persubject",),
    "acrosssubject": ("clean_acrosssubject",),
}
_LSTM_NOISE_KINDS = {
    "perrecording":  ("noise",),
    "persubject":    ("noise_persubject",),
    "acrosssubject": ("noise_acrosssubject",),
}


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

    _OPTIONAL_KINDS = frozenset({
        "clean_persubject", "clean_acrosssubject",
        "noise_persubject", "noise_acrosssubject",
    })
    for kind in ["original", "clean", "noise", "ica", "sim", "noiseica",
                 "noisesim", "simlocal", "noisesimlocal",
                 "clean_persubject", "clean_acrosssubject",
                 "noise_persubject", "noise_acrosssubject"]:
        fname = Path(str(fname_clean).replace("clean.", kind + "."))
        if kind in _OPTIONAL_KINDS and not fname.exists():
            continue

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


def compute_et_xarrays(path="processed", diff=False, nb_files=None,
                        dryrun=False, lstm_condition="perrecording"):
    lstm_kinds = list(_LSTM_KINDS[lstm_condition])
    kinds = lstm_kinds + ["ica", "sim", "simlocal"]
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
        raw_dict, evoked, event_id, _ = insts
        # Close file handles immediately — raw data is not needed beyond evoked
        for r in raw_dict.values():
            r.close()
        del raw_dict, insts

        evoked_eye = get_evoked_eye(fname_clean, event_id, event_list=None,
                                    diff=diff, adjust_for_RT=True)

        for event_id in evoked_eye:
            present_kinds = tuple(k for k in kinds if k in evoked)
            df, topo_ev = get_snrs(evoked, event_id, kinds=present_kinds)
            df["subject"], df["run"] = run, subject
            snr_dfs.append(df)

            for key, df_topo in topo_ev.items():
                df_topo["subject"], df_topo["run"] = run, subject
                topo_ev_dfs[key].append(df_topo)

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
        del evoked

    if not snr_dfs:
        raise RuntimeError(
            f"No valid recordings found in '{path}'. "
            "Ensure Steps 1–4 have been run successfully."
        )
    snr_df = pd.concat(snr_dfs)
    topo_ev_df = {key: pd.concat(topo_ev_dfs[key])
                  for key in topo_ev_dfs if topo_ev_dfs[key]}

    snr_xr = snr_df.melt(id_vars=["approach", "condition", "event_id",
                                  "subject", "run"],
                         var_name="ch_name", value_name="snr")\
                   .set_index(["approach", "condition", "event_id",
                               "subject", "run", "ch_name"])\
                   .to_xarray()
    if not dryrun:
        suffix = f"_{lstm_condition}" + ("_diff" if diff else "")
        snr_xr.to_netcdf(f"snr{suffix}.netcdf")

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
        topo_ev_xr.to_netcdf(f"topo_erp{suffix}.netcdf")

    et_signals_df = pd.concat(et_signals_dfs)
    et_signals_xr = et_signals_df.set_index(["times", "ch_name", "subject",
                                             "run", "event_id"]).to_xarray()

    if not dryrun:
        et_signals_xr.to_netcdf(f"et_signals{suffix}.netcdf")

    return snr_xr, topo_ev_xr, et_signals_xr


def compute_erp_xarrays(path="processed", diff=False, nb_files=None,
                         dryrun=False, lstm_condition="perrecording"):
    eeg_signals_dfs = []
    lstm_kinds = list(_LSTM_KINDS[lstm_condition])
    lstm_noise_kinds = list(_LSTM_NOISE_KINDS[lstm_condition])
    kinds = lstm_kinds + lstm_noise_kinds + [
        "ica", "noiseica", "sim", "simlocal", "noisesim", "noisesimlocal"]

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
        del insts

        eeg_names = raw["original"].copy().pick("eeg").ch_names

        nsample = min(len(raw["original"].times), len(raw["ica"]))
        # Cache original data once to avoid redundant EDF reads per kind
        original_data = raw["original"].get_data(picks=eeg_names)[:, :nsample]
        signal = rms(original_data)

        for kind in topo_dfs:
            if kind not in raw:
                continue
            noise = original_data - raw[kind].get_data(picks=eeg_names)[:, :nsample]
            noise = rms(noise)

            nsr = noise / signal
            percent_noise = nsr
            percent_noise *= 100

            data_dict = dict(list(zip(eeg_names, percent_noise)))
            df = pd.DataFrame([data_dict])
            df["subject"] = subject
            df["run"] = run
            topo_dfs[kind].append(df)

        # Close file handles — all raw data has been extracted
        for r in raw.values():
            r.close()
        del raw, original_data

        for kind in ["original"] + kinds:
            if kind not in evoked:
                continue
            # Keep wide format (n_cond×n_time rows × n_ch cols) to avoid a
            # 128× row-explosion from melt; convert to xarray after the loop.
            df_wide = evoked[kind].copy()
            df_wide.index.names = ["event_id", "times"]
            df_wide["subject"] = subject
            df_wide["run"] = run
            df_wide["kind"] = kind
            eeg_signals_dfs.append(df_wide.reset_index())

        nave_xrs.append(xr.DataArray(
            [[nave["original"].values]],
            coords={"subject": [subject],
                    "run": [run],
                    "event_id": nave["original"].index.values},
            dims=["subject", "run", "event_id"]))
        del evoked

    if not eeg_signals_dfs:
        raise RuntimeError(
            f"No valid recordings found in '{path}'. "
            "Ensure Steps 1–4 have been run successfully."
        )

    # Convert wide-format accumulation → xarray without a long-format detour.
    # Each element of eeg_signals_dfs is (n_cond×n_time, n_ch+3) wide;
    # going through melt/long-format would inflate memory by n_ch (≈128×).
    eeg_signals_df = pd.concat(eeg_signals_dfs)
    del eeg_signals_dfs
    eeg_signals_df = eeg_signals_df.set_index(
        ["times", "kind", "event_id", "subject", "run"])
    # to_xarray() creates a Dataset with one variable per channel;
    # to_array() stacks them into a single (ch_name, ...) DataArray.
    ds = eeg_signals_df.to_xarray()
    del eeg_signals_df
    eeg_signals_xr = ds.to_array(dim="ch_name").to_dataset(name="amp")
    del ds

    topo_df = {kind: pd.concat(topo_dfs[kind])
               for kind in topo_dfs if topo_dfs[kind]}

    eeg_signals_xr["nave"] = xr.combine_by_coords(nave_xrs)

    if not dryrun:
        suffix = f"_{lstm_condition}" + ("_diff" if diff else "")
        eeg_signals_xr.to_netcdf(f"eeg_signals{suffix}.netcdf")

    for kind in topo_df:
        topo_df[kind]["kind"] = kind
    topo_df = pd.concat(topo_df.values())
    topo_xr = topo_df.melt(id_vars=["subject", "run", "kind"],
                           var_name="ch_name",
                           value_name="percent")\
                     .set_index(["subject", "run", "ch_name", "kind"])\
                     .to_xarray()

    if not dryrun:
        topo_xr.to_netcdf(f"topo_raw{suffix}.netcdf")

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
