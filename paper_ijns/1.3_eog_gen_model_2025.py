import marimo

__generated_with = "0.20.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import mne
    import seaborn as sns
    from tqdm.notebook import tqdm
    import eoglearn

    from mesh import plotly_sphere, get_eye_mesh, get_scalp_mesh
    from plotting import plot_head
    from eoglearn.io.eegeyenet import pixels_to_radians
    from eoglearn.models.utils import optimal_alpha

    return (
        Path,
        eoglearn,
        get_eye_mesh,
        get_scalp_mesh,
        mne,
        np,
        optimal_alpha,
        plot_head,
        plotly_sphere,
        plt,
    )


@app.cell
def _(mo):
    mo.md("""
    # Biophysical EOG Forward Model Generation

    This notebook builds a biophysical EOG forward model using SimNIBS finite-element
    dipole simulations on the `ernie` head model. It produces two output files:

    - **`leadfield.npy`** — full mesh leadfield `(6, n_nodes)`
    - **`leadfield_elect_only.npy`** — electrode-subset leadfield `(6, 129)`

    **Prerequisites:**
    - SimNIBS 4.x installed with the `ernie` CHARM head model
    - FreeSurfer `ernie` subject for coregistration
    - One or more processed EDF files (or internet access for EEGEyeNet download)
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Configuration
    """)
    return


@app.cell
def _(mo):
    pathfem_input = mo.ui.text(
        value="/Users/christian/Applications/SimNIBS-4.0/bin/",
        label="SimNIBS installation path (containing m2m_ernie/)",
        full_width=True,
    )
    subjects_dir_input = mo.ui.text(
        value="/Applications/freesurfer/7.2.0/subjects/",
        label="FreeSurfer subjects directory",
        full_width=True,
    )
    mo.vstack([pathfem_input, subjects_dir_input])
    return pathfem_input, subjects_dir_input


@app.cell
def _(mo):
    mo.md("""
    ## Step 1: Load the ernie SimNIBS head mesh

    Tissue segmentation was performed with the CHARM pipeline:
    ```bash
    charm ernie ernie_T1.nii.gz ernie_T2.nii.gz
    ```
    Surface extraction with FreeSurfer:
    ```bash
    recon-all -subjid ernie -i T1.nii.gz -T2 T2_reg.nii.gz -all
    ```
    """)
    return


@app.cell
def _(pathfem_input):
    from simnibs import read_msh, sim_struct
    subpath = "m2m_ernie"
    pathfem = pathfem_input.value
    msh_file = pathfem + subpath + "/ernie.msh"
    mesh = read_msh(msh_file)
    print(f"Loaded mesh: {mesh.nodes.node_coord.shape[0]} nodes")
    return mesh, msh_file, sim_struct


@app.cell
def _(mo):
    mo.md("""
    ## Step 2: Identify eye tissue and compute dipole positions

    Tissue tags: 1=WM, 2=GM, 3=CSF, 4=Bone, 5=Scalp, **6=Eye**, 7=Compact Bone,
    8=Spongy, 9=Blood, 10=Muscle
    """)
    return


@app.cell
def _(mesh, np, sim_struct):
    # Identify eye tissue elements (tag1 == 6)
    eye_ctr = mesh.elements_baricenters().value[mesh.elm.tag1 == 6]

    # Compute eye centers (left: x < 0, right: x > 0)
    left_center = eye_ctr[eye_ctr[:, 0] < 0].mean(0)
    right_center = eye_ctr[eye_ctr[:, 0] > 0].mean(0)

    dipole_positions = np.array([
        left_center, left_center, left_center,
        right_center, right_center, right_center,
    ])
    # 6 unit dipoles: x/y/z for each eye
    dipole_moments = np.array([
        [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 0, 0], [0, 1, 0], [0, 0, 1],
    ])

    # Tissue conductivities from SimNIBS defaults
    S = sim_struct.SESSION()
    tdcs = S.add_tdcslist()
    ids_tag = (mesh.elm.tag1 % 1000) - 1
    cond = np.zeros_like(ids_tag, dtype=float)
    for _i in range(10):
        cond[ids_tag == _i] = tdcs.cond[_i].value

    print(f"Left eye center:  {left_center}")
    print(f"Right eye center: {right_center}")
    return cond, dipole_moments, dipole_positions


@app.cell
def _(mo):
    mo.md("""
    ## Step 3: Run electric dipole simulation

    > ⚠️ **This step is computationally expensive** and may take 10–30 minutes.
    > Run it once; the result is saved to `leadfield.npy`.
    """)
    return


@app.cell
def _(cond, dipole_moments, dipole_positions, mesh, np):
    import os as _os
    electric_dipole = None
    source_model = "partial integration"
    if _os.path.exists("leadfield.npy"):
        sim = np.load("leadfield.npy")
        print(f"Loaded existing leadfield.npy  shape: {sim.shape}")
    else:
        from simnibs.simulation import electric_dipole
        sim = electric_dipole(
            mesh, cond, dipole_positions, dipole_moments,
            source_model, solver_options=None, units="mm",
        )
        np.save("leadfield.npy", sim)
        print(f"Saved leadfield.npy  shape: {sim.shape}")
    return (sim,)


@app.cell
def _(mo):
    mo.md("""
    ## Step 4: EEG electrode coregistration

    Co-register the GSN-HydroCel-129 electrode montage to the FreeSurfer
    `ernie` head using MNE's ICP coregistration, then project electrode
    positions onto the head surface.
    """)
    return


@app.cell
def _(mo):
    eeg_data_path_input = mo.ui.text(
        value="processed/",
        label="Path to processed EDF files (for montage reference)",
        full_width=True,
    )
    eeg_data_path_input
    return (eeg_data_path_input,)


@app.cell
def _(Path, eeg_data_path_input, eoglearn, mne):
    _edf_files = list(Path(eeg_data_path_input.value).glob("*_clean.edf"))
    if _edf_files:
        _raw_ref = mne.io.read_raw_edf(_edf_files[0], verbose=False)
        _montage = mne.channels.make_standard_montage("GSN-HydroCel-129")
        _raw_ref.set_montage(_montage)
        info_ref = _raw_ref.info
    else:
        # Fallback: use a downloaded EEGEyeNet file
        _fpath = eoglearn.datasets.fetch_eegeyenet(subject="EP10", run=1)
        _raw_ref = eoglearn.io.read_raw_eegeyenet(_fpath)
        _montage = mne.channels.make_standard_montage("GSN-HydroCel-129")
        _raw_ref.set_montage(_montage)
        info_ref = _raw_ref.info
    montage = _montage
    print(f"Montage has {len(montage.ch_names)} channels")
    return (info_ref,)


@app.cell
def _(info_ref, subjects_dir_input):
    from mne.coreg import Coregistration
    subject = "ernie"
    subjects_dir = subjects_dir_input.value
    coreg = Coregistration(info_ref, subject, subjects_dir, fiducials="estimated")
    coreg.fit_fiducials(verbose=True)
    coreg.fit_icp(n_iterations=6, nasion_weight=2.0, verbose=True)
    print("Coregistration complete.")
    return (coreg,)


@app.cell
def _(coreg, info_ref, mne, np):
    # Project electrode positions from head surface onto MRI coordinates
    from mne._freesurfer import _get_head_surface
    from mne.surface import _project_onto_surface

    _head_surf = _get_head_surface(
        "head", subject="ernie", subjects_dir=coreg._subjects_dir
    )
    _trans = coreg.trans
    _eeg_picks = mne.pick_types(info_ref, eeg=True)
    _eeg_pos = np.array([info_ref["chs"][p]["loc"][:3] for p in _eeg_picks])
    _eeg_pos_mri = mne.transforms.apply_trans(_trans, _eeg_pos)
    _, _, eegp_locs = _project_onto_surface(_eeg_pos_mri, _head_surf, project_rrs=True)
    print(f"Projected {len(eegp_locs)} electrode positions onto head surface")
    return (eegp_locs,)


@app.cell
def _(mo):
    mo.md("""
    ## Step 5: Map electrodes to nearest mesh nodes → save electrode leadfield
    """)
    return


@app.cell
def _(eegp_locs, mesh, np, sim, subjects_dir_input):
    from scipy.spatial import KDTree
    from mne.surface import read_surface
    import os

    # Get CRS offset from FreeSurfer outer skin surface
    _fname_surf = os.path.join(
        subjects_dir_input.value, "ernie/bem/outer_skin.surf"
    )
    try:
        _meta = read_surface(_fname_surf, return_dict=True, read_metadata=True)[2]
        _cras = _meta["cras"]
    except Exception:
        _cras = np.zeros(3)

    sensor_points = np.vstack(eegp_locs) * 1000 + _cras

    tree = KDTree(mesh.nodes.node_coord)
    electrode_ids = [tree.query(qp, k=1)[1] for qp in sensor_points]

    leadfield_elect_only = sim[:, electrode_ids]
    np.save("leadfield_elect_only.npy", leadfield_elect_only)
    print(f"Saved leadfield_elect_only.npy  shape: {leadfield_elect_only.shape}")
    return electrode_ids, sensor_points


@app.cell
def _(mo):
    mo.md("""
    ## Step 6: 3D head visualization (scalp + eyes + colour-coded electrodes)
    """)
    return


@app.cell
def _(
    electrode_ids,
    get_eye_mesh,
    get_scalp_mesh,
    msh_file,
    plot_head,
    plotly_sphere,
    sensor_points,
    sim,
):
    import matplotlib.colors as mcolors
    import matplotlib

    scalp_trace = get_scalp_mesh(msh_file, color="lightgrey", opacity=0.7)
    eye_trace = get_eye_mesh(msh_file)
    _meshes = [scalp_trace, eye_trace]

    _values = sim[:, electrode_ids].mean(0)
    _norm = mcolors.Normalize(vmin=_values.min(), vmax=_values.max())
    _cmap = matplotlib.colormaps["plasma"]
    _rgba = _cmap(_norm(_values))
    _colors = [
        f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})"
        for r, g, b, _ in _rgba
    ]
    for _center, _color in zip(sensor_points, _colors):
        _meshes.append(
            plotly_sphere(_center, radius=5.0, resolution=20, color=_color, opacity=1.0)
        )
    fig_3d = plot_head(_meshes)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Step 7: Validation — compare simulated vs. real EOG signals

    Use the `get_sim_eog` function from `analyses.py` (which loads `leadfield_elect_only.npy`)
    to generate a simulated EOG for a sample subject and visually compare it to
    the recorded EEG frontal channels.
    """)
    return


@app.cell
def _(mo):
    val_subject = mo.ui.text(value="EP10", label="Subject")
    val_run = mo.ui.text(value="1", label="Run")
    mo.hstack([val_subject, val_run])
    return val_run, val_subject


@app.cell
def _(val_run, val_subject):
    from analyses import get_sim_eog
    raw_sim_val, raw_val = get_sim_eog(
        val_subject.value, int(val_run.value), return_raw=True
    )
    print(f"Simulated raw:  {raw_sim_val}")
    print(f"Original raw:   {raw_val}")
    return raw_sim_val, raw_val


@app.cell
def _(mne, optimal_alpha, plt, raw_sim_val, raw_val):
    from eoglearn.viz import overlay_raws_stack
    _EOG_CH = [
        "E127", "E126", "E17", "E21", "E14",
        "E25", "E22", "E15", "E16", "E9", "E8",
    ]
    _x_sim = raw_sim_val.get_data(picks="eeg")
    _x_raw = raw_val.get_data(picks="eeg")
    _alpha = optimal_alpha(_x_sim, _x_raw)
    _raw_sim_scaled = mne.io.RawArray(_x_sim * _alpha, raw_val.copy().pick("eeg").info)

    # Gaze angle plots
    _gaze = raw_val.get_data(picks=["L-GAZE-X", "L-GAZE-Y"])
    fig_gaze, ax_gaze = plt.subplots(figsize=(12, 3))
    ax_gaze.plot(raw_val.times[:3000], _gaze[0, :3000], label="Gaze X")
    ax_gaze.plot(raw_val.times[:3000], _gaze[1, :3000], label="Gaze Y")
    ax_gaze.set_xlabel("Time (s)")
    ax_gaze.set_ylabel("Gaze (px)")
    ax_gaze.legend()
    fig_gaze.tight_layout()
    fig_gaze
    return (overlay_raws_stack,)


@app.cell
def _(mne, optimal_alpha, overlay_raws_stack, raw_sim_val, raw_val):
    _EOG_CH = [
        "E127", "E126", "E17", "E21", "E14",
        "E25", "E22", "E15", "E16", "E9", "E8",
    ]
    _x_sim = raw_sim_val.get_data(picks="eeg")
    _x_raw = raw_val.get_data(picks="eeg")
    _alpha = optimal_alpha(_x_sim, _x_raw)
    _raw_sim_scaled = mne.io.RawArray(_x_sim * _alpha, raw_val.copy().pick("eeg").info)

    fig_overlay, _ = overlay_raws_stack(
        [raw_val, _raw_sim_scaled],
        picks=_EOG_CH,
        start=0, duration=20,
        labels=["Original EEG", "Simulated EOG (scaled)"],
        title="Original vs Simulated EOG — frontal channels",
    )
    fig_overlay
    return


if __name__ == "__main__":
    app.run()
