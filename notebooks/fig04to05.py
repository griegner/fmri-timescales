from pathlib import Path

import nibabel as nib
import numpy as np


def scale(X, axis=0):
    "return zero mean, unit variance X"
    return (X - X.mean(axis=axis)) / X.std(axis=axis)


def get_X(dtseries_paths):
    "load and scale CIFTI images"
    for dtseries_path in dtseries_paths:
        img = nib.cifti2.load(dtseries_path)
        X = scale(img.get_fdata(dtype=np.float32))
        yield X


def cifti_to_volume(data, axis):
    """convert CIFTI subcortex data to NIFTI"""
    assert isinstance(axis, nib.cifti2.BrainModelAxis)
    data = data.T[axis.volume_mask]
    vox_indices = tuple(axis.voxel[axis.volume_mask].T)  # ([x0, x1, ...], [y0, ...], [z0, ...])
    vol_data = np.zeros(axis.volume_shape + data.shape[1:], dtype=data.dtype)
    vol_data[vox_indices] = data
    return nib.Nifti1Image(vol_data, axis.affine)


def se_decomposition(stderrs, taus):
    """std error decomposition by the law of total variance"""
    var_within = np.mean(stderrs**2, axis=0)
    var_between = np.var(taus, axis=0)
    return np.sqrt(var_within + var_between)


def get_maps(estimator):
    """get maps for LLS or NLS estimator"""
    taus = np.load(f"data/nsubjects-180_nregions-91282_estimator-{estimator}_tau.npy").astype(np.float32)
    stderrs = np.load(f"data/nsubjects-180_nregions-91282_estimator-{estimator}_se.npy").astype(np.float32)

    maps = {  # subject #100610
        "tau": taus[0, :],
        "se(tau)": stderrs[0, :],
        "tstat": (taus[0, :] - 0.5) / stderrs[0, :],
        "rse": stderrs[0, :] / taus[0, :],
    }
    maps["tau_N"] = taus.mean(axis=0)
    maps["se(tau_N)"] = se_decomposition(stderrs, taus)
    maps["tstat_N"] = (maps["tau_N"] - 0.5) / maps["se(tau_N)"]
    maps["rse_N"] = maps["se(tau_N)"] / maps["tau_N"]

    return maps


def get_vmax(arr1, arr2):
    """return the midpoints of the 99th-percentile of the two arrays"""
    perc1 = np.nanpercentile(arr1, 99)
    perc2 = np.nanpercentile(arr2, 99)
    return np.round((perc1 + perc2) / 2, 1)


def fit_timescale_models(input_path, output_path, lls, nls):
    """fit LLS and NLS to individual subject maps"""
    assert isinstance(input_path, Path) and isinstance(output_path, Path), "must be pathlib objects"
    output_path.mkdir(parents=True, exist_ok=True)

    for sub_path in input_path.glob("sub-*"):
        print("...", sub_path)
        sub = sub_path.stem[4:]
        dtseries_paths = sorted(sub_path.glob("*.dtseries.nii"))
        X = np.vstack(list(get_X(dtseries_paths)))

        lls_ = lls.fit(X, n_timepoints=len(X)).estimates_
        np.save(output_path / f"sub-{sub}_task-rest_estimator-lls_tau.npy", lls_["tau"])
        np.save(output_path / f"sub-{sub}_task-rest_estimator-lls_se.npy", lls_["se(tau)"])

        nls_ = nls.fit(X, n_timepoints=len(X)).estimates_
        np.save(output_path / f"sub-{sub}_task-rest_estimator-nls_tau.npy", nls_["tau"])
        np.save(output_path / f"sub-{sub}_task-rest_estimator-nls_se.npy", nls_["se(tau)"])
