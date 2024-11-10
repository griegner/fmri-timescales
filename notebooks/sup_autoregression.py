import nibabel as nb
import numpy as np
from scipy import ndimage, signal


def _scale_gaussian_variance(std):
    """
    Variance scaling for smoothing with 1D Gaussian filter.
    """
    limit = std * 5
    x = np.arange(-limit, limit + 1)
    gaussian_1d = np.exp(-(x**2) / (2 * std**2))
    gaussian_1d = gaussian_1d / gaussian_1d.sum()  # sum = 1
    gaussian_1d_cov = signal.fftconvolve(gaussian_1d, gaussian_1d[::-1], mode="full")
    return gaussian_1d_cov[gaussian_1d_cov.size // 2]


def smooth_signal(X_wn, std):
    """Smooth signal/timeseries by applying a Gaussian filter.

    Parameters
    ----------
    X_wn : ndarray of shape (n_timepoints, )
        The input white noise signal/timeseries.
    std : float
        The standard deviation of the smoothing kernel

    Returns
    -------
    X_smoothed : ndarray of shape (n_timepoints, )
        The smoothed image, with the same mean and std of the input.
    """
    std_scaling_factor = np.sqrt(1 / _scale_gaussian_variance(std))
    X_smoothed = ndimage.gaussian_filter1d(X_wn, sigma=std)
    return X_smoothed * std_scaling_factor


"""https://nbviewer.org/github/neurohackademy/nh2020-curriculum/blob/master/we-nibabel-markiewicz/NiBabel.ipynb"""


def cifti_to_volume(data, axis):
    """Convert CIFTI subcortex data to NIFTI"""
    assert isinstance(axis, nb.cifti2.BrainModelAxis)
    data = data.T[axis.volume_mask]
    vox_indices = tuple(axis.voxel[axis.volume_mask].T)  # ([x0, x1, ...], [y0, ...], [z0, ...])
    vol_data = np.zeros(axis.volume_shape + data.shape[1:], dtype=data.dtype)
    vol_data[vox_indices] = data
    return nb.Nifti1Image(vol_data, axis.affine)


def cifti_to_surface(data, axis, surf_name):
    """Convert CIFTI cortex data to GIFTI"""
    assert isinstance(axis, nb.cifti2.BrainModelAxis)
    for (
        name,
        data_indices,
        model,
    ) in axis.iter_structures():  # iterate over volumetric and surface structures
        if name == surf_name:
            data = data.T[data_indices]
            vtx_indices = model.vertex  # 1-N, except medial wall vertices
            surf_data = np.zeros((vtx_indices.max() + 1,) + data.shape[1:], dtype=data.dtype)
            surf_data[vtx_indices] = data
            return surf_data
    raise ValueError(f"No structure named {surf_name}")


def split_cifti(cifti):
    """Split CIFTI into subcortex, cortex left, cortex right"""
    data = cifti.get_fdata(dtype=np.float32)
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    axis = cifti.header.get_axis(1)
    return (
        cifti_to_volume(data, axis),
        cifti_to_surface(data, axis, "CIFTI_STRUCTURE_CORTEX_LEFT"),
        cifti_to_surface(data, axis, "CIFTI_STRUCTURE_CORTEX_RIGHT"),
    )
