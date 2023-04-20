### fMRI-timescales
> mapping the spatial gradients of timescales in the Human Connectome Project (HCP)

### requirements
[nilearn](https://github.com/nilearn/nilearn)
[neurodsp](https://github.com/neurodsp-tools/neurodsp)
[notebook](https://github.com/jupyter/notebook) (if running examples)
[pytest](https://github.com/pytest-dev/pytest/) (if running unit tests)

### fMRI data
The complete HCP dataset can be accessed at [ConnectomeDB](https://db.humanconnectome.org/app/template/Login.vm;jsessionid=07FB635B78590A7214F2DB247B57D052), which includes resting-state fMRI and behavioral measures for 812 subjects. The `HCP1200 Parcellation+Timeseries+Netmats` download (10GB) includes denoised timeseries from parcelling group-level ICA maps at various spatial scales. More documentation can be found [here](https://www.humanconnectome.org/storage/app/media/documentation/s1200/HCP1200-DenseConnectome+PTN+Appendix-July2017.pdf).  

For example purposes, a subset of HCP data can be accessed as a numpy array at `./examples/data/nsubjects-10_nregions-300_hcp.npy` in the shape `(n_subjects, n_regions, n_timepoints)`.  

In addition, semi-realistic fMRI timeseries can be simulated using the `src.sim.sim_fmri()` function, which can generate timeseries with specific auto- and cross-correlation structures (temporal and spatial correlation). 
