# Reproducibility

The repository now exposes the following reproducible workflow:

- fixed seeding via `lam_jepa.utils.set_seed`
- checkpoint save/load including RNG state
- seed sweep scripts
- aggregation scripts with confidence intervals
- clear task definitions for all benchmark families

Recommended reporting format:

- mean ± std across seeds
- 95% bootstrap confidence intervals
- paired permutation tests for ablations
- task-wise breakdowns for OOD and ed-tech subsets
