# Development repair: paired statistical report

Date: 2026-10-10. Base: `3a6462753de89b85d5867c8e67fa87fa1eb2b13d`. Evidence class: engineering regression verification.

## Failure and scientific scope

The report combined an A-only mean and interval with an A-versus-B paired test. A fixed absolute permutation tolerance changed results with units; a variance floor manufactured finite standardized effects. Independent review then reproduced finite-input overflow at 1e308.

ARC confirmatory test remains unopened. This utility repair does not modify the bespoke frozen ARC analysis scripts, retained outputs, manuscripts, or prior scientific conclusion.

## Implementation contract

Report the paired mean difference and its paired bootstrap interval; keep method means separate; validate aligned finite one-dimensional pairs and resampling parameters; use relative normalized sign-flip tails; return null for undefined standardized effects; normalize mean, variance and quantile reductions before restoring units.

A constant paired difference has a constant interval even when the individual methods vary. Scaling units preserves the sampled sign-flip p-value; representable extreme means and intervals stay finite.

## Verification and retained failures

The exact reproduction command, environment versions, source SHA-256 identities, test output, and exit status are in `research/verification/paired_statistical_report_20261010/receipt.json`. The canonical state retains the base-source regression failure counts and observed review failures. This record was written after exploratory defect discovery, and is not a preregistered confirmatory experiment.

An independent agent examined the modified source and regression assertions. For LAM's paired-report repair, that review found an extreme-value overflow, which was fixed and added to the regression suite before publication. This is project-controlled review, not external scientific replication.

## Limits and next action

The p-value remains a Monte Carlo sign-flip test with a plus-one correction, not exact enumeration. Validity requires independent aligned units and sign-exchangeability. The percentile bootstrap does not establish those assumptions. Unrepresentable paired differences or sample SD are rejected.

Closed as a reproducible negative / failure-mechanism study; scope is the frozen tested configuration.

- Review utility compatibility: mean now denotes the paired difference, method means have separate fields, and effect_size can be null when undefined.
- Keep the completed negative ARC evidence unchanged.
- Any scientific architecture repair or stronger claim requires a separately versioned successor hypothesis, new falsifier and new protected protocol.
