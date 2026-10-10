# Beam planner v2 development contract

State: SMOKE_PASSED (14 planner tests); synthetic software verification only. Frozen ARC-v5 source,
metrics, seed policy, claim ledger and unopened confirmatory test remain at
their recorded boundary. The old `planner.py` remains available unchanged.

The old public planner unpacks four values although `LatentActionModel.step`
returns six. Even with a four-value transition it generates only one child per
parent, so the initially single beam never branches. Sorting candidate scores
by batch mean would additionally couple distinct examples if branching were
added without a corresponding state representation change.

Import `beam_plan_v2` from `lam_jepa.planner_v2` explicitly. Each retained parent
expands every discrete action using the existing `action_override` API with
deterministic transitions. Scores retain the declared 0.6 value + 0.4 verifier
minus 0.01 latent-norm objective. Beam ranking, ancestry and final actions are
independent for each batch row. Exact ties preserve parent order then ascending
action ID. Temperature is passed through the transition API; with all actions
considered, policy probabilities are not a separate score term.

Search runs without gradients and temporarily switches the transition and heads
to evaluation mode, restoring every mixed child training flag on success or
failure. Candidate/state storage is bounded before expansion. Non-finite states
or scores, ignored action overrides, and malformed shapes are errors.

The state-element budget is a pre-allocation estimate for planner tensors, not
a measured peak-memory guarantee; temporary arrays and model internals add
memory. The search-depth budget limits the number of expansion rounds.

Tests use a hand-constructed depth-two tree where greedy search misses the best
path, check complete ancestry, require batched/single-example equivalence, run
the actual six-output model, and verify that parameters and RNG state remain
unchanged. Run `PYTHONPATH=src python -m pytest tests/test_planner_v2.py -q`.

This is a successor algorithm implementation, not evidence that planning helps
ARC reasoning or educational outcomes. Any such experiment needs a new frozen
protocol and valid evaluation design before outcome access.
