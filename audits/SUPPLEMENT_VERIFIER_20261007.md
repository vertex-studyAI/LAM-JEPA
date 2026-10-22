# Anonymous supplement verification — 7 October 2026

The previous personal-email pattern used double-escaped regular-expression boundaries. It could miss an ordinary mailbox address. The archive verifier also accepted extra unmanifested files and duplicate member names. Those gaps matter when an anonymous source supplement is reviewed independently of its builder.

The verifier now checks ordinary personal and institutional email addresses in both contents and member names, exact manifest coverage, unique names, safe regular-file paths, UTF-8 content, and well-formed byte/hash records. It rejects modified, unbound, ambiguous, and unscannable members. No research runner, frozen protocol, source model, manuscript, or result file changes.

## Verification

```sh
python -m unittest discover -s tests -p test_anonymous_bundle_integrity.py -v
python -m unittest discover -s tests -p test_retained_row_integrity.py -v
python scripts/paper/build_aistats2027_anonymous_bundle.py --output /tmp/rechecked-supplement.zip
```

All 12 archive counterexample tests and 16 existing retained-row tests pass. The real supplement was built before and after the verifier change using frozen scientific source commit `760aa7f9a73a177d5ff4ba7eb470f7e68ace63cb`. Both archives are byte-identical: 143,225 bytes, SHA-256 `2e48c739df51acff2eea0c20974b57fc4d9306f5ef5959a2b382809abf800714`.

A historical Git object is required for the full bundle build. A shallow clone must first retrieve that exact source revision. The dependency-free counterexample suite uses explicitly synthetic file-content fixtures and requires no historic download or experiment execution.

The corresponding JSON receipt binds this check to the inspected repository base and records the unchanged baseline-file count. The generated ZIP is not a new submitted artifact. Historical missing archives, scientific scope, publication metadata, and permission for a future submission remain separate from this verifier result.
