# Upstream integrity report

- Pinned clean checkout: `dce69e48a952449e873a791812e506df878bc8a9` (detached, clean).
- Expected experiment branch and base commit were verified.
- Claimed audit commit `63e4def7159940ba7d60e4e6250eee868342388c` is not present in the source object database or a full clone of configured origin. Its provenance remains a blocker for official-code audit claims.
- The source repository is never added to `sys.path`; only the clean checkout is accepted.
- `audit_upstream()` rejects a dirty or differently pinned checkout.

