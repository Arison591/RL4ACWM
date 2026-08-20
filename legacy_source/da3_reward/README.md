# Vendored DA3 Mono reward core

This package is the Haoran DA3 Mono implementation copied from the local
`/home/ma-user/modelarts/user-job-dir/code/haoran/da3_reward` working tree on
2026-08-20. That directory was untracked at Haoran repository commit
`32f3340401c1be3dbf3e06ac81b22e8ee8bd2db4`, so the commit is recorded as
context rather than falsely claimed as the source revision of these files.

TempFlow-specific video loading and reward-schema adaptation live in
`experiments/tempflow_video/da3_video_reward.py`. The DA3 model source and
weights remain external; they are never committed to this repository.
