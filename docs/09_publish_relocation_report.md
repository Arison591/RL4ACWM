# Publish relocation report

After explicit user authorization, all 45 untracked TempFlow research files were moved from the publish
checkout into `legacy_source/`: algorithm modules, seven configs, five documents, five launch scripts,
twelve tests and three tools. The TempFlow-only README section was removed while the unrelated security
edit was preserved.

The publish checkout now has zero TempFlow path matches and zero case-insensitive TempFlow content
matches outside `.git`. Its remaining six modified tracked files are non-TempFlow pre-existing work.
The private pre-migration archive remains the recovery source for the original layout.
