# Superseded exploratory CV output — not frozen

This is preserved from the first complete Qwen train-only CV run, before requiring
fold-consistent improvement. Its exploratory rule selected class-weighted noul from
a small aggregate OOF gain despite improvements in only 3/5 folds. Do not use this
as the frozen method. The final conservative selection is in
`../qwen-clean300-multi120-trainonly/selection.json` and its hash-locked
`frozen-config.json`.
