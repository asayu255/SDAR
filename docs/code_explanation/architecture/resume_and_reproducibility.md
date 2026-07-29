# checkpoint再開と再現性

再開時に復元すべき状態は model/optimizer/scheduler だけではない。dataloader position、
global step、環境内部の episode schedule、seed と epoch の組み合わせを揃える必要がある。

Search の episode は dataloader の `env_kwargs` から供給されるため、loader位置の復元が中心になる。
WebShop と AlfWorld は process/Ray actor 内で seed 付き schedule を再構築するため、単純な再起動では
先頭 episode へ戻る。`fast_forward_env_schedules` は完了済み global step 数だけ stateful leaf env
を進め、連続実行時の次 episode へ追いつかせる。

この処理は best effort である。leaf が `fast_forward` を公開しない stateless 環境はskipし、
例外はmessageとして記録して学習を継続する。その場合は model state が正しくても episode順の
完全再現性は保証されない。

task-balanced sampler は seed と epoch から task内shuffleを決める。各taskの不足分をcycleして
同数抽出するため、再開step・epoch・distributed rank-local slicing の三者を揃える必要がある。
filter/retryにより一global stepでreset回数が変わる設定は、環境fast-forwardの前提外である。
