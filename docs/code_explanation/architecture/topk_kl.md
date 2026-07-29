# Top-k + Tail KL

## Teacher forward

teacher logits から各 response 位置の `logsumexp` を計算し、teacher 自身の
top-k value/token ID を取得する。top-k value から同じ位置の logsumexp を
引くことで、full-vocabulary softmax に対する teacher top-k log-prob を得る。

```text
input_ids:             (batch, sequence_length)
responses:             (batch, response_length)
teacher_topk_ids:      (batch, response_length, k)
teacher_topk_logprobs: (batch, response_length, k)
```

remove-padding 経路では `(batch,sequence_length)` を `unpad_input` で
`(total_nnz)` 系へ変換し、計算後に `pad_input` で batch/sequence 配置へ戻す。
vocabulary ID は bf16 では 256 を超える整数を正確に表せないため、shared
padding 経路を float32 で通し、`round().long()` で int64 に戻す。

## Student gather

student は teacher と同じ `teacher_topk_ids` で logits を gather する。
したがって student/teacher の第3軸は同一 token support である。

```text
student_topk_logprobs: (batch, response_length, k)
```

student log-prob は gradient を保持し、teacher log-prob は upstream と
`topk_kl_per_token()` の双方で detach される。

## KL と集約

top-k 内では `p_s * (log p_s - log p_t)` を k 軸で合計する。top-k 外の全
vocabulary は `1 - sum(top-k probability)` の tail bucket 1個へ集約し、
同じ KL 項を加える。`eps` clamp は `log(0)` を防ぐ。

```text
teacher_kld:  (batch, response_length)
response_mask:(batch, response_length)
scalar:       token-mean(teacher_kld, response_mask)
```

最後に `teacher_kl_loss_coef` を掛けて `policy_loss` へ加え、backward する。
これは teacher top-k 全点を使う dense approximation であり、student が実際に
sample した1 token だけを使う `low_var_kl` estimator とは別物である。

現在の guard は top-k forward と fused kernel / Ulysses SP の組み合わせを
未対応として拒否する。top-k が post-forward logits と同一 token ID gather を
必要とし、既存 fused/SP 経路の出力契約にその情報がないためである。
