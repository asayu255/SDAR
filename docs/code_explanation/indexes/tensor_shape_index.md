# Tensor shape索引

`B`: turn-row batch、`S`: prompt+response sequence、`R`: response長、`K`: teacher top-k、`V`: vocabulary、`nnz`: paddingを除いたtoken数。

| tensor | shape | dtype | meaning |
|---|---|---|---|
| `input_ids` | `(B, S)` | integer token id | student/teacher共通入力 |
| `responses` | `(B, R)` | integer token id | student on-policy response |
| `attention_mask` | `(B, S)` | bool/int mask | paddingを除外 |
| `response_mask` | `(B, R)` | bool/int mask | 有効response tokenだけを集約 |
| `teacher_topk_ids` | `(B, R, K)` | float32→long | padding後にround/longへ復元 |
| `teacher_topk_logprobs` | `(B, R, K)` | floating | teacher top-k log-softmax |
| `teacher_logsumexp` | `(B, R)` | floating | tail mass算出の正規化項 |
| `student_topk_logprobs` | `(B, R, K)` | floating | teacher token ID位置をgather |
| `teacher_kld` | `(B, R)` | floating | top-k support + tail bucket KL |
| `task_ids` | `(B,)` | long | worker側per-task row mask |
| `indices` / `unpad_input` | `(nnz,)` | long | padded tokenからrmpadへの写像 |
| `input_ids_rmpad` | `(1, nnz)` | integer token id | FlashAttention varlen入力 |
| `logits_rmpad` | `(1, nnz, V)` | floating | pad_input前のmodel出力 |

teacher token IDはvocabulary indexのためbf16では256を超える整数を全て厳密表現できない。DP auto-padding時はfloat32 bufferを経由し、scatter後に`round().long()`でindexへ戻す。
