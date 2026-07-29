# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import itertools
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch.nn as nn
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch.nn.functional as F
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from torch.nn.utils import rnn


# [EXPLAIN] `duplicate` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def duplicate(output, mask, lens, act_sizes):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Duplicate the output based on the action sizes.
    """
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    output = torch.cat([output[i:i+1].repeat(j, 1, 1) for i, j in enumerate(act_sizes)], dim=0)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    mask = torch.cat([mask[i:i+1].repeat(j, 1) for i, j in enumerate(act_sizes)], dim=0)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    lens = list(itertools.chain.from_iterable([lens[i:i+1] * j for i, j in enumerate(act_sizes)]))
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return output, mask, lens


# [EXPLAIN] `get_aggregated` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_aggregated(output, lens, method):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Get the aggregated hidden state of the encoder.
    B x D
    """
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if method == 'mean':
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return torch.stack([output[i, :j, :].mean(0) for i, j in enumerate(lens)], dim=0)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif method == 'last':
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return torch.stack([output[i, j-1, :] for i, j in enumerate(lens)], dim=0)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif method == 'first':
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return output[:, 0, :]


# [EXPLAIN] `EncoderRNN` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class EncoderRNN(nn.Module):
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, input_size, num_units, nlayers, concat,
                 bidir, layernorm, return_last):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        super().__init__()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.layernorm = (layernorm == 'layer')
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if layernorm:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.norm = nn.LayerNorm(input_size)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.rnns = []
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i in range(nlayers):
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if i == 0:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                input_size_ = input_size
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                output_size_ = num_units
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                input_size_ = num_units if not bidir else num_units * 2
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                output_size_ = num_units
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.rnns.append(
                nn.GRU(input_size_, output_size_, 1,
                       bidirectional=bidir, batch_first=True))

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.rnns = nn.ModuleList(self.rnns)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.init_hidden = nn.ParameterList(
            [nn.Parameter(
                torch.zeros(size=(2 if bidir else 1, 1, num_units)),
                requires_grad=True) for _ in range(nlayers)])
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.concat = concat
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.nlayers = nlayers
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.return_last = return_last

        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.reset_parameters()

    # [EXPLAIN] `reset_parameters` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def reset_parameters(self):
        # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
        with torch.no_grad():
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for rnn_layer in self.rnns:
                # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
                for name, p in rnn_layer.named_parameters():
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    if 'weight_ih' in name:
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        torch.nn.init.xavier_uniform_(p.data)
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    elif 'weight_hh' in name:
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        torch.nn.init.orthogonal_(p.data)
                    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                    elif 'bias' in name:
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        p.data.fill_(0.0)
                    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
                    else:
                        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                        p.data.normal_(std=0.1)

    # [EXPLAIN] `get_init` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def get_init(self, bsz, i):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self.init_hidden[i].expand(-1, bsz, -1).contiguous()

    # [EXPLAIN] `forward` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def forward(self, inputs, input_lengths=None):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        bsz, slen = inputs.size(0), inputs.size(1)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.layernorm:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            inputs = self.norm(inputs)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        output = inputs
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        outputs = []
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        lens = 0
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if input_lengths is not None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            lens = input_lengths  # .data.cpu().numpy()
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for i in range(self.nlayers):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            hidden = self.get_init(bsz, i)
            # output = self.dropout(output)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if input_lengths is not None:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                output = rnn.pack_padded_sequence(output, lens,
                                                  batch_first=True,
                                                  enforce_sorted=False)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            output, hidden = self.rnns[i](output, hidden)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if input_lengths is not None:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                output, _ = rnn.pad_packed_sequence(output, batch_first=True)
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if output.size(1) < slen:
                    # used for parallel
                    # padding = Variable(output.data.new(1, 1, 1).zero_())
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    padding = torch.zeros(
                        size=(1, 1, 1), dtype=output.type(),
                        device=output.device())
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    output = torch.cat(
                        [output,
                         padding.expand(
                             output.size(0),
                             slen - output.size(1),
                             output.size(2))
                         ], dim=1)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if self.return_last:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                outputs.append(
                    hidden.permute(1, 0, 2).contiguous().view(bsz, -1))
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                outputs.append(output)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.concat:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return torch.cat(outputs, dim=2)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return outputs[-1]


# [EXPLAIN] `BiAttention` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class BiAttention(nn.Module):
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, input_size, dropout):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        super().__init__()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.dropout = nn.Dropout(dropout)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.input_linear = nn.Linear(input_size, 1, bias=False)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.memory_linear = nn.Linear(input_size, 1, bias=False)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.dot_scale = nn.Parameter(
            torch.zeros(size=(input_size,)).uniform_(1. / (input_size ** 0.5)),
            requires_grad=True)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.init_parameters()

    # [EXPLAIN] `init_parameters` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def init_parameters(self):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return

    # [EXPLAIN] `forward` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def forward(self, context, memory, mask):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        bsz, input_len = context.size(0), context.size(1)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        memory_len = memory.size(1)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        context = self.dropout(context)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        memory = self.dropout(memory)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        input_dot = self.input_linear(context)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        memory_dot = self.memory_linear(memory).view(bsz, 1, memory_len)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        cross_dot = torch.bmm(
            context * self.dot_scale,
            memory.permute(0, 2, 1).contiguous())
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        att = input_dot + memory_dot + cross_dot
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        att = att - 1e30 * (1 - mask[:, None])

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        weight_one = F.softmax(att, dim=-1)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        output_one = torch.bmm(weight_one, memory)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        weight_two = (F.softmax(att.max(dim=-1)[0], dim=-1)
                      .view(bsz, 1, input_len))
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        output_two = torch.bmm(weight_two, context)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return torch.cat(
            [context, output_one, context * output_one,
             output_two * output_one],
            dim=-1)