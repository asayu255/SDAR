# Adapted from https://github.com/XiaoxiaoGuo/rcdqn/blob/master/agents/nn/networks.py
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch.nn as nn
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch.nn.functional as F
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from .modules import EncoderRNN, BiAttention, get_aggregated, duplicate


# [EXPLAIN] `RCDQN` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class RCDQN(nn.Module):
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, vocab_size, embedding_dim, hidden_dim, arch, grad, embs=None, gru_embed='embedding', get_image=0, bert_path=''):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        super().__init__()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.word_dim = embedding_dim
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.word_emb = nn.Embedding(vocab_size, embedding_dim)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if embs is not None:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print('Loading embeddings of shape {}'.format(embs.shape))
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.word_emb.weight.data.copy_(torch.from_numpy(embs))
            # self.word_emb.weight.requires_grad = False
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.hidden_dim = hidden_dim
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.keep_prob = 1.0
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.rnn = EncoderRNN(self.word_dim, self.hidden_dim, 1,
                              concat=True,
                              bidir=True,
                              layernorm='None',
                              return_last=False)
        # self.rnn = AutoModelForMaskedLM.from_pretrained("google/bert_uncased_L-4_H-128_A-2").bert
        # self.linear_bert = nn.Linear(128, 256)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.att_1 = BiAttention(self.hidden_dim * 2, 1 - self.keep_prob)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.att_2 = BiAttention(self.hidden_dim * 2, 1 - self.keep_prob)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.att_3 = BiAttention(embedding_dim, 1 - self.keep_prob)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.linear_1 = nn.Sequential(
            nn.Linear(self.hidden_dim * 8, self.hidden_dim),
            nn.LeakyReLU())

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.rnn_2 = EncoderRNN(self.hidden_dim, self.hidden_dim, 1,
                                concat=True,
                                bidir=True,
                                layernorm='layer',
                                return_last=False)
        # self.self_att = BiAttention(self.hidden_dim * 2, 1 - self.keep_prob)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.linear_2 = nn.Sequential(
            nn.Linear(self.hidden_dim * 12, self.hidden_dim * 2),
            nn.LeakyReLU())

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.linear_3 = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, 1),
        )

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.get_image = get_image
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.get_image:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.linear_image = nn.Linear(512, self.hidden_dim)

    # [EXPLAIN] `prepare` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def prepare(self, ids):
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """
        Prepare the input for the encoder. Pass it through pad, embedding, and rnn.
        """
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        lens = [len(_) for _ in ids]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        ids = [torch.tensor(_) for _ in ids]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        ids = nn.utils.rnn.pad_sequence(ids, batch_first=True).cuda()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        mask = (ids > 0).float()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        embed = self.word_emb(ids)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        output = self.rnn(embed, lens)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return ids, lens, mask, embed, output
    
    # [EXPLAIN] `forward` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def forward(self, state_batch, act_batch, value=False, q=False, act=False):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.arch == 'bert':
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return self.bert_forward(state_batch, act_batch, value, q, act)

        # state representation
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        obs_ids, obs_lens, obs_mask, obs_embed, obs_output = self.prepare([state.obs for state in state_batch])
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        goal_ids, goal_lens, goal_mask, goal_embed, goal_output = self.prepare([state.goal for state in state_batch])

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        state_output = self.att_1(obs_output, goal_output, goal_mask)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        state_output = self.linear_1(state_output)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.get_image:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            images = [state.image_feat for state in state_batch]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            images = [torch.zeros(512) if _ is None else _ for _ in images] 
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            images = torch.stack([_ for _ in images]).cuda()  # BS x 512
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            images = self.linear_image(images)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            state_output = torch.cat([images.unsqueeze(1), state_output], dim=1)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            obs_lens = [_ + 1 for _ in obs_lens]
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            obs_mask = torch.cat([obs_mask[:, :1], obs_mask], dim=1)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        state_output = self.rnn_2(state_output, obs_lens)
        
        # state value
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if value:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            values = get_aggregated(state_output, obs_lens, 'mean')
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            values = self.linear_3(values).squeeze(1)

        # action
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        act_sizes = [len(_) for _ in act_batch]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        act_batch = list(itertools.chain.from_iterable(act_batch))
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        act_ids, act_lens, act_mask, act_embed, act_output = self.prepare(act_batch)

        # duplicate based on action sizes
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        state_output, state_mask, state_lens = duplicate(state_output, obs_mask, obs_lens, act_sizes)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        goal_embed, goal_mask, goal_lens = duplicate(goal_embed, goal_mask, goal_lens, act_sizes)

        # full contextualized 
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        state_act_output = self.att_2(act_output, state_output, state_mask)

        # based on goal and action tokens
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        goal_act_output = self.att_3(act_embed, goal_embed, goal_mask)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        output = torch.cat([state_act_output, goal_act_output], dim=-1)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        output = get_aggregated(output, act_lens, 'mean')
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        output = self.linear_2(output)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        act_values = self.linear_3(output).squeeze(1)
        # Log softmax
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not q:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            act_values = torch.cat([F.log_softmax(_, dim=0) for _ in act_values.split(act_sizes)], dim=0)

        # Optionally, output state value prediction
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if value:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return act_values, act_sizes, values
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return act_values, act_sizes



