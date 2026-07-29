# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import os
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import sys
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import shutil
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import os.path as osp
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import json
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import time
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import datetime
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import tempfile
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from collections import defaultdict
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import wandb

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
DEBUG = 10
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
INFO = 20
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
WARN = 30
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
ERROR = 40

# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
DISABLED = 50


# [EXPLAIN] `KVWriter` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class KVWriter(object):
    # [EXPLAIN] `writekvs` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def writekvs(self, kvs):
        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
        raise NotImplementedError


# [EXPLAIN] `SeqWriter` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class SeqWriter(object):
    # [EXPLAIN] `writeseq` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def writeseq(self, seq):
        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
        raise NotImplementedError


# [EXPLAIN] `HumanOutputFormat` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class HumanOutputFormat(KVWriter, SeqWriter):
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, filename_or_file):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if isinstance(filename_or_file, str):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.file = open(filename_or_file, 'wt')
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.own_file = True
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
            assert hasattr(filename_or_file, 'read'), 'expected file or str, got %s' % filename_or_file
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.file = filename_or_file
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.own_file = False

    # [EXPLAIN] `writekvs` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def writekvs(self, kvs):
        # Create strings for printing
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        key2str = {}
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for (key, val) in sorted(kvs.items()):
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if isinstance(val, float):
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                valstr = '%-8.3g' % (val,)
            # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
            else:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                valstr = str(val)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            key2str[self._truncate(key)] = self._truncate(valstr)

        # Find max widths
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if len(key2str) == 0:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            print('WARNING: tried to write empty key-value dict')
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            keywidth = max(map(len, key2str.keys()))
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            valwidth = max(map(len, key2str.values()))

        # Write out the data
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        dashes = '-' * (keywidth + valwidth + 7)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        lines = [dashes]
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for (key, val) in sorted(key2str.items()):
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            lines.append('| %s%s | %s%s |' % (
                key,
                ' ' * (keywidth - len(key)),
                val,
                ' ' * (valwidth - len(val)),
            ))
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        lines.append(dashes)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.file.write('\n'.join(lines) + '\n')

        # Flush the output to the file
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.file.flush()

    # [EXPLAIN] `_truncate` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _truncate(self, s):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return s[:20] + '...' if len(s) > 23 else s

    # [EXPLAIN] `writeseq` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def writeseq(self, seq):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        seq = list(seq)
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for (i, elem) in enumerate(seq):
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.file.write(elem)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if i < len(seq) - 1:  # add space unless this is the last one
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                self.file.write(' ')
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.file.write('\n')
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.file.flush()

    # [EXPLAIN] `close` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def close(self):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.own_file:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.file.close()


# [EXPLAIN] `JSONOutputFormat` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class JSONOutputFormat(KVWriter):
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, filename):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.file = open(filename, 'wt')

    # [EXPLAIN] `writekvs` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def writekvs(self, kvs):
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for k, v in sorted(kvs.items()):
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if hasattr(v, 'dtype'):
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                v = v.tolist()
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                kvs[k] = float(v)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.file.write(json.dumps(kvs) + '\n')
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.file.flush()

    # [EXPLAIN] `close` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def close(self):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.file.close()


# [EXPLAIN] `WandBOutputFormat` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class WandBOutputFormat(KVWriter):
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, filename):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        group = None
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if filename.endswith('trial'):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            group = filename[:-6]
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        wandb.init(project='web_drrn', name=filename, group=group)

    # [EXPLAIN] `writekvs` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def writekvs(self, kvs):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        wandb.log(kvs)

    # [EXPLAIN] `close` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def close(self):
        # [EXPLAIN] 現在の分岐または反復の制御を明示する。
        pass


# [EXPLAIN] `CSVOutputFormat` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class CSVOutputFormat(KVWriter):
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, filename):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.file = open(filename, 'w+t')
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.keys = []
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.sep = ','

    # [EXPLAIN] `writekvs` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def writekvs(self, kvs):
        # Add our current row to the history
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        extra_keys = kvs.keys() - self.keys
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if extra_keys:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.keys.extend(extra_keys)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.file.seek(0)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            lines = self.file.readlines()
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.file.seek(0)
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for (i, k) in enumerate(self.keys):
                # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
                if i > 0:
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    self.file.write(',')
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                self.file.write(k)
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.file.write('\n')
            # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
            for line in lines[1:]:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                self.file.write(line[:-1])
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                self.file.write(self.sep * len(extra_keys))
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                self.file.write('\n')
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for (i, k) in enumerate(self.keys):
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if i > 0:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                self.file.write(',')
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            v = kvs.get(k)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if v is not None:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                self.file.write(str(v))
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.file.write('\n')
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.file.flush()

    # [EXPLAIN] `close` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def close(self):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.file.close()


# [EXPLAIN] `TensorBoardOutputFormat` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class TensorBoardOutputFormat(KVWriter):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Dumps key/value pairs into TensorBoard's numeric format.
    """

    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, dir):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        os.makedirs(dir, exist_ok=True)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.dir = dir
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.step = 1
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        prefix = 'events'
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        path = osp.join(osp.abspath(dir), prefix)
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        import tensorflow as tf
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from tensorflow.python import pywrap_tensorflow
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from tensorflow.core.util import event_pb2
        # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
        from tensorflow.python.util import compat
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.tf = tf
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.event_pb2 = event_pb2
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.pywrap_tensorflow = pywrap_tensorflow
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.writer = pywrap_tensorflow.EventsWriter(compat.as_bytes(path))

    # [EXPLAIN] `writekvs` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def writekvs(self, kvs):
        # [EXPLAIN] `summary_val` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
        def summary_val(k, v):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            kwargs = {'tag': k, 'simple_value': float(v)}
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return self.tf.Summary.Value(**kwargs)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        summary = self.tf.Summary(value=[summary_val(k, v) for k, v in kvs.items()])
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        event = self.event_pb2.Event(wall_time=time.time(), summary=summary)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        event.step = self.step  # is there any reason why you'd want to specify the step?
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.writer.WriteEvent(event)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.writer.Flush()
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        self.step += 1

    # [EXPLAIN] `close` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def close(self):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.writer:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self.writer.Close()
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.writer = None


# [EXPLAIN] `make_output_format` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def make_output_format(format, ev_dir, log_suffix='', args=None):
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    os.makedirs(ev_dir, exist_ok=True)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if format == 'stdout':
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return HumanOutputFormat(sys.stdout)
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif format == 'log':
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return HumanOutputFormat(osp.join(ev_dir, 'log%s.txt' % log_suffix))
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif format == 'json':
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return JSONOutputFormat(osp.join(ev_dir, 'progress%s.json' % log_suffix))
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif format == 'csv':
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return CSVOutputFormat(osp.join(ev_dir, 'progress%s.csv' % log_suffix))
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif format == 'tensorboard':
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return TensorBoardOutputFormat(osp.join(ev_dir, 'tb%s' % log_suffix))
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif format == 'wandb':
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return WandBOutputFormat(ev_dir)
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
        raise ValueError('Unknown format specified: %s' % (format,))


# ================================================================
# API
# ================================================================

# [EXPLAIN] `logkv` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def logkv(key, val):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Log a value of some diagnostic
    Call this once for each diagnostic quantity, each iteration
    If called many times, last value will be used.
    """
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    Logger.CURRENT.logkv(key, val)


# [EXPLAIN] `logkv_mean` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def logkv_mean(key, val):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    The same as logkv(), but if called many times, values averaged.
    """
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    Logger.CURRENT.logkv_mean(key, val)


# [EXPLAIN] `logkvs` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def logkvs(d):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Log a dictionary of key-value pairs
    """
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for (k, v) in d.items():
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        logkv(k, v)


# [EXPLAIN] `dumpkvs` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def dumpkvs():
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Write all of the diagnostics from the current iteration

    level: int. (see logger.py docs) If the global logger level is higher than
                the level argument here, don't print to stdout.
    """
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    Logger.CURRENT.dumpkvs()


# [EXPLAIN] `getkvs` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def getkvs():
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return Logger.CURRENT.name2val


# [EXPLAIN] `log` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def log(*args, level=INFO):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Write the sequence of args, with no separators, to the console and output files (if you've configured an output file).
    """
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    Logger.CURRENT.log(*args, level=level)


# [EXPLAIN] `debug` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def debug(*args):
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    log(*args, level=DEBUG)


# [EXPLAIN] `info` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def info(*args):
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    log(*args, level=INFO)


# [EXPLAIN] `warn` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def warn(*args):
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    log(*args, level=WARN)


# [EXPLAIN] `error` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def error(*args):
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    log(*args, level=ERROR)


# [EXPLAIN] `set_level` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def set_level(level):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Set logging threshold on current logger.
    """
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    Logger.CURRENT.set_level(level)


# [EXPLAIN] `get_dir` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def get_dir():
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Get directory that log files are being written to.
    will be None if there is no output directory (i.e., if you didn't call start)
    """
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return Logger.CURRENT.get_dir()


# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
record_tabular = logkv
# [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
dump_tabular = dumpkvs


# [EXPLAIN] `ProfileKV` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class ProfileKV:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Usage:
    with logger.ProfileKV("interesting_scope"):
        code
    """

    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, n):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.n = "wait_" + n

    # [EXPLAIN] `__enter__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __enter__(self):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.t1 = time.time()

    # [EXPLAIN] `__exit__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __exit__(self, type, value, traceback):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        Logger.CURRENT.name2val[self.n] += time.time() - self.t1


# [EXPLAIN] `profile` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def profile(n):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Usage:
    @profile("my_func")
    def my_func(): code
    """

    # [EXPLAIN] `decorator_with_name` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def decorator_with_name(func):
        # [EXPLAIN] `func_wrapper` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
        def func_wrapper(*args, **kwargs):
            # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
            with ProfileKV(n):
                # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                return func(*args, **kwargs)

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return func_wrapper

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return decorator_with_name


# ================================================================
# Backend
# ================================================================

# [EXPLAIN] `Logger` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class Logger(object):
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    DEFAULT = None  # A logger with no output files. (See right below class definition)
    # So that you can still log to the terminal without setting up any output files
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    CURRENT = None  # Current logger being used by the free functions above

    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, dir, output_formats):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.name2val = defaultdict(float)  # values this iteration
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.name2cnt = defaultdict(int)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.level = INFO
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.dir = dir
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.output_formats = output_formats

    # Logging API, forwarded
    # ----------------------------------------
    # [EXPLAIN] `logkv` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def logkv(self, key, val):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.name2val[key] = val

    # [EXPLAIN] `logkv_mean` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def logkv_mean(self, key, val):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if val is None:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.name2val[key] = None
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        oldval, cnt = self.name2val[key], self.name2cnt[key]
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.name2val[key] = oldval * cnt / (cnt + 1) + val / (cnt + 1)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.name2cnt[key] = cnt + 1

    # [EXPLAIN] `dumpkvs` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def dumpkvs(self):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.level == DISABLED: return
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for fmt in self.output_formats:
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if isinstance(fmt, KVWriter):
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                fmt.writekvs(self.name2val)
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.name2val.clear()
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        self.name2cnt.clear()

    # [EXPLAIN] `log` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def log(self, *args, level=INFO):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.level <= level:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            self._do_log(args)

    # Configuration
    # ----------------------------------------
    # [EXPLAIN] `set_level` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def set_level(self, level):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.level = level

    # [EXPLAIN] `get_dir` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def get_dir(self):
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self.dir

    # [EXPLAIN] `close` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def close(self):
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for fmt in self.output_formats:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            fmt.close()

    # Misc
    # ----------------------------------------
    # [EXPLAIN] `_do_log` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _do_log(self, args):
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for fmt in self.output_formats:
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if isinstance(fmt, SeqWriter):
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                fmt.writeseq(map(str, args))


# [EXPLAIN] `configure` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def configure(dir=None, format_strs=None):
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if dir is None:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        dir = os.getenv('OPENAI_LOGDIR')
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if dir is None:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        dir = osp.join(tempfile.gettempdir(),
                       datetime.datetime.now().strftime("openai-%Y-%m-%d-%H-%M-%S-%f"))
    # [EXPLAIN] 後続処理が依存する shape、dtype、設定または分散条件を検証する。
    assert isinstance(dir, str)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    os.makedirs(dir, exist_ok=True)

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    log_suffix = ''
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    rank = 0
    # check environment variables here instead of importing mpi4py
    # to avoid calling MPI_Init() when this module is imported
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for varname in ['PMI_RANK', 'OMPI_COMM_WORLD_RANK']:
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if varname in os.environ:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            rank = int(os.environ[varname])
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if rank > 0:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        log_suffix = "-rank%03i" % rank

    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if format_strs is None:
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if rank == 0:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            format_strs = os.getenv('OPENAI_LOG_FORMAT', 'stdout,log,csv').split(',')
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            format_strs = os.getenv('OPENAI_LOG_FORMAT_MPI', 'log').split(',')
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    format_strs = filter(None, format_strs)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    output_formats = [make_output_format(f, dir, log_suffix) for f in format_strs]

    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    Logger.CURRENT = Logger(dir=dir, output_formats=output_formats)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    log('Logging to %s' % dir)


# [EXPLAIN] `_configure_default_logger` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _configure_default_logger():
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    format_strs = None
    # keep the old default of only writing to stdout
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if 'OPENAI_LOG_FORMAT' not in os.environ:
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        format_strs = ['stdout']
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    configure(format_strs=format_strs)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    Logger.DEFAULT = Logger.CURRENT


# [EXPLAIN] `reset` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def reset():
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if Logger.CURRENT is not Logger.DEFAULT:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        Logger.CURRENT.close()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        Logger.CURRENT = Logger.DEFAULT
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        log('Reset logger')


# [EXPLAIN] `scoped_configure` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class scoped_configure(object):
    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, dir=None, format_strs=None):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.dir = dir
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.format_strs = format_strs
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.prevlogger = None

    # [EXPLAIN] `__enter__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __enter__(self):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.prevlogger = Logger.CURRENT
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        configure(dir=self.dir, format_strs=self.format_strs)

    # [EXPLAIN] `__exit__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __exit__(self, *args):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        Logger.CURRENT.close()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        Logger.CURRENT = self.prevlogger


# ================================================================

# [EXPLAIN] `_demo` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _demo():
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    info("hi")
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    debug("shouldn't appear")
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    set_level(DEBUG)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    debug("should appear")
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    dir = "/tmp/testlogging"
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if os.path.exists(dir):
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        shutil.rmtree(dir)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    configure(dir=dir)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    logkv("a", 3)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    logkv("b", 2.5)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    dumpkvs()
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    logkv("b", -2.5)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    logkv("a", 5.5)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    dumpkvs()
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    info("^^^ should see a = 5.5")
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    logkv_mean("b", -22.5)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    logkv_mean("b", -44.4)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    logkv("a", 5.5)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    dumpkvs()
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    info("^^^ should see b = 33.3")

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    logkv("b", -2.5)
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    dumpkvs()

    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    logkv("a", "longasslongasslongasslongasslongasslongassvalue")
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    dumpkvs()


# ================================================================
# Readers
# ================================================================

# [EXPLAIN] `read_json` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def read_json(fname):
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    import pandas
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    ds = []
    # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
    with open(fname, 'rt') as fh:
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for line in fh:
            # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
            ds.append(json.loads(line))
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return pandas.DataFrame(ds)


# [EXPLAIN] `read_csv` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def read_csv(fname):
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    import pandas
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return pandas.read_csv(fname, index_col=None, comment='#')


# [EXPLAIN] `read_tb` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def read_tb(path):
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    path : a tensorboard file OR a directory, where we will find all TB files
           of the form events.*
    """
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    import pandas
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    import numpy as np
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from glob import glob
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    from collections import defaultdict
    # [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
    import tensorflow as tf
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if osp.isdir(path):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        fnames = glob(osp.join(path, "events.*"))
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    elif osp.basename(path).startswith("events."):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        fnames = [path]
    # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
    else:
        # [EXPLAIN] 不正な設定・shape・task または実行状態を例外として明示する。
        raise NotImplementedError("Expected tensorboard file or directory containing them. Got %s" % path)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    tag2pairs = defaultdict(list)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    maxstep = 0
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for fname in fnames:
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for summary in tf.train.summary_iterator(fname):
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if summary.step > 0:
                # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
                for v in summary.summary.value:
                    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                    pair = (summary.step, v.simple_value)
                    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                    tag2pairs[v.tag].append(pair)
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                maxstep = max(summary.step, maxstep)
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    data = np.empty((maxstep, len(tag2pairs)))
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    data[:] = np.nan
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    tags = sorted(tag2pairs.keys())
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for (colidx, tag) in enumerate(tags):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        pairs = tag2pairs[tag]
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for (step, value) in pairs:
            # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
            data[step - 1, colidx] = value
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return pandas.DataFrame(data, columns=tags)


# configure the default logger on import
# _configure_default_logger()

# [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
if __name__ == "__main__":
    # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
    _demo()
