# coding: utf-8
import sys
sys.path.append('..')
from common.optimizer import SGD
from common.trainer import RnnlmTrainer
from dataset import ptb
from simple_rnnlm import SimpleRnnlm
from common.util import preprocess

# ハイパーパラメータの設定
batch_size = 1
wordvec_size = 2
hidden_size = 2  # RNNの隠れ状態ベクトルの要素数
time_size = 4  # RNNを展開するサイズ
lr = 0.05
max_epoch = 150

# 学習データの読み込み
corpus, word_to_id, id_to_word = preprocess('You say goodbye and I say hello <eos>' * 10)
# corpus_size = 1000  # テスト用にデータセットを小さくする
# corpus = corpus[:corpus_size]
print(corpus)
print(id_to_word)
vocab_size = int(max(corpus) + 1)
xs = corpus[:-1]  # 入力
ts = corpus[1:]  # 出力（教師ラベル）

# モデルの生成
model = SimpleRnnlm(vocab_size, wordvec_size, hidden_size)
optimizer = SGD(lr)
trainer = RnnlmTrainer(model, optimizer)

trainer.fit(xs, ts, max_epoch, batch_size, time_size)
# trainer.plot()

model.reset_state()
start_word = "you"
start_id = word_to_id[start_word]

word_ids = model.generate(start_id)
text = ' '.join([id_to_word[i] for i in word_ids])
text = text.replace(' <eos>', '.\n')
print(text)
