import sys
sys.path.append('..')
import matplotlib.pyplot as plt
import numpy as np
from common.optimizer import SGD
from dataset import ptb
from simple_rnnlm import SimpleRnnlm

# settings of hyper-parameters
wordvec_size = 100
hidden_size = 100 # the number of elements of RNN's hidden state vector

# load learning data (minimize the dataset)
corpus, word_to_id, id_to_word = ptb.load_data('train')
vocab_size = int(max(corpus) + 1)

# generation of the model
model = SimpleRnnlm(vocab_size, wordvec_size, hidden_size)

model.load_params()

model.reset_state()

skip_words = ['<unk>']
skip_ids = [word_to_id[w] for w in skip_words]

start_word = "you"
start_id = word_to_id[start_word]

word_ids = model.generate(start_id, skip_ids=skip_ids)

txt = ' '.join([id_to_word[i] for i in word_ids])
txt = txt.replace(' <eos>', '.\n')
print(txt)