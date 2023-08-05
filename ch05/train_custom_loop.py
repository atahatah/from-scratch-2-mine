import sys
sys.path.append('..')
import matplotlib.pyplot as plt
import numpy as np
from common.optimizer import SGD
from dataset import ptb
from simple_rnnlm import SimpleRnnlm

# settings of hyper-parameters
batch_size = 100
wordvec_size = 150
hidden_size = 150 # the number of elements of RNN's hidden state vector
time_size = 10
lr = 0.15
max_epoch = 1000

# load learning data (minimize the dataset)
corpus, word_to_id, id_to_word = ptb.load_data('train')
corpus_size = 929589
corpus = corpus[:corpus_size]
vocab_size = int(max(corpus) + 1)

xs = corpus[:-1] # input
ts = corpus[1:]  # output (supervise labels)
data_size = len(xs)
print('corpus size: %d, vocabulary size: %d' % (corpus_size, vocab_size))

# variables used when learning
max_iters = data_size // (batch_size * time_size)
time_idx = 0
total_loss = 0
loss_count = 0
ppl_list = []

# generation of the model
model = SimpleRnnlm(vocab_size, wordvec_size, hidden_size)
optimizer = SGD(lr)

# calculate the start position to load each samples of the mini-batch
jump = (corpus_size - 1) // batch_size
offsets = [i * jump for i in range(batch_size)]

for epoch in range(max_epoch):
    for iter in range(max_iters):
        # get the mini-batch
        batch_x = np.empty((batch_size, time_size), dtype='i')
        batch_t = np.empty((batch_size, time_size), dtype='i')
        for t in range(time_size):
            for i, offset in enumerate(offsets):
                batch_x[i, t] = xs[(offset + time_idx) % data_size]
                batch_t[i, t] = ts[(offset + time_idx) % data_size]
            time_idx += 1
        
        # calculate the gradient and update the parameters
        loss = model.forward(batch_x, batch_t)
        model.backward()
        optimizer.update(model.params, model.grads)
        total_loss += loss
        loss_count += 1
    
    # evaluate perplexity on each epoch
    ppl = np.exp(total_loss / loss_count)
    print('| epoch %d | perplexity %.2f' % (epoch+1, ppl))
    ppl_list.append(float(ppl))
    if ppl < 4.0:
        break
    total_loss, loss_count = 0, 0

model.save_params()

print('******************************************************************')
model.reset_state()

word_ids = model.generate(word_to_id["you"])

for id in word_ids:
    word = id_to_word[id]
    if word != "<eos>":
        print(' '+word, end='')
    else:
        print('.')