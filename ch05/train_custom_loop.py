import sys
sys.path.append('..')
import matplotlib.pyplot as plt
import numpy as np
from common.optimizer import SGD
from dataset import ptb
from simple_rnnlm import SimpleRnnlm

# settings of hyper-parameters
batch_size = 10
wordvec_size = 100
hidden_size = 100 # the number of elements of RNN's hidden state vector
time_size = 5
lr = 0.1
max_epoch = 100

# load learning data (minimize the dataset)
corpus, word_to_id, id_to_word = ptb.load_data('train')
corpus_size = 1000
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
    # print('| epoch %d | perplexity %.2f' % (epoch+1, ppl))
    ppl_list.append(float(ppl))
    total_loss, loss_count = 0, 0

print('******************************************************************')
model.reset_state()
words = ['you', 'said', 'u.s.', 'and']
ids = []
for word in words:
    ids.append(word_to_id[word])

print('-----------------------ids------------------------')
print(ids)

scores = model.evaluate(np.array([ids]))
print('-----------------------scores------------------------')
print(scores)
print('-----------------------result------------------------')
result = np.argmax(scores, axis=2)
print(result)

for id in result[0]:
    print(id_to_word[id])