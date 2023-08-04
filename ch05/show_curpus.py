import sys
sys.path.append('..')
from dataset import ptb

# load learning data (minimize the dataset)
corpus, word_to_id, id_to_word = ptb.load_data('train')
# corpus_size = 1000
# corpus = corpus[:corpus_size]
vocab_size = int(max(corpus) + 1)

xs = corpus[:-1] # input
ts = corpus[1:]  # output (supervise labels)
data_size = len(xs)
# print('corpus size: %d, vocabulary size: %d' % (corpus_size, vocab_size))

sentences = [[]]
sentence_num = 0
for id in corpus:
    word = id_to_word[id]
    if(word != '<eos>'):
        sentences[sentence_num].append(word)
    else:
        sentences[sentence_num].append('.')
        sentence_num += 1
        sentences.append([])

for sentence in sentences:
    for word in sentence:
        print(word + ' ', end='')
    print()

print()
print("num of sentences: "+sentence_num.__str__())
print("size of corpus: "+corpus.size.__str__())