import gensim
from myconstants import get_constant, W2V_DIM


class MySentences(object):
  def __init__(self, dirname):
    self.dirname = dirname

  def __iter__(self):
      for line in open(self.dirname):
        yield line.split()


def w2v_train(test=True):
  sentence_path = get_constant('W2V_SRC_PATH', test)
  model_path = get_constant('W2V_MODEL_PATH', test)
  model_txt_path = get_constant('W2V_MODEL_TXT_PATH', test)

  sentences = MySentences(sentence_path) # a memory-friendly iterator
  model = gensim.models.Word2Vec(sentences, size=W2V_DIM)
  model.save(model_path)
  model.save_word2vec_format(model_txt_path, binary=False)
