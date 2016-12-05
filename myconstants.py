# My settings and Constants


DEFAULT_PUNCTS = ['\'']
KEEP_STOPWORDS = ['against', 'not', 'no', 'very', 'just', 'over', 'only']
TOTAL_REVIEWS = 400000
HUNDRED = 100

CLASSES = [[1,2],[3]]
CLASS_LEN = 80000
CLASS_LEN_TEST = 100

DATA_DIR = 'database/'

DATA_NAME = 'review80000_5.json'
TRAIN_NAME = 'train_data.json'
VALIDATE_NAME ='validate_data.json'
ACCURACY_NAME = VALIDATE_NAME

DATA_PATH = DATA_DIR + DATA_NAME
TRAIN_PATH = DATA_DIR + TRAIN_NAME
VALIDATE_PATH = DATA_DIR + VALIDATE_NAME

# Test database
DATA_TEST_DIR = DATA_DIR + 'test/'

DATA_PATH_TEST = DATA_TEST_DIR + DATA_NAME
TRAIN_PATH_TEST = DATA_TEST_DIR + TRAIN_NAME
VALIDATE_PATH_TEST = DATA_TEST_DIR + VALIDATE_NAME
ACCURACY_PATH_TEST = DATA_TEST_DIR + ACCURACY_NAME

# Use small data to validate in progross accuracy
ACCURACY_PATH = ACCURACY_PATH_TEST

# Word 2 vector
W2V_DIM = 200
TOTAL_REVIEWS_TEST = 500

W2V_DIR = 'word2vec/'

W2V_SRC_NAME = 'reviews.txt'
W2V_MODEL_NAME = 'w2v_model'
W2V_MODEL_TXT = 'w2v_model.txt'

W2V_SRC_PATH = DATA_DIR + W2V_SRC_NAME
W2V_MODEL_PATH = W2V_DIR + W2V_MODEL_NAME
W2V_MODEL_TXT_PATH = W2V_DIR + W2V_MODEL_TXT

# Test database
W2V_TEST_DIR = W2V_DIR + 'test/'

W2V_SRC_PATH_TEST = DATA_TEST_DIR + W2V_SRC_NAME
W2V_MODEL_PATH_TEST = W2V_TEST_DIR + W2V_MODEL_NAME
W2V_MODEL_TXT_PATH_TEST = W2V_TEST_DIR + W2V_MODEL_TXT


# Doc 2 vector
D2V_DIM = 300

D2V_DIR = 'doc2vec/'

D2V_SRC_NAME = W2V_SRC_NAME
D2V_MODEL_NAME = 'd2v_model'

D2V_SRC_PATH = DATA_DIR + D2V_SRC_NAME
D2V_MODEL_PATH = D2V_DIR + D2V_MODEL_NAME

D2V_TEST_DIR = D2V_DIR + 'test/'

D2V_SRC_PATH_TEST = DATA_TEST_DIR + D2V_SRC_NAME
D2V_MODEL_PATH_TEST = D2V_TEST_DIR + D2V_MODEL_NAME


MATRIX_DIR = DATA_DIR + 'matrix/'

NUMPY_W = 'w'
NUMPY_B = 'b'

def get_constant(name, test):
  return eval(name + '_TEST') if test else eval(name)


def get_save_name(name, mode, classified=True, classification=CLASSES):
  path = MATRIX_DIR + name + str(mode)
  if classified:
    for cla in classification:
      path += '_'

      for rating in cla:
        path += str(rating)

  path += '.npy'

  # import pdb; pdb.set_trace()
  return path

class Mode():
  PCA_COMPONENTS = 4

  MODE_PCA = 0
  MODE_MEAN = 1
  MODE_D2V = 2

