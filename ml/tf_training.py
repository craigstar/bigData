import gensim
import numpy as np
import json
import tensorflow as tf
import scipy.spatial.distance as dis
import itertools

import myconstants
from myconstants import get_constant, get_save_name, CLASS_LEN, CLASS_LEN_TEST
import functions as f
from d2v import d2v_sentence


CLASSES = {j:idx for idx, cla in enumerate(myconstants.CLASSES) for j in cla}
TOTAL_CLASSES = len(myconstants.CLASSES)

CHOICES = []
for c in CLASSES:
  CHOICES += range((c-1)*CLASS_LEN, c*CLASS_LEN)

CHOICES_TEST = []
for c in CLASSES:
  CHOICES_TEST += range((c-1)*CLASS_LEN_TEST, c*CLASS_LEN_TEST)


def sentence_mean(sentence, model):
  sentence = sentence.split()

  vec_sum = []
  count = 0

  for word in sentence:
    try:
      if not len(vec_sum):
        vec_sum = model[word]
        count += 1
      else:
        vec_sum = np.add(vec_sum, model[word])
        count += 1
    except:
      pass
  return np.divide(vec_sum, count) if count else None



def sentence_pca(sentence, model):
  from sklearn.decomposition import PCA

  sentence = sentence.split()
  words = []

  for word in sentence:
    try:
      if not len(words):
        words = [model[word]]
      else:
        words = np.append(words, [model[word]], axis=0)
    except:
      pass

  if len(words) < 4:
    return

  words_t = np.transpose(words)
  pca = PCA(n_components=4)
  pca.fit(words_t)
  pca_m = pca.transform(words_t)
  pca_m = pca_m.flatten()
  # print pca_m
  return pca_m


def sentence_d2v(sentence, current_i, model, datatype='TRAIN'):
  if datatype in ['TRAIN', 'TEST']:
    sentence = model.docvecs[datatype + '_' + str(current_i)]
  else:
    sentence = model.infer_vector(sentence, alpha=0.025, min_alpha=0.025)
  # print sentence
  return sentence


def one_hot(rating):
  # import pdb; pdb.set_trace()
  rating_vec = np.zeros([TOTAL_CLASSES])
  hot_pos = CLASSES[int(rating)]
  rating_vec[hot_pos] = 1

  return rating_vec


def get_validate_data(file_path, model, mode):
  validate_data_file = open(file_path, 'r')

  validate_reviews = []
  validate_ratings = []

  for idx, line in enumerate(validate_data_file):
    line = line.strip()
    data = json.loads(line)

    rating = data['overall']

    if mode == myconstants.Mode.MODE_PCA:
      sen_represent = sentence_pca(data['reviewText'], model)
    elif mode == myconstants.Mode.MODE_MEAN:
      sen_represent = sentence_mean(data['reviewText'], model)
    elif mode == myconstants.Mode.MODE_D2V:
      sen_represent = sentence_d2v(data['reviewText'], idx, model, 'TEST')

    # import pdb; pdb.set_trace()
    if sen_represent is None or not int(rating) in CLASSES:
      continue

    rating = one_hot(rating)

    validate_reviews.append(sen_represent)
    validate_ratings.append(rating)

  return [np.array(validate_ratings), np.array(validate_reviews)]


def form_sen_list(sen_list, sen_represent, current_rating, max_count):
  full_flag = True
  for rating, sens in sen_list.iteritems():
    if rating == current_rating and len(sens) < max_count:
      sen_list[rating].append(sen_represent)
    elif len(sens) < max_count:
      full_flag = False

  return sen_list, full_flag


def print_comparison(sen_list, max_count):
  np.set_printoptions(precision=6, suppress=True)
  compare_matrix = np.zeros([5,5])

  for i in range(5):
    for j in range(i, 5):
      compare_matrix[i][j] = 1 - dis.cosine(sen_list[i+1][-1], sen_list[j+1][-1])

  for i in range(5):
    pairs = itertools.combinations(range(max_count), 2)
    print 'rating: {}'.format(i+1)

    for pair in pairs:
      sim = 1 - dis.cosine(sen_list[i+1][pair[0]], sen_list[i+1][pair[1]])
      print '{}: {}'.format(pair, sim)
  print compare_matrix
  return


def test_similarity(stop_at=None, max_count=2, test=True, mode=myconstants.Mode.MODE_PCA):
  sentence_path = get_constant('TRAIN_PATH', test)
  model_path = get_constant('W2V_MODEL_PATH', test=False)

  model = gensim.models.Word2Vec.load(model_path)

  sen_list = {i: [] for i in range(1, 6)}

  count = 1

  train_file = open(sentence_path, 'r')
  for line in train_file:
    line = line.strip()
    data = json.loads(line)

    current_rating = data['overall']
    sentence = data['reviewText']

    if mode == myconstants.Mode.MODE_PCA:
      sen_represent = sentence_pca(sentence, model)
    elif mode == myconstants.Mode.MODE_MEAN:
      sen_represent = sentence_mean(sentence, model)

    sen_list, full_flag = form_sen_list(sen_list, sen_represent, current_rating, max_count)


    if full_flag or (stop_at and stop_at == count):
      print_comparison(sen_list, max_count)
      print count
      break

    count += 1

  return


def next_batch(size, test, model, mode):
  from random import choice
  if test:
    choices = CHOICES_TEST
  else:
    choices = CHOICES

  p = [choice(choices) for n in range(size)]

  sentence_path = get_constant('TRAIN_PATH', test)

  mean_batch = []
  rating_batch = []

  with open(sentence_path, 'r') as train_file:
    lines = train_file.readlines()

    for i in p:
      current_i = i
      sen_represent = None
      rating_valid = False

      while sen_represent == None:
        line = lines[current_i]
        data = json.loads(line)

        rating = data['overall']
        sentence = data['reviewText']

        if mode == myconstants.Mode.MODE_PCA:
          sen_represent = sentence_pca(sentence, model)
        elif mode == myconstants.Mode.MODE_MEAN:
          sen_represent = sentence_mean(sentence, model)
        elif mode == myconstants.Mode.MODE_D2V:
          # import pdb; pdb.set_trace()
          sen_represent = sentence_d2v(sentence, current_i, model)

        current_i = choice(choices)

      rating_vec = one_hot(rating)

      mean_batch.append(sen_represent)
      rating_batch.append(rating_vec)
  return [np.array(mean_batch), np.array(rating_batch)]


def tf_train(test=True, learning_rate=0.5, mode=myconstants.Mode.MODE_MEAN, loops=1000):
  accuracy_path = get_constant('ACCURACY_PATH', test)
  w2v_model_path = get_constant('W2V_MODEL_PATH', test)
  d2v_model_path = get_constant('D2V_MODEL_PATH', test)

  # total_num_reviews = f.num_of_reviews(sentence_path)
  # total_num_reviews = myconstants.TOTAL_REVIEWS
  # unit_percent_reviews = int(total_num_reviews / HUNDRED / 10)


  # Load model

  if mode == myconstants.Mode.MODE_PCA:
    vec_dim = myconstants.Mode.PCA_COMPONENTS * myconstants.W2V_DIM
    model = gensim.models.Word2Vec.load(w2v_model_path)
  elif mode == myconstants.Mode.MODE_MEAN:
    vec_dim = myconstants.W2V_DIM
    model = gensim.models.Word2Vec.load(w2v_model_path)
  elif mode == myconstants.Mode.MODE_D2V:
    vec_dim = myconstants.D2V_DIM
    model = gensim.models.Doc2Vec.load(d2v_model_path)

  accuracy_ratings, accuracy_reviews = get_validate_data(accuracy_path, model, mode)


  # tf training graph
  graph = tf.Graph()
  with graph.as_default():
    sess = tf.InteractiveSession()

    try:
      W = tf.Variable(np.load(get_save_name(myconstants.NUMPY_W, mode)))
      b = tf.Variable(np.load(get_save_name(myconstants.NUMPY_B, mode)))
    except:
      W = tf.Variable(tf.truncated_normal([vec_dim, TOTAL_CLASSES]))
      b = tf.Variable(tf.truncated_normal([TOTAL_CLASSES]))
      print('New {}: {}'.format(mode, myconstants.CLASSES))

    x = tf.placeholder(tf.float32, [None, vec_dim])
    y = tf.matmul(x, W) + b

    y_ = tf.placeholder(tf.float32, [None, TOTAL_CLASSES])

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
    # cross_entropy += tf.reduce_mean(tf.square(b))
    # cross_entropy += tf.reduce_mean(tf.square(W))
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    init = tf.initialize_all_variables()
    sess.run(init)

    count = 1
    progress = 0

    for line in range(loops):
      mean_batch, rating_batch = next_batch(100, test, model, mode)

      sess.run(train_step, feed_dict={x: mean_batch, y_: rating_batch})

      if count % 10 == 0:
        print_value = "percent test accuracy {}, ".format(accuracy.eval(feed_dict={x: accuracy_reviews, y_: accuracy_ratings}))
        print_value += 'loop: {}'.format(count)
        print(print_value)
      count += 1

    print('{}: {}'.format(mode, myconstants.CLASSES))
    np.save(get_save_name(myconstants.NUMPY_W, mode), sess.run(W))
    np.save(get_save_name(myconstants.NUMPY_B, mode), sess.run(b))

  return


def predict(sentence, classification, vec_dim, mode=myconstants.Mode.MODE_MEAN):
  sentence.shape = [1, vec_dim]

  graph = tf.Graph()
  with graph.as_default():
    sess = tf.Session()
    x = tf.Variable(sentence)

    try:
      W = tf.Variable(np.load(get_save_name(myconstants.NUMPY_W, mode=mode, classification=classification)))
      b = tf.Variable(np.load(get_save_name(myconstants.NUMPY_B, mode=mode, classification=classification)))
    except:
      print 'you have not trained your model yet!'

    y = tf.argmax(tf.nn.softmax(tf.matmul(x, W) + b), 1)

    init = tf.initialize_all_variables()
    sess.run(init)
    result = sess.run(y)
    total_class = classification
    return total_class[result]



def form_classes(classes):
  formed_classes = []
  for cla in classes:
    formed_classes.append([cla])
  return formed_classes


def tolerate_class(prediction):
  minimum = max(1, prediction[0]-1)
  maximum = min(5, prediction[0]+1)
  return [minimum, prediction[0], maximum]


def test_cascade(mode=myconstants.Mode.MODE_MEAN, classification=None):
  test=False

  accuracy_path = get_constant('ACCURACY_PATH', test)
  w2v_model_path = get_constant('W2V_MODEL_PATH', test)
  d2v_model_path = get_constant('D2V_MODEL_PATH', test)

  # Load model
  if mode == myconstants.Mode.MODE_PCA:
    vec_dim = myconstants.Mode.PCA_COMPONENTS * myconstants.W2V_DIM
    model = gensim.models.Word2Vec.load(w2v_model_path)
  elif mode == myconstants.Mode.MODE_MEAN:
    vec_dim = myconstants.W2V_DIM
    model = gensim.models.Word2Vec.load(w2v_model_path)
  elif mode == myconstants.Mode.MODE_D2V:
    vec_dim = myconstants.D2V_DIM
    model = gensim.models.Doc2Vec.load(d2v_model_path)


  accuracy_ratings, accuracy_reviews = get_validate_data(accuracy_path, model, mode)

  accuracy_normal = {1:0, 2:0, 3:0, 4:0, 5:0}
  accuracy_cascade = {1:0, 2:0, 3:0, 4:0, 5:0}
  for idx, reviews in enumerate(accuracy_reviews):
    temp_classes = classification
    import operator
    rating, value = max(enumerate(accuracy_ratings[idx]), key=operator.itemgetter(1))
    rating += 1

    # while 1:
    #   if len(temp_classes) > 1:
    #     p = predict(reviews, temp_classes, vec_dim, mode)
    #     temp_classes = form_classes(p)
    #   else:
    #     if temp_classes[0][0] == rating:
    #       accuracy_cascade[temp_classes[0][0]] += 1

    #     p = predict(reviews, myconstants.CLASSES, vec_dim, mode)
    #     if p[0] == rating:
    #       accuracy_normal[p[0]] += 1
    #     break

    p = predict(reviews, myconstants.CLASSES, vec_dim, mode)
    # print tolerate_class(p)
    if rating in tolerate_class(p):
      accuracy_cascade[rating] += 1

    p = predict(reviews, myconstants.CLASSES, vec_dim, mode)
    if p[0] == rating:
      accuracy_normal[rating] += 1
    print idx
  print accuracy_normal, accuracy_cascade
  return accuracy_normal, accuracy_cascade
