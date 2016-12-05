import json
import re
from nltk.corpus import stopwords

import myconstants
from myconstants import get_constant


def del_punctuations(sentence, exception_list=myconstants.DEFAULT_PUNCTS):
  regex_list = '[^a-zA-Z' + '^\\'.join(exception_list) + ']'

  regex = re.compile(regex_list)
  sentence = regex.sub(' ', sentence)

  return sentence


def num_of_reviews(data_path):
  data_file = open(data_path, 'r')
  total_lines = sum(1 for _ in data_file)
  data_file.close()
  return total_lines


def del_stopwords(sentence, exception_list=myconstants.KEEP_STOPWORDS):
  sentence_list = sentence.split()
  stop = set(stopwords.words('english')) - set(exception_list)
  sentence_set = set(sentence_list) - stop
  new_sentence = [word for word in sentence_list if word in sentence_set]
  return ' '.join(new_sentence)

def scrape(
  stop_at=None,
  train=0.80,
  test=True
):
# This function is to scrape only TestReview and Overall fields from original Amazon database

  HUNDRED = 100

  data_path = get_constant('DATA_PATH', test)
  train_path = get_constant('TRAIN_PATH', test)
  validate_path = get_constant('VALIDATE_PATH', test)
  reviews_path = get_constant('W2V_SRC_PATH', test)

  total_num_reviews = num_of_reviews(data_path)
  unit_percent_reviews = int(total_num_reviews / HUNDRED)

  data_file = open(data_path, 'r')
  train_file = open(train_path, 'w')
  validate_file = open(validate_path, 'w')
  reviews_file = open(reviews_path, 'w')

  progress = 0
  count = 1

  for line in data_file:
    line = line.strip()
    data = json.loads(line)

    review = del_punctuations(data['reviewText'])
    review = review.lower()
    review = del_stopwords(review)

    output = {"overall": data['overall'], "reviewText": review}

    if count <= train * total_num_reviews:
      train_file.write(json.dumps(output)+'\n')
    else:
      validate_file.write(json.dumps(output)+'\n')

    reviews_file.write(review + '\n')

    # print count
    if count % unit_percent_reviews == 0:
      progress = int(count * HUNDRED / float(total_num_reviews))
      print '{}%'.format(progress)
    elif count == total_num_reviews:
      print '100%'

    count += 1

    if stop_at and progress == stop_at:
      break

  data_file.close()
  train_file.close()
  validate_file.close()
  reviews_file.close()

  return
