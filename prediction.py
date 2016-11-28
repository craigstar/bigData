import ml
import tensorflow as tf
import numpy as np
import myconstants
from myconstants import get_constant, get_save_name, CLASS_LEN, CLASS_LEN_TEST
from ml.tf_training import sentence_mean
import gensim


def predict(sentence, classification, mode=myconstants.Mode.MODE_MEAN):
    print 'your review is : %s' % sentence
    sentence = ml.del_punctuations(sentence)
    sentence = sentence.lower()
    sentence = ml.del_stopwords(sentence)

    while sentence == '':
        sentence = raw_input("Meaningless sentence! Please enter another sentence :")
        print 'your review is : %s' % sentence
        sentence = ml.del_punctuations(sentence)
        sentence = sentence.lower()
        sentence = ml.del_stopwords(sentence)

    print 'your key words for vector are : %s' % sentence

    # load model
    test = False
    accuracy_path = get_constant('ACCURACY_PATH', test)
    w2v_model_path = get_constant('W2V_MODEL_PATH', test)
    d2v_model_path = get_constant('D2V_MODEL_PATH', test)

    # load model
    if mode == myconstants.Mode.MODE_PCA:
        vec_dim = myconstants.Mode.PCA_COMPONENTS * myconstants.W2V_DIM
        model = gensim.models.Word2Vec.load(w2v_model_path)
    elif mode == myconstants.Mode.MODE_MEAN:
        vec_dim = myconstants.W2V_DIM
        model = gensim.models.Word2Vec.load(w2v_model_path)
    elif mode == myconstants.Mode.MODE_D2V:
        vec_dim = myconstants.D2V_DIM
        model = gensim.models.Doc2Vec.load(d2v_model_path)

    vector = np.array(sentence_mean(sentence, model))
    vector.shape = [1, vec_dim]
    sess = tf.Session()
    x = tf.Variable(vector)

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
    # import pdb; pdb.set_trace()
    print 'predicted rating : ', total_class[result]
    print ''
    return total_class[result]


def form_classes(classes):
    formed_classes = []
    for cla in classes:
        formed_classes.append([cla])
    return formed_classes


classification = [[1,2,3],[4,5]]
# classification = [[1],[2],[3],[4],[5]]

mode = myconstants.Mode.MODE_MEAN

if __name__ == '__main__':
    while 1:
        sentence = raw_input("Please enter a sentence (Enter nothing to end prediction):")
        if sentence == '':
            print 'Prediction end.'
            exit()
        else:
            temp_classes = classification
            while 1:
                if len(temp_classes) > 1:
                    p = predict(sentence, temp_classes, mode)
                    temp_classes = form_classes(p)
                else:
                    break
