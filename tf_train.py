import ml
import myconstants
from myconstants import get_constant

mode = myconstants.Mode.MODE_MEAN
# ml.test_similarity(stop_at=1000, max_count=3, test=True, mode=mode)

learning_rate = 0.1
# ml.tf_train(learning_rate=learning_rate, test=False, mode=mode, loops=3000)


classes = [[1,2],[3,4,5]]
# ml.save_prediction(classification=classes)

ml.test_cascade(classification=classes)
