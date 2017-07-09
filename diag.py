import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from sklearn.metrics import fbeta_score
from keras.models import model_from_json
import numpy as np  # linear algebra
import pandas as pd
from tqdm import tqdm
import preprocessor
from model import *
from config import *

from random import choice


class Prediction:

    def __init__(self, file_name, prediction, ground_truth):
        self.file_name = file_name
        self.model_output = prediction
        self.ground_truth = ground_truth

    def rounded_prediction(self):
        return [str(round(x, 2)) for x in self.model_output]

    def __repr__(self):
        tuple = (self.file_name, ' '.join(list(self.rounded_prediction())), self.ground_truth)
        return "<Prediction file_name:%s\n predictions:%s\n ground_truth:%s\n>" % tuple

    def __str__(self):
        return "%s, %s, %s" % (self.file_name, self.model_output, self.ground_truth)

    def label(self, thresholds):
        output = [inv_mapping[i] for i, value in enumerate(self.model_output) if value > thresholds[i]]
        return output

    def post_threshold(self, thresholds):
        output = [value > thresholds[i] for i, value in enumerate(self.model_output)]
        return output


def f2_score(y_true, y_pred):
        # fbeta_score throws a confusing error if inputs are not numpy arrays
        y_true, y_pred, = np.array(y_true), np.array(y_pred)
        # We need to use average='samples' here, any other average method will generate bogus results
        return fbeta_score(y_true, y_pred, beta=2, average='samples')

def evaluate_thresholds(pred, thresh):

    ground_truth = [x.ground_truth for x in pred]

    predict_data = []
    for prediction in pred:
        predict_data.append(prediction.post_threshold(thresh))

    score = f2_score(ground_truth, predict_data)
    return score


def run():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    model_name = 'spectral128'
    model_folder = models_path + '/' + model_name
    model_weights_path = model_folder + '/model.h5'
    model_predictions_path = model_folder + '/predictions.pickle'

    predict = False

    file_names = current_model.split.validation_file_names()

    #current_model.thresholds = [0.20] * 17
    #current_model.save_thresholds(models_path + '/threshold.pickle')

    predictions = []
    if predict:

        x_data = current_model.split.validation_images()
        y_data = current_model.split.validation_some_hot()

        current_model.load_weights(model_weights_path)


        model_output = current_model.model.predict(x_data)
        predictions = [Prediction(file_name, prediction, ground_truth) for file_name, prediction, ground_truth in zip(file_names, model_output, y_data)]

        with open(model_predictions_path, 'wb') as handle:
            pickle.dump(predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        with open(model_predictions_path, 'rb') as handle:
            predictions = pickle.load(handle)



    thresholds = current_model.load_thresholds(models_path + '/threshold.pickle')

    predictions_labels = []
    for prediction in predictions:
        thresholded = prediction.label(thresholds)
        predictions_labels.append(thresholded)

    #print(predictions[:10], sep='\n')
    #print(predictions_labels[:10])

    print(thresholds)
    original_score = evaluate_thresholds(predictions, thresholds)
    print('Original Score: {}'.format(original_score))
    for _ in range(10):
        for n in range(17):
            delta = choice([-0.1, -0.05, -0.01, -0.002, 0.002, 0.005, 0.01, 0.05, 0.1])
            test_thresholds = thresholds.copy()
            test_thresholds[n] += delta
            score = evaluate_thresholds(predictions, test_thresholds)
            if score > original_score:
                thresholds = test_thresholds
                original_score = score
                print(score)

    print('New Score: {}'.format(evaluate_thresholds(predictions, thresholds)))
    print(thresholds)
    current_model.thresholds = thresholds
    current_model.save_thresholds(models_path + '/threshold.pickle')

    ground_truth = [x.ground_truth for x in predictions]
    predict_data = []
    for prediction in predictions:
        thresholded = prediction.post_threshold(thresholds)
        predict_data.append(thresholded)


    total_num = ground_truth.__len__()
    print("Total number of items: {}".format(total_num))

    for index in range(17):

        total_features = sum([x[index] == 1 for x in ground_truth])
        #print("Total: ", total_features)

        true_positives = sum([x[index] == 1 and y[index] == 1 for x, y in zip(ground_truth, predict_data)])
        #print("True Positives: ", true_positives)

        false_negatives = sum([x[index] == 1 and y[index] == 0 for x, y in zip(ground_truth, predict_data)])
        #print("False Negatives: ", false_negatives)

        false_positives = sum([x[index] == 0 and y[index] == 1 for x, y in zip(ground_truth, predict_data)])
        #print("False Positives: ", false_positives)

        accuracy = true_positives / (true_positives + false_positives)
        #print("Accuracy ", accuracy)

        recall = true_positives / (true_positives + false_negatives)
        impact = total_features - (total_features * (accuracy + 4*recall)/5)
        print(" Recall {0:.2f} Impact {1:4.0f}".format(recall, impact), inv_mapping[index])


    tags_list = [None] * len(predictions_labels)
    for i, tags in enumerate(predictions_labels):
        tags_list[i] = ' '.join(map(str, tags))

    final_data = [[filename.split(".")[0], tags] for filename, tags in zip(file_names, tags_list)]
    final_df = pd.DataFrame(final_data, columns=['image_name', 'tags'])
    final_df.head()

    final_df.to_csv(diag_path + '/{}.csv'.format(model_name), index=False)