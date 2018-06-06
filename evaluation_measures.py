from dcase_util.data import ProbabilityEncoder
import sed_eval
import numpy
from sklearn.metrics import f1_score


def get_f_measure_by_class(keras_model, nb_tags, generator, steps, thresholds=None):
    """ get f measure for each class given a model and a generator of data (X, y)
    Parameters
    ----------
    keras_model : Model, model to get predictions
    nb_tags : int, number of classes which are represented
    generator : generator, data generator used to get f_measure
    steps : int, number of steps the generator will be used before stopping
    thresholds : int or list, thresholds to apply to each class to binarize probabilities
    Return
    ------
    macro_f_measure : list, f measure for each class
    """

    # Calculate external metrics
    TP = numpy.zeros(nb_tags)
    TN = numpy.zeros(nb_tags)
    FP = numpy.zeros(nb_tags)
    FN = numpy.zeros(nb_tags)
    for counter, (X, y) in enumerate(generator):
        if counter == steps:
            break
        predictions = keras_model.predict(X)

        if len(predictions.shape) == 3:
            # average data to have weak labels
            predictions = numpy.mean(predictions, axis=1)
            y = numpy.mean(y, axis=1)

        if thresholds is None:
            binarization_type = 'global_threshold'
            thresh = 0.2
        else:
            if type(thresholds) is list:
                thresh = thresholds
                binarization_type = "class_threshold"
            else:
                binarization_type = "global_threshold"
                thresh = thresholds
        predictions = ProbabilityEncoder().binarization(predictions,
                                                        binarization_type=binarization_type,
                                                        threshold=thresh,
                                                        time_axis=0
                                                        )

        TP += (predictions + y == 2).sum(axis=0)
        FP += (predictions - y == 1).sum(axis=0)
        FN += (y - predictions == 1).sum(axis=0)
        TN += (predictions + y == 0).sum(axis=0)

    macro_f_measure = numpy.zeros(nb_tags)
    mask_f_score = 2*TP + FP + FN != 0
    macro_f_measure[mask_f_score] = 2*TP[mask_f_score] / (2*TP + FP + FN)[mask_f_score]

    return numpy.mean(macro_f_measure)


def select_threshold(y_pred, y_true):
    best_threshold = []
    grid = numpy.arange(0.0, 0.5, 0.01)
    grid_2 = [0.1, 0.2, 0.3, 0.4, 0.5]
    for i in range(10):
        y_p = y_pred[:, i]
        y_t = y_true[:, i]
        f1_max = 0
        j_max = 0
        for j in grid:
            y_p_c = numpy.zeros(y_p.shape)
            y_p_c[numpy.where(y_p >= j)] = 1
            f1 = f1_score(y_t, y_p_c, 'binary')
            if f1 >= f1_max:
                f1_max = f1
                j_max = j
        best_threshold.append(j_max)
    return best_threshold

