from dcase_util.data import ProbabilityEncoder
import sed_eval
import numpy


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
            thresh = 0.5
        else:
            binarization_type = "class_threshold"
            assert type(thresholds) is list
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

    return macro_f_measure