import logging
from collections import OrderedDict

from aif360.metrics import ClassificationMetric

import modeldevelopment.settings as settings

logger = logging.getLogger(__name__)


def compute_metrics(dataset_true, dataset_predictions, display=True):
    """ Compute the key metrics """
    try:
        classified_metric = ClassificationMetric(
            dataset_true,
            dataset_predictions,
            unprivileged_groups=settings.UNPRIVILEGED_GROUPS,
            privileged_groups=settings.PRIVILEGED_GROUPS,
        )
        metrics = OrderedDict()

        metrics["Balanced accuracy"] = 0.5 * (
                classified_metric.true_positive_rate() + classified_metric.true_negative_rate())
        metrics["Statistical parity difference"] = classified_metric.statistical_parity_difference()
        metrics["Disparate impact"] = classified_metric.disparate_impact()
        metrics["error_rate_difference"] = classified_metric.error_rate_difference()
        metrics["error_rate_ratio"] = classified_metric.error_rate_ratio()
        metrics["average_odds_difference"] = classified_metric.average_odds_difference()
        metrics["equal_opportunity_difference"] = classified_metric.equal_opportunity_difference()

        if display:
            for k in metrics:
                print("%s  = %.4f" % (k, metrics[k]))
            print('\n')
        return metrics
    except Exception as e:
        logger.error(e)
