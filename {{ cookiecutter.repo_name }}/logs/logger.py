import logging
from lightgbm.callback import _format_eval_result

def log_best_lgbm(model, metric):
    logging.debug("[{}]\t Best score {}".format(model.best_iteration, model.best_score['valid_0'][metric]))

def log_best_xgbm(model, metric):
    logging.debug("[{}]\t Best score {}".format(model.best_iteration, model.best_score))

def log_evaluation_lgbm(logger, period=1, show_stdv=True, level=logging.DEBUG):
    def _callback(env):
        if period > 0 and env.evaluation_result_list \
                and (env.iteration + 1) % period == 0:
            result = '\t'.join([
                _format_eval_result(x, show_stdv)
                for x in env.evaluation_result_list
            ])
            logger.log(level, '[{}]\t{}'.format(env.iteration + 1, result))
    _callback.order = 10
    return _callback


def log_evaluation_xgbm(logger, period=1, show_stdv=True):
    def _fmt_metric(value, show_stdv=True):
        """format metric string"""
        if len(value) == 2:
            return '%s:%g' % (value[0], value[1])
        elif len(value) == 3:
            if show_stdv:
                return '%s:%g+%g' % (value[0], value[1], value[2])
            else:
                return '%s:%g' % (value[0], value[1])
        else:
            raise ValueError("wrong metric value")

    def callback(env):
        if env.rank != 0 or len(env.evaluation_result_list) == 0 or period is False:
            return
        i = env.iteration
        if i % period == 0 or i + 1 == env.begin_iteration or i + 1 == env.end_iteration:
            msg = '\t'.join([_fmt_metric(x, show_stdv) for x in env.evaluation_result_list])
            logger.info('[%d]\t%s' % (i, msg))

    return callback