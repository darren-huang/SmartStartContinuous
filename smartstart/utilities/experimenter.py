"""Module for making it easy to run experiments

This methods defined in this module make it easy to run multiple experiments
in parallel and using different sets of parameters.
"""
from multiprocessing import Pool, cpu_count

from sklearn.model_selection import ParameterGrid
from tqdm import *


def run_experiment(param_grid, n_processes=-1):
    """Method for running experiments

    This method is used for making it easy to run experiments in parallel and
    use different combinations of parameters. The param_grid dictionary
    describes what task or tasks must be executed and what parameters and/or
    parameter combinations must be used. Each experiment receives its own
    unique id.

    The param_grid consists of the following entries:
        * task : function that has to be executed
        * num_exp : number of time each experiment has to be executed (optional)
        * params : lists holding the params that have to be used

    Example of a param_grid:

        param_grid = {'task': task_func, 'num_exp': 5, 'alpha': [0.05, 0.1]}

    and the task func should then be in the following format:

        def task_func(params):
            # Define the code that has to be executed params defined can be
            # accessed like this
            alpha = params['alpha']

            # a run parameter, useful when num_exp > 1
            run = param['run']

            # and the unique id
            id = params['id']


    In total the task function will be executed 10 times. 5 times with each
    alpha parameter. See tutorial or examples for a full example.

    Parameters
    ----------
    param_grid : :obj:`dict`
        dictionary containing the parameters to be used during the experiments
    n_processes : :obj:`int`
        number of processes to run in parallel (Default value = -1,
        all the cpu_cores in the system are used)

    Raises
    ------
    Exception
        Please define a task function in the param_grid to execute.
    """
    if type(param_grid) is not list:
        param_grid = [param_grid]

    for params in param_grid:
        if 'task' not in params:
            raise Exception('Please define a task function in the param_grid '
                            'to execute.')

        if 'num_exp' not in params:
            params['run'] = [0]
        else:
            params['run'] = range(params['num_exp'])
            del params['num_exp']

        if type(params['task']) is not list:
            params['task'] = [params['task']]


    params = list(ParameterGrid(param_grid))
    for i in range(len(params)):
        params[i]['id'] = i

    print("\033[1mNumber of processes: %d\033[0m" % len(params))

    if n_processes == -1:
        n_processes = cpu_count()
    if n_processes > 1:
        with Pool(n_processes) as p:
            r = list(tqdm(p.imap_unordered(process_task, params), total=len(params)))
    else:
        for single_param in params:
            process_task(single_param)


def process_task(params):
    """Helper method for executing the task defined in params

    Parameters
    ----------
    params : :obj:`dict`
        dictionary with the parameters to be used in the experiment
    """
    params['task'](params)
