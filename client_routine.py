import pandas as pd
import requests
import json
from typing import List
from camels.client.validate_connection import server_url
from camels.client.data_manager import DataLoader
from camels.client.evaluation_manager import Evaluator
from camels.client.model_manager import ModelManager
from camels.client.client_database_identifier import Algorithm, Metric, Task, Learner

# turn off warnings to avoid cluttering console
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


# calls server function to initially populate database
def c_populate_database() -> None:
    print("Populating database on server.")
    try:
        response = requests.post(f"{server_url()}populate_database")
    except requests.exceptions.ConnectionError:
        print("Connection to the server could not be established!")
        return
    if response.status_code != 200:
        print(f"Error: The server returned with status code {response.status_code}.")
        return
    print(f"Server returned: {json.loads(response.text)}")

    return


# starts the evaluation pipeline
def c_evaluate_algorithms(algos: List[Algorithm], metrics: List[Metric], task: Task, data: pd.DataFrame) -> None:
    """
    @param algos: list of algorithms that should be evaluated
    @param metrics: list of metrics the algorithms should be evaluated for
    @param task: the prediction task
    @param data: a dataframe with the data to evaluate on
    """

    loader = DataLoader(data, True, True, task)
    evaluator = Evaluator(loader, algos, metrics)
    evaluator.evaluate()

    return


# calls for the server to train the meta learner
def c_train_meta_learner(metrics: List[Metric], tasks: List[Task], learners: List[Learner]) -> None:
    """
    @param metrics: list of metrics the meta-learner(s) should be trained for
    @param tasks: list of tasks the meta-learner(s) should be trained for
    @param learners: list of meta-learner algorithms to train
    """

    metric_names = [metric.name for metric in metrics]
    task_names = [task.name for task in tasks]
    learner_names = [learner.name for learner in learners]

    print("Training meta learner.")
    try:
        response = requests.post(f"{server_url()}train_meta_learner", data={'metric_names': json.dumps(metric_names),
                                                                            'task_names': json.dumps(task_names),
                                                                            'learner_names': json.dumps(learner_names)})
    except requests.exceptions.ConnectionError:
        print("Connection to the server could not be established!")
        return
    if response.status_code != 200:
        print(f"Error: The server returned with status code {response.status_code}.")
        return
    response_list = json.loads(response.text)
    for eval_cv in response_list:
        print(f"Server returned:\n{pd.read_json(eval_cv)}")

    return


# calls the server to return a prediction for the selected data
def c_predict_with_meta_learner(metric: Metric, task: Task, learner: Learner, data: pd.DataFrame) -> None:
    """
    @param metric: the metric for which the meta-learner should predict
    @param learner: the type of meta-learner to use
    @param task: the prediction task to solve for
    @param data: a dataframe with data to predict for
    """

    loader = DataLoader(data, True, False, task)
    manager = ModelManager(loader, metric, learner, task)
    manager.predict_algo_performance()
    manager.return_best_model()

    return
