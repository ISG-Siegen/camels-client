import requests
import json
from camels.client.validate_connection import server_url
from camels.client.data_manager import DataLoader
from camels.client.client_database_identifier import Algorithm, Metric, Task, Learner
from camels.wrapper_master import train_best_model


# stores information about predictions from the server
class ModelManager:

    def __init__(self, loader: DataLoader, metric: Metric, learner: Learner, task: Task):
        self.loader = loader
        self.metric = metric
        self.learner = learner
        self.task = task
        self.predicted_algo_performance = None
        self.best_model = None

    def predict_algo_performance(self) -> None:
        self.loader.preprocess()
        meta_data = self.loader.obtain_metadata()

        print("Predicting with meta learner.")
        try:
            response = requests.get(f"{server_url()}predict_with_meta_learner",
                                    params={
                                        'meta_data': json.dumps(meta_data),
                                        'metric_name': self.metric.name,
                                        'task_name': self.task.name,
                                        'learner_name': self.learner.name
                                    })
        except requests.exceptions.ConnectionError:
            print("Connection to the server could not be established!")
            return
        if response.status_code != 200:
            print(f"Error: The server returned with status code {response.status_code}.")
            return
        response_pred = json.loads(response.text)

        predicted_algo_performance = {}
        for algo_id, score in enumerate(response_pred[0]):
            predicted_algo_performance[Algorithm(algo_id+1)] = score

        self.predicted_algo_performance = predicted_algo_performance

        print(f"Predicted best algorithm with {self.learner} meta learner.\n"
              f"Best algorithm for metric {self.metric} on task {self.task} predicted to be "
              f"{min(self.predicted_algo_performance, key=self.predicted_algo_performance.get)}.")

    def return_best_model(self):
        if self.predicted_algo_performance is None:
            print("Exception: Need predictions before returning a model.")
            raise Exception

        # get best algorithm
        best_algo = min(self.predicted_algo_performance, key=self.predicted_algo_performance.get)
        self.best_model = train_best_model(self.loader.data, best_algo)
