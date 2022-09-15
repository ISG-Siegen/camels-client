from typing import List
import numpy as np
import requests
import json
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.utils import shuffle
from camels.client.validate_connection import server_url
from camels.client.data_manager import DataLoader
from camels.client.client_database_identifier import Algorithm, Metric
from camels.wrapper_master import fit_and_predict

pd.options.mode.chained_assignment = None


# stores data and evaluation routines for client evaluation procedures
class Evaluator:

    def __init__(self, loader: DataLoader, algos: List[Algorithm], metrics: List[Metric]):
        self.loader = loader
        self.algos = algos
        self.metrics = metrics

    # evaluates a list of algorithms on a list of metrics for a single data set
    def evaluate(self) -> None:
        print("Preprocessing data.")
        self.loader.preprocess()

        existing_runs = None
        # checks if runs exist for the data on the given algorithms and metrics
        if self.loader.upload_eval:
            print("Checking metadata run status.")

            algo_names = [algo.name for algo in self.algos]
            metric_names = [metric.name for metric in self.metrics]

            try:
                response = requests.get(f"{server_url()}check_data_status",
                                        params={'l_hash': self.loader.local_hash,
                                                'algo_names': json.dumps(algo_names),
                                                'task_name': self.loader.data_set_type.name,
                                                'metric_names': json.dumps(metric_names)})
            except requests.exceptions.ConnectionError:
                print("Connection to the server could not be established!")
                return
            if response.status_code != 200:
                print(f"Error: The server returned with status code {response.status_code}.")
                return

            existing_runs = np.array(json.loads(response.text))
            for run in existing_runs:
                print(f"Server returned: Run for {Algorithm[run[0]].name} on {Metric[run[1]].name} exists.")

        print("Evaluating algorithms.")
        for algo in self.algos:
            if existing_runs is not None and len(existing_runs) > 0:
                algo_existing_runs = existing_runs[existing_runs[:, 0] == algo.name]
                algo_metric_combos = np.array([[algo.name, m.name] for m in self.metrics])
                all_combos_exist = np.isin(algo_metric_combos, algo_existing_runs).all()
                if all_combos_exist:
                    print(f"All algorithm and metric combos for {algo.name} already exist. "
                          f"Aborting evaluation for this algorithm.")
                    continue

            print(f"Started evaluation for {algo.name}.")
            scores = {metric.name: [] for metric in self.metrics}

            # split data with user based partitioning to mitigate cold-start
            users = self.loader.data["user"].unique()
            indices = shuffle(np.arange(len(users)))
            user_groups = self.loader.data[self.loader.data.user.isin(users[indices])].groupby('user')
            d_test = user_groups.apply(lambda x: x.sample(frac=0.2)).reset_index(0, drop=True)
            train_test_idx = pd.Series(True, index=self.loader.data.index)
            train_test_idx[d_test.index] = False
            d_train = self.loader.data[train_test_idx]

            # avoid zeroes in ratings due to some algorithms failing
            d_train.loc[d_train["rating"] == 0, "rating"] = 0.000001

            # perform fit and predict
            prediction, fit_duration, predict_duration = fit_and_predict(d_train, d_test, algo)

            # build score dictionary
            for metric in self.metrics:
                if metric == Metric.FitTime:
                    score = fit_duration
                elif metric == Metric.PredictTime:
                    score = predict_duration
                elif metric == Metric.RootMeanSquaredError:
                    score = mean_squared_error(prediction, d_test["rating"], squared=False)
                elif metric == Metric.MeanAbsoluteError:
                    score = mean_absolute_error(prediction, d_test["rating"])
                else:
                    print(f"Exception: Unsupported metric {metric.name}.")
                    raise Exception

                print(f"{metric.name} score: {score}.")
                scores[metric.name].append(score)

            final_scores = {k: np.array(v).mean() for k, v in scores.items()}

            # organize evaluations to database structure
            algo_evaluations = []
            for metric, score in final_scores.items():
                # only upload if override is true or the run does not exist yet
                if not any([([algo.name, Metric[metric].name] == entry).all() for entry in existing_runs]):
                    algo_evaluations.append({
                        "Hash": self.loader.local_hash,
                        "Algorithm": [algo.name],
                        "Task": [self.loader.data_set_type.name],
                        "Metric": [Metric[metric].name],
                        "Score": [float(score)]
                    })

            if algo_evaluations is not None and len(algo_evaluations) > 0 and self.loader.upload_eval:
                print("Uploading runs.")
                try:
                    response = requests.post(f"{server_url()}save_runs",
                                             data={'evaluations': json.dumps(algo_evaluations)})
                except requests.exceptions.ConnectionError:
                    print("Connection to the server could not be established!")
                    return
                if response.status_code != 200:
                    print(f"Error: The server returned with status code {response.status_code}.")
                    return
                print(f"Server returned: {json.loads(response.text)}")
            else:
                print("Skipping upload.")

        print("Evaluation finished.")
        return
