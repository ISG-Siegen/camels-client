import pandas as pd
import requests
import json
from collections import Counter
from camels.client.validate_connection import server_url
from camels.client.client_database_identifier import Task
from camels.wrapper_master import calculate_meta_features


# controls preprocessing routines and metadata acquisition
class DataLoader:

    def __init__(self, data: pd.DataFrame = None, prune: bool = True,
                 upload_eval: bool = True, data_set_type: Task = None):
        self.data = data
        self.prune = prune
        self.upload_eval = upload_eval
        self.data_set_type = data_set_type
        self.local_hash = None

    def obtain_metadata(self) -> dict:
        if self.data is None:
            print("Exception: Preprocess data before getting metadata.")
            raise Exception

        return calculate_meta_features(self.data)

    def preprocess(self) -> None:
        # check if data has exactly three cols
        data_cols = list(self.data)
        if len(data_cols) != 3:
            print("Data needs to have exactly three columns.")
            return

        # check if data has user, item and rating cols
        required_cols = ['user', 'item', 'rating']
        if not all(col in data_cols for col in required_cols):
            print("Data needs to have the columns: user, item, rating.")
            return

        # prune data
        if self.prune:
            self.data = self.data[~self.data.drop(columns=["rating"]).duplicated(keep="last")]

            # such that every user and item have at least five ratings and at most 1000
            u_cnt = Counter(self.data["user"])
            sig_users = [k for k in u_cnt if ((u_cnt[k] >= 5) and (u_cnt[k] <= 1000))]
            # i_cnt = Counter(self.data["item"])
            # sig_items = [k for k in i_cnt if ((i_cnt[k] >= 5) and (i_cnt[k] <= 1000))]
            self.data = self.data[self.data["user"].isin(sig_users)]
            # self.data = self.data[self.data["item"].isin(sig_items)]

        # get metadata only if evaluation should be uploaded to server
        if self.upload_eval:
            # get the dataframe hash to compare against the database
            self.local_hash = int(pd.util.hash_pandas_object(self.data).sum())

            # send hash to server to check if the data set already exists
            print("Checking if hash exists on server.")
            try:
                response = requests.get(f"{server_url()}check_hash", params={'l_hash': self.local_hash})
            except requests.exceptions.ConnectionError:
                print("Connection to the server could not be established!")
                return
            if response.status_code != 200:
                print(f"Error: The server returned with status code {response.status_code}.")
                return
            response = json.loads(response.text)
            print(f"Server returned: {response[0]}")

            # only calculate metadata, if the data set has not yet been uploaded
            if response[1]:
                # new uploads need a data set name
                # get meta data
                meta_data = self.obtain_metadata()
                # append with administrative content
                meta_data["Hash"] = [self.local_hash]

                # send meta data to server
                print("Saving metadata on server.")
                try:
                    response = requests.post(f"{server_url()}save_metadata", data={'meta_data': json.dumps(meta_data)})
                except requests.exceptions.ConnectionError:
                    print("Connection to the server could not be established!")
                    return
                if response.status_code != 200:
                    print(f"Error: The server returned with status code {response.status_code}.")
                    return
                print(f"Server returned: {json.loads(response.text)}")

                return
            # otherwise calculate metadata of data set
            else:
                return
