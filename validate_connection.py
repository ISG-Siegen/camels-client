from pathlib import Path
import requests
import json
import os

file_path = Path(os.path.dirname(os.path.abspath(__file__)))


def server_url() -> str:
    with open(file_path / 'connection_settings.json') as file:
        config = json.load(file)
    return config["server-ip"]


def sync_database_identifier() -> None:
    print("Downloading server database identifier.")
    try:
        response = requests.get(f"{server_url()}validate_connection")
    except requests.exceptions.ConnectionError:
        print("Connection to the server could not be established!")
        return
    if response.status_code != 200:
        print(f"Error: The server returned with status code {response.status_code}.")
        return
    print("Downloading complete.")
    server_identifier = response.text

    if os.path.exists(file_path / 'client_database_identifier.py'):
        print("Checking if server and client database identifier match.")
        with open(file_path / 'client_database_identifier.py', "rb") as file:
            client_identifier = file.read().decode()
        if client_identifier == server_identifier:
            print("Server and client database identifier match.")
            return
        else:
            print("Overwriting client database identifier with downloaded database identifier.")
            with open(file_path / 'client_database_identifier.py', "wb") as file:
                file.write(server_identifier.encode())
            return
    else:
        print("Saving downloaded database identifier.")
        with open(file_path / 'client_database_identifier.py', "wb") as file:
            file.write(server_identifier.encode())
        return


sync_database_identifier()
