import json
import os
from termcolor import colored
from datetime import datetime
from typing import Optional, Union


class EvaluationOutputCollector:
    """
    A class to collect and display output from a simulation of patient and medical doctor interaction and actions.
    """

    def __init__(self, dataset_name, hadm_id, save_dir):
        # directory = os.path.join(save_dir, dataset_name)
        # os.makedirs(directory, exist_ok=True)

        base_filename = f"{dataset_name}_{hadm_id}_conversation"
        extension = ".json"
        counter = 1
        file_path = os.path.join(save_dir, f"{base_filename}_{counter}{extension}")

        while os.path.exists(file_path):
            counter += 1
            file_path = os.path.join(save_dir, f"{base_filename}_{counter}{extension}")

        # Initialize empty JSON
        with open(file_path, "w") as file:
            json.dump([], file)

        self.filename = file_path
        self.items = []  # In-memory log

    def _save_event(self, event: dict):
        """Append an event to the JSON file."""
        with open(self.filename, "r+") as file:
            data = json.load(file)
            data.append(event)
            file.seek(0)
            json.dump(data, file, ensure_ascii=False, indent=4)
            file.truncate()

    def display_message(self, name: str, message: str):
        """
        Display and store a conversational message from the assistant or user.

        Parameters:
        - name (str): Who sent the message (e.g., "Doctor", "Patient").
        - message (str): The raw message content (Markdown).
        """

        entry = {
            "type": "message",
            "role": name,
            "content": message,
            "timestamp": datetime.now().isoformat(),
        }

        self.items.append(entry)
        self._save_event(entry)

        # Console preview
        color = "green" if name == "Doctor" else "blue"
        print(colored(f"{name}: {message}", color))

    def display_action(self, message: str, function_name: str, arguments: Optional[Union[str, list, dict]] = None):
        """
        Display and store a system action (e.g., function call result).

        Parameters:
        - message (str): Description or result from the tool.
        - function_name (str): The name of the function/tool used.
        """
        if isinstance(message, list):
            message = "\n".join(str(m) for m in message)

        entry = {
            "type": "action",
            "role": "System",
            "function": function_name,
            "content": message,
            "arguments": arguments,
            "timestamp": datetime.now().isoformat(),
        }

        self.items.append(entry)
        self._save_event(entry)

        print(colored(f"[{function_name.upper()}] {message}", "magenta"))