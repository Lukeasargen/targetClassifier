import csv
import datetime
import os


class Logger():
    def __init__(self, name, headers):
        self.name = name
        self.headers = headers
        self.path = "runs/" + name + datetime.datetime.now().strftime("_%Y%m%d_%H%M%S") + ".csv"
        if not os.path.exists(self.path):
            with open(self.path, 'a+', newline='') as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerow(headers)

    def update(self, row_dict):
        with open(self.path, 'a+', newline='') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow([row_dict[key] for key in self.headers])

