import time
import uuid
import pandas as pd
from threading import Thread
from inference_on_events import inference_on_events
from utility import send_request
from config import config

class Job():
    def __init__(self, pothole_events):
        super().__init__()
        self.uuid = uuid.uuid4()
        self.arrived_at = time.time()
        self.start_at = None
        self.data = pothole_events
        self.end_at = None

    def start(self):
        print("Start job async")
        self.start_at = time.time()
        events = inference_on_events(self.data)
        obj_events = [event.to_dict() for event in events]
        send_request(config.callback_url, obj_events)
        print("Generated {} events".format(len(obj_events)))
        print("End job async")

    def end(self):
        self.end_at = time.time()

    def stats(self):
        stats = {"uuid": [self.uuid], "time_elapsed": [self.end_at -
                 self.start_at], "waiting_time": [self.start_at-self.arrived_at]}
        df = pd.DataFrame(stats)
        return df


class JobQueue():
    def __init__(self, verbose=False):
        super().__init__()
        self.pending_jobs = []
        self.running_job = None
        self.complete_jobs = []
        self.verbose = verbose
        self.worker = None

    def add(self, pothole_events):
        new_job = Job(pothole_events)
        self.pending_jobs.append(new_job)
        if not self.worker:
            self.worker = Thread(target=self.run_next)
            self.worker.start()
        return new_job

    def run_next(self, delay=0.5):
        if not self.running_job and len(self.pending_jobs) > 0:
            self.running_job = self.pending_jobs.pop(0)
            self.run_job_wrapper()
            time.sleep(delay)
            self.run_next()
        else:
            self.worker = None

    def run_job_wrapper(self):
        self.running_job.start()
        self.running_job.end()
        self.complete_jobs.append(self.running_job)
        if self.verbose:
            print(self.running_job.stats())
            print("Freeing running_job")
        self.running_job = None
