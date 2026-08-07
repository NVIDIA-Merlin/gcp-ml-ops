# Copyright (c) 2021 NVIDIA Corporation. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import logging
from time import monotonic, sleep
from queue import Queue
from threading import Lock, Thread

import argparse
from google.cloud import pubsub_v1
import collections

from sklearn import metrics
import pandas as pd
import numpy as np

import kfp
import datetime

from message_validation import (
    PayloadValidationError,
    validate_payload,
    validate_publish_time,
)

# logging.basicConfig(level=logging.INFO,
#                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# client = kfp.Client(host='https://320d47d67af4e8cf-dot-us-central1.pipelines.googleusercontent.com')

def get_pipeline_id(name, client):
    pl_id = None
    page_size = 100
    page_token = ''
    while True:
        res = client.list_pipelines(page_size=page_size, page_token=page_token)
        pl_list = res.pipelines
        for pl in pl_list:
            if pl.name == name:
                pl_id = pl.id
                return pl_id
        page_token = res.next_page_token
        if not page_token:
            break
    return pl_id

def get_pipeline_info(input_name, client_host):
    page_size = 200
    page_token = ''
    pipeline_runs = []

    client = kfp.Client(host=client_host)

    res = client.list_runs(page_size=page_size, page_token=page_token)
    for runs in res.runs:
        if runs.resource_references[1].name == input_name:
            pipeline_runs.append(runs)

    if len(pipeline_runs) !=0:
        for prun in pipeline_runs:
            if prun.status == 'Running':
                return None

    # if prun.status == 'Succeeded':
        tmp = { 'pipelineID': prun.resource_references[1].key.id,
            'experimentID': prun.resource_references[0].key.id,
            'status': prun.status,
            'new_run_name': 'triggered_'+str(datetime.datetime.now())}
            
        return tmp
            # pid = get_pipeline_id(input_name,client)
            # print("pid: ", name)

    return None

def trigger_kfp(pipeline_name, client_host):
    logging.warning("Triggering Kubeflow Pipeline...")

    # If pipeline is already running --> False
    # Else -> True
    try:
        pipeline_info = get_pipeline_info(pipeline_name, client_host)
    except Exception as e:
        logging.error(f"Triggering pipeline error: {e}")
        return False

    logging.info(f"Pipeline info: {pipeline_info}")

    if pipeline_info != None:
            print("Using pipeline ID: ", pipeline_info['pipelineID'], " triggering ", pipeline_info['new_run_name'], " at: ", datetime.datetime.now())
            client = kfp.Client(host=client_host)
            res = client.run_pipeline(pipeline_info['experimentID'], pipeline_info['new_run_name'], pipeline_id=pipeline_info['pipelineID'])
            return True
    else:
        logging.info("Did not trigger the pipeline")
        return False


class AccMonitor:
    def __init__(
        self,
        project_id,
        expected_model_name,
        subscription_id,
        topic_id,
        timeout,
        evaluate_period=500,
        acc_threshold=0.5,
        min_trigger_len=0.5,
        pipeline_name="merlin-pipeline",
        min_log_length=320,
        log_time_delta=60,
        pv_location="/var/lib/data/",
        client_host=None,
        max_event_age_seconds=300,
        max_future_skew_seconds=60,
        trigger_cooldown_seconds=3600,
        max_seen_event_ids=10000,
    ):
        self.evaluate_period = evaluate_period
        self.pipeline_name = pipeline_name
        self.pv_location = pv_location
        self.client_host = client_host
        self.request_queue = Queue(maxsize=self.evaluate_period)

        self.project_id = project_id
        self.expected_model_name = expected_model_name
        self.subscription_id = subscription_id
        self.topic_id = topic_id
        self.expected_topic_path = f"projects/{project_id}/topics/{topic_id}"
        self.timeout = timeout
        self.acc_threshold = acc_threshold
        self.max_event_age_seconds = max_event_age_seconds
        self.max_future_skew_seconds = max_future_skew_seconds
        self.trigger_cooldown_seconds = trigger_cooldown_seconds
        self.last_trigger_attempt_time = None

        self.max_seen_event_ids = max_seen_event_ids
        self.seen_event_ids = collections.OrderedDict()
        self.seen_event_ids_lock = Lock()

        self.min_trigger_len = min_trigger_len * self.evaluate_period
        self.min_log_length = min_log_length
        self.log_time_delta = datetime.timedelta(seconds=log_time_delta)

        self.label_queue = collections.deque(maxlen=self.evaluate_period)
        self.pred_queue = collections.deque(maxlen=self.evaluate_period)

    def _verify_subscription_topic(self):
        subscriber_client = pubsub_v1.SubscriberClient()
        subscription_path = subscriber_client.subscription_path(
            self.project_id, self.subscription_id
        )
        try:
            subscription = subscriber_client.get_subscription(
                request={"subscription": subscription_path}
            )
        finally:
            subscriber_client.close()

        if subscription.topic != self.expected_topic_path:
            raise RuntimeError(
                "configured subscription is not attached to the expected topic"
            )
        return subscription_path

    def _remember_event(self, event_id):
        with self.seen_event_ids_lock:
            if event_id in self.seen_event_ids:
                self.seen_event_ids.move_to_end(event_id)
                return False
            self.seen_event_ids[event_id] = None
            if len(self.seen_event_ids) > self.max_seen_event_ids:
                self.seen_event_ids.popitem(last=False)
            return True

    def _forget_event(self, event_id):
        with self.seen_event_ids_lock:
            self.seen_event_ids.pop(event_id, None)

    def run(self):
        subscription_path = self._verify_subscription_topic()

        def enqueue_request(self):
            """
            Receives messages from a Pub/Sub subscription and adds the request to a queue.

            The idea is to decouple message processing from message reception so that
            if there are a large number of messages at once, processing does not cause delays in the
            thread recieving messages.
            """

            # Initialize a Subscriber client
            subscriber_client = pubsub_v1.SubscriberClient()

            def callback(message):
                now = datetime.datetime.now(datetime.timezone.utc)
                try:
                    validate_publish_time(
                        message.publish_time,
                        now=now,
                        max_age_seconds=self.max_event_age_seconds,
                        max_future_skew_seconds=self.max_future_skew_seconds,
                    )
                    payload = validate_payload(
                        message.data,
                        now=now,
                        max_event_age_seconds=self.max_event_age_seconds,
                        max_future_skew_seconds=self.max_future_skew_seconds,
                        expected_model_name=self.expected_model_name,
                    )
                except PayloadValidationError as error:
                    logging.warning(
                        "Rejecting Pub/Sub message %s: %s",
                        message.message_id,
                        error,
                    )
                    message.ack()
                    return

                event_id = payload["event_id"]
                if not self._remember_event(event_id):
                    logging.info("Ignoring duplicate event %s", event_id)
                    message.ack()
                    return

                try:
                    self.request_queue.put(payload)
                except Exception:
                    self._forget_event(event_id)
                    message.nack()
                    logging.exception(
                        "Could not queue Pub/Sub message %s", message.message_id
                    )
                    return

                message.ack()

            flow_control = pubsub_v1.types.FlowControl(
                max_messages=self.evaluate_period
            )
            streaming_pull_future = subscriber_client.subscribe(
                subscription_path, callback=callback, flow_control=flow_control
            )
            logging.info(f"Listening for messages on {subscription_path}..\n")

            try:
                # Calling result() on StreamingPullFuture keeps the main thread from
                # exiting while messages get processed in the callbacks.
                streaming_pull_future.result(timeout=self.timeout)
            except Exception:
                logging.exception(
                    "Pub/Sub receive loop stopped for %s", subscription_path
                )
                streaming_pull_future.cancel()

            subscriber_client.close()

        # Start the enqueue thread as a daemon
        enqueue = Thread(target=enqueue_request, args=(self,))
        enqueue.daemon = True
        enqueue.start()

        """
        Fetches request from a queue, and calculates the rolling accuracy over last N requests
        If the rolling accuracy is below a pre-specified threshold, raises an alarm

        - We have access to the features here. We save the requests into a .parquet file
        in batches

        - PubSub usually does not guarantee in-order delivery of messages
        """

        # Initialization
        rolling_acc = 1.0

        CATEGORICAL_COLUMNS =  ['C' + str(x) for x in range(1,27)]
        CONTINUOUS_COLUMNS = ['I' + str(x) for x in range(1,14)]
        LABEL_COLUMNS = ['label']
        col_names =  LABEL_COLUMNS + CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS
        DATETIME_FORMAT = '%d_%m_%Y-%H-%M-%S'
        last_log_time = datetime.datetime.strptime('01_01_1970-00-00-00', DATETIME_FORMAT)

        # Create an empty dataframe
        df_temp = pd.DataFrame(columns = col_names)

        while True:
            while self.request_queue.empty():
                # sleep so .get doesnt eat CPU cycles if queue is empty
                sleep(0.1)

            # Fetch the payload
            payload = self.request_queue.get()

            request = np.concatenate((np.array([payload["trueval"]], float),
                                      np.array(payload["input0"]),
                                      np.array(payload["input1"])))

            # Append new request to the dataframe
            df_temp = pd.concat(
                [df_temp, pd.DataFrame([request], columns=col_names)], ignore_index=True
            )

            # Write to a file if there are a minimum number of samples available,
            # and if a minimum amount of time has passed since last write
            # TOFIX: This is problematic if no new request comes for a while and
            # there are many requests in the dataframe ready to be written already
            current_time = datetime.datetime.now()
            if (df_temp.shape[0] >= self.min_log_length) and \
                                (current_time - last_log_time >= self.log_time_delta):
                filename = current_time.strftime(DATETIME_FORMAT) + ".parquet"
                logging.info(f"Writing {df_temp.shape[0]} records to {self.pv_location} / {filename}...")
                # print(f"Writing {df_temp.shape[0]} records to {self.pv_location+filename}...")
                df_temp.reset_index(inplace=True, drop=True)
                df_temp.to_parquet(self.pv_location+"/"+filename)

                # Clear the dataframe
                df_temp = pd.DataFrame(columns = col_names)
                last_log_time = current_time

            # Circular buffer of size evaluate_period
            self.label_queue.append(payload["trueval"])
            self.pred_queue.append(payload["predval"])

            try:
                # This will fail if there is only one class in label_queue, catch and pass
                # in that case
                rolling_acc = metrics.roc_auc_score(self.label_queue, self.pred_queue)
                logging.info(f"Rolling AUC score: {rolling_acc}")
            except ValueError:
                pass


            if (rolling_acc < self.acc_threshold) and (
                len(self.label_queue) > self.min_trigger_len
            ):
                current_monotonic_time = monotonic()
                if (
                    self.last_trigger_attempt_time is not None
                    and current_monotonic_time - self.last_trigger_attempt_time
                    < self.trigger_cooldown_seconds
                ):
                    remaining_seconds = self.trigger_cooldown_seconds - (
                        current_monotonic_time - self.last_trigger_attempt_time
                    )
                    logging.info(
                        "Retraining cooldown active for another %.0f seconds",
                        remaining_seconds,
                    )
                    continue

                self.last_trigger_attempt_time = current_monotonic_time
                success = trigger_kfp(self.pipeline_name, self.client_host)
                if success:
                    self.label_queue.clear()
                    self.pred_queue.clear()
                    rolling_acc = 1.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    print("In Performance monitoring module")

    parser.add_argument("--project_id",
                        type=str,
                        required=True,
                        default="dl-tme",
                        help="Google Cloud project ID")

    parser.add_argument("--subscription_id",
                        type=str,
                        required=True,
                        default="sub_id",
                        help="Pub/Sub subscription ID")
    parser.add_argument("--topic_id",
                        type=str,
                        required=True,
                        help="Expected Pub/Sub topic ID")

    parser.add_argument("--expected_model_name",
                        type=str,
                        required=True,
                        help="Only accept events for this model")

    parser.add_argument("--timeout",
                        type=int,
                        required=False,
                        default=None,
                        help="Timeout for Streaming Pull")

    parser.add_argument("--evaluate_period",
                        type=int,
                        required=False,
                        default=500,
                        help="Evaluate over the last evaluate_period samples")

    parser.add_argument("--min_trigger_len",
                        type=float,
                        required=False,
                        default=0.5,
                        help="Minimum number of samples in queue before monitoring based trigger. \
                               As a percentage of evaluate_period ")

    parser.add_argument("--acc_threshold",
                        type=float,
                        required=False,
                        default=0.5,
                        help="AUC ROC threshold for trigger. Default 0.8")

    parser.add_argument("--pipeline_name",
                        type=str,
                        required=False,
                        default='merlin-pipeline',
                        help="Name of the original pipeline")

    parser.add_argument("--min_log_length",
                        type=int,
                        required=False,
                        default=320,
                        help="Minimum number of req of the .parquet/.csv that is created")

    parser.add_argument("--log_time_delta",
                        type=int,
                        required=False,
                        default=60,
                        help="Minimum amount of time delta (in secs) between two subsequent .parquet files")

    parser.add_argument("--PV_loc",
                        type=str,
                        required=False,
                        default='/var/lib/data/new_data/',
                        help="Location of PV to write the files")


    parser.add_argument("--kfp_host",
                        type=str,
                        required=True,
                        help="Kubeflow Pipelines API endpoint")

    parser.add_argument("--max_event_age_seconds",
                        type=int,
                        default=300,
                        help="Maximum accepted event and publish age")

    parser.add_argument("--max_future_skew_seconds",
                        type=int,
                        default=60,
                        help="Maximum accepted clock skew into the future")

    parser.add_argument("--trigger_cooldown_seconds",
                        type=int,
                        default=3600,
                        help="Minimum time between retraining trigger attempts")

    args = parser.parse_args()

    if args.evaluate_period < 2:
        parser.error("--evaluate_period must be at least 2")
    if not 0 < args.min_trigger_len <= 1:
        parser.error("--min_trigger_len must be greater than 0 and at most 1")
    if not 0 <= args.acc_threshold <= 1:
        parser.error("--acc_threshold must be between 0 and 1")
    if args.min_log_length < 1:
        parser.error("--min_log_length must be positive")
    if args.log_time_delta < 0:
        parser.error("--log_time_delta cannot be negative")
    if args.max_event_age_seconds < 1:
        parser.error("--max_event_age_seconds must be positive")
    if args.max_future_skew_seconds < 0:
        parser.error("--max_future_skew_seconds cannot be negative")
    if args.trigger_cooldown_seconds < 0:
        parser.error("--trigger_cooldown_seconds cannot be negative")

    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO, datefmt='%d-%m-%y %H:%M:%S')
    logging.info(f"Args: {args}")

    logging.info("Starting accuracy monitor...")


    am = AccMonitor(project_id=args.project_id,
                    subscription_id=args.subscription_id,
                    timeout=args.timeout,
                    evaluate_period=args.evaluate_period,
                    acc_threshold=args.acc_threshold,
                    topic_id=args.topic_id,
                    min_trigger_len=args.min_trigger_len,
                    pipeline_name=args.pipeline_name,
                    min_log_length=args.min_log_length,
                    log_time_delta=args.log_time_delta,
                    pv_location=args.PV_loc,
                    expected_model_name=args.expected_model_name,
                    client_host=args.kfp_host,
                    max_event_age_seconds=args.max_event_age_seconds,
                    max_future_skew_seconds=args.max_future_skew_seconds,
                    trigger_cooldown_seconds=args.trigger_cooldown_seconds)

    am.run()
