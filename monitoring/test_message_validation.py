# Copyright (c) 2026 NVIDIA Corporation. All Rights Reserved.
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

import datetime
import json
import unittest
import uuid

from message_validation import (
    MAX_PAYLOAD_BYTES,
    PayloadValidationError,
    validate_payload,
    validate_publish_time,
)


NOW = datetime.datetime(2026, 8, 7, 12, 0, tzinfo=datetime.timezone.utc)


def valid_payload():
    return {
        "schema_version": 1,
        "event_id": str(uuid.uuid4()),
        "event_time": NOW.isoformat().replace("+00:00", "Z"),
        "model_name": "dcn_ens",
        "model_version": "current",
        "input0": [0.5] * 13,
        "input1": list(range(26)),
        "trueval": 1,
        "predval": 0.75,
    }


def encode(payload):
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


class MessageValidationTest(unittest.TestCase):
    def test_accepts_and_normalizes_valid_payload(self):
        payload = valid_payload()

        result = validate_payload(encode(payload), now=NOW)

        self.assertEqual(result["event_id"], str(uuid.UUID(payload["event_id"])))
        self.assertEqual(result["event_time"], "2026-08-07T12:00:00Z")
        self.assertEqual(result["input0"], [0.5] * 13)

    def test_rejects_unexpected_fields(self):
        payload = valid_payload()
        payload["command"] = "retrain"

        with self.assertRaises(PayloadValidationError):
            validate_payload(encode(payload), now=NOW)

    def test_rejects_duplicate_json_keys(self):
        raw = encode(valid_payload())
        raw = raw[:-1] + b',"schema_version":1}'

        with self.assertRaises(PayloadValidationError):
            validate_payload(raw, now=NOW)

    def test_rejects_wrong_feature_shape(self):
        payload = valid_payload()
        payload["input0"] = [0.5] * 12

        with self.assertRaises(PayloadValidationError):
            validate_payload(encode(payload), now=NOW)

    def test_rejects_nonfinite_values(self):
        payload = valid_payload()
        payload["predval"] = float("nan")

        with self.assertRaises(PayloadValidationError):
            validate_payload(encode(payload), now=NOW)

    def test_rejects_stale_or_future_events(self):
        stale = valid_payload()
        stale["event_time"] = (NOW - datetime.timedelta(seconds=301)).isoformat()
        future = valid_payload()
        future["event_time"] = (NOW + datetime.timedelta(seconds=61)).isoformat()

        with self.assertRaises(PayloadValidationError):
            validate_payload(encode(stale), now=NOW)
        with self.assertRaises(PayloadValidationError):
            validate_payload(encode(future), now=NOW)

    def test_rejects_oversized_payload(self):
        with self.assertRaises(PayloadValidationError):
            validate_payload(b"x" * (MAX_PAYLOAD_BYTES + 1), now=NOW)

    def test_rejects_invalid_labels_and_identifiers(self):
        payload = valid_payload()
        payload["trueval"] = True

        with self.assertRaises(PayloadValidationError):
            validate_payload(encode(payload), now=NOW)

        payload = valid_payload()
        payload["model_name"] = "../../other-model"
        with self.assertRaises(PayloadValidationError):
            validate_payload(encode(payload), now=NOW)

    def test_rejects_events_for_another_model(self):
        with self.assertRaises(PayloadValidationError):
            validate_payload(
                encode(valid_payload()),
                now=NOW,
                expected_model_name="another_model",
            )

    def test_validates_publish_time_freshness(self):
        self.assertEqual(validate_publish_time(NOW, now=NOW), NOW)
