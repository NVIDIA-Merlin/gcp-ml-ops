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

import datetime
import json
import math
import re
import uuid


SCHEMA_VERSION = 1
CONTINUOUS_FEATURE_COUNT = 13
CATEGORICAL_FEATURE_COUNT = 26
MAX_PAYLOAD_BYTES = 64 * 1024
MAX_ABSOLUTE_CONTINUOUS_VALUE = 1_000_000_000_000
MAX_CATEGORICAL_VALUE = (2 ** 63) - 1

_REQUIRED_FIELDS = frozenset({
    "schema_version",
    "event_id",
    "event_time",
    "model_name",
    "model_version",
    "input0",
    "input1",
    "trueval",
    "predval",
})
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class PayloadValidationError(ValueError):
    """Raised when an inference event cannot be trusted as monitoring input."""


def _reject_nonfinite_json(value):
    raise PayloadValidationError("non-finite JSON numbers are not allowed")


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise PayloadValidationError("duplicate JSON object key")
        result[key] = value
    return result


def _validate_timestamp_freshness(
    value,
    now,
    max_age_seconds,
    max_future_skew_seconds,
    field_name,
):
    if not isinstance(value, datetime.datetime) or value.tzinfo is None:
        raise PayloadValidationError(f"{field_name} must include a timezone")

    value = value.astimezone(datetime.timezone.utc)
    age_seconds = (now - value).total_seconds()
    if age_seconds > max_age_seconds:
        raise PayloadValidationError(f"{field_name} is too old")
    if age_seconds < -max_future_skew_seconds:
        raise PayloadValidationError(f"{field_name} is too far in the future")
    return value


def validate_publish_time(
    publish_time,
    now=None,
    max_age_seconds=300,
    max_future_skew_seconds=60,
):
    now = now or datetime.datetime.now(datetime.timezone.utc)
    return _validate_timestamp_freshness(
        publish_time,
        now,
        max_age_seconds,
        max_future_skew_seconds,
        "publish_time",
    )


def _validate_identifier(value, field_name):
    if not isinstance(value, str) or not _IDENTIFIER_PATTERN.fullmatch(value):
        raise PayloadValidationError(
            f"{field_name} must be 1-128 letters, numbers, dots, underscores, or dashes"
        )
    return value


def _validate_continuous_features(values):
    if not isinstance(values, list) or len(values) != CONTINUOUS_FEATURE_COUNT:
        raise PayloadValidationError(
            f"input0 must contain exactly {CONTINUOUS_FEATURE_COUNT} values"
        )

    normalized = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise PayloadValidationError("input0 values must be numeric")
        value = float(value)
        if not math.isfinite(value) or abs(value) > MAX_ABSOLUTE_CONTINUOUS_VALUE:
            raise PayloadValidationError("input0 contains an out-of-range value")
        normalized.append(value)
    return normalized


def _validate_categorical_features(values):
    if not isinstance(values, list) or len(values) != CATEGORICAL_FEATURE_COUNT:
        raise PayloadValidationError(
            f"input1 must contain exactly {CATEGORICAL_FEATURE_COUNT} values"
        )

    normalized = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise PayloadValidationError("input1 values must be integers")
        if value < 0 or value > MAX_CATEGORICAL_VALUE:
            raise PayloadValidationError("input1 contains an out-of-range value")
        normalized.append(value)
    return normalized


def validate_payload(
    raw_payload,
    now=None,
    max_event_age_seconds=300,
    max_future_skew_seconds=60,
    max_payload_bytes=MAX_PAYLOAD_BYTES,
    expected_model_name=None,
):
    if not isinstance(raw_payload, (bytes, bytearray)):
        raise PayloadValidationError("payload must be UTF-8 bytes")
    if not raw_payload or len(raw_payload) > max_payload_bytes:
        raise PayloadValidationError("payload size is invalid")

    try:
        payload_text = bytes(raw_payload).decode("utf-8", errors="strict")
        payload = json.loads(
            payload_text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PayloadValidationError("payload is not valid UTF-8 JSON") from error

    if not isinstance(payload, dict):
        raise PayloadValidationError("payload must be a JSON object")

    fields = frozenset(payload)
    if fields != _REQUIRED_FIELDS:
        missing = sorted(_REQUIRED_FIELDS - fields)
        unexpected = sorted(fields - _REQUIRED_FIELDS)
        raise PayloadValidationError(
            f"payload fields do not match the schema; missing={missing}, unexpected={unexpected}"
        )

    if payload["schema_version"] != SCHEMA_VERSION:
        raise PayloadValidationError("unsupported schema_version")

    try:
        event_id = str(uuid.UUID(payload["event_id"]))
    except (AttributeError, TypeError, ValueError) as error:
        raise PayloadValidationError("event_id must be a UUID") from error

    event_time_text = payload["event_time"]
    if not isinstance(event_time_text, str):
        raise PayloadValidationError("event_time must be an ISO-8601 string")
    try:
        event_time = datetime.datetime.fromisoformat(
            event_time_text.replace("Z", "+00:00")
        )
    except ValueError as error:
        raise PayloadValidationError("event_time must be an ISO-8601 timestamp") from error

    now = now or datetime.datetime.now(datetime.timezone.utc)
    event_time = _validate_timestamp_freshness(
        event_time,
        now,
        max_event_age_seconds,
        max_future_skew_seconds,
        "event_time",
    )
    model_name = _validate_identifier(payload["model_name"], "model_name")
    model_version = _validate_identifier(payload["model_version"], "model_version")
    if expected_model_name is not None:
        expected_model_name = _validate_identifier(
            expected_model_name, "expected_model_name"
        )
        if model_name != expected_model_name:
            raise PayloadValidationError("model_name does not match the configured model")

    true_value = payload["trueval"]
    if isinstance(true_value, bool) or true_value not in (0, 1):
        raise PayloadValidationError("trueval must be either 0 or 1")

    predicted_value = payload["predval"]
    if isinstance(predicted_value, bool) or not isinstance(
        predicted_value, (int, float)
    ):
        raise PayloadValidationError("predval must be numeric")
    predicted_value = float(predicted_value)
    if not math.isfinite(predicted_value) or not 0.0 <= predicted_value <= 1.0:
        raise PayloadValidationError("predval must be between 0 and 1")

    return {
        "schema_version": SCHEMA_VERSION,
        "event_id": event_id,
        "event_time": event_time.isoformat().replace("+00:00", "Z"),
        "model_name": model_name,
        "model_version": model_version,
        "input0": _validate_continuous_features(payload["input0"]),
        "input1": _validate_categorical_features(payload["input1"]),
        "trueval": int(true_value),
        "predval": predicted_value,
    }
