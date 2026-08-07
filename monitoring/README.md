# Monitoring event security

The monitoring service treats every Pub/Sub message as untrusted input. It only
accepts JSON in `message.data`; message attributes are not used as monitoring
or retraining input.

## Event contract

Each event must contain exactly these fields:

| Field | Requirement |
| --- | --- |
| `schema_version` | Integer `1` |
| `event_id` | Unique UUID |
| `event_time` | Timezone-aware ISO-8601 timestamp |
| `model_name` | Configured model name |
| `model_version` | 1-128 character identifier |
| `input0` | Exactly 13 finite numeric values |
| `input1` | Exactly 26 non-negative integer values |
| `trueval` | Integer `0` or `1` |
| `predval` | Finite number between `0` and `1` |

The encoded payload must be valid UTF-8 JSON and no larger than 64 KiB.
Duplicate keys, unknown fields, non-finite values, malformed identifiers, and
out-of-range values are rejected.

Events and their Pub/Sub publish times must be recent. The defaults allow an age
of five minutes and future clock skew of one minute. Event UUIDs are cached to
ignore duplicate deliveries, and retraining trigger attempts have a default
one-hour cooldown.

## Topic binding and access

Set all of the following chart values:

- `pubsub`: subscription consumed by the monitoring service
- `pubsub_topic`: the only topic that subscription may reference
- `expected_model_name`: the only model whose events are accepted
- `trigger_cooldown_seconds`: minimum interval between trigger attempts

The monitor reads the subscription metadata at startup and exits if the
subscription is not attached to the configured topic.

Pub/Sub does not provide the publisher identity to subscribers, so topic IAM is
the provenance boundary for these events:

- Grant the inference publisher `roles/pubsub.publisher` only on the expected
  topic.
- Grant the monitoring workload identity `roles/pubsub.subscriber` only on
  the expected subscription.
- Do not grant broad project-level publishing permissions.
- Do not grant `allUsers` or `allAuthenticatedUsers` access to the topic or
  subscription.

Malformed, expired, mismatched-model, and duplicate messages are acknowledged
and ignored so poison messages do not create an endless redelivery loop. Valid
messages are acknowledged only after they have entered the bounded processing
queue.

The publisher in `perf-monitor-test.py` emits the required schema. Each
publisher should wait for the publish future and treat a publish failure as a
failed inference event.
