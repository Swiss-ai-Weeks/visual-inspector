# MoveMatch — Technical Explanation

How the app decides which player performed a required sequence of movements
correctly, in order, and within the time limit — the "fastest person" result.

Everything below is **local, deterministic, and offline**: no VSS, no LLM, no
network. The perception model (`YOLOv8n-pose` ONNX) runs on CPU via
`cv2.dnn`. Given the same video and sequence, the app always produces the same
winner.

## The pipeline at a glance

```
video ──▶ ensure_mp4 ──▶ extract_events ──▶ validate ──▶ pick winner ──▶ snapshot
          (app.py)        (detect.py)        (validator.py)  (app.py)      (detect.py)
```

1. **Normalize** the upload to H.264/AAC MP4 (`ensure_mp4`, [app.py:122](app/app.py#L122)).
2. **Perceive** — sample frames, run pose detection, track each player, and
   classify their action over time into `ActionEvent`s (`extract_events`,
   [detect.py:240](app/detect.py#L240)).
3. **Validate** — for each player, check the detected action sequence against
   the expected one, in order, under the time limit (`validate`,
   [validator.py:18](app/validator.py#L18)).
4. **Pick the winner** and box them in a frame (`run_pipeline`,
   [app.py:147](app/app.py#L147); `winner_snapshot`, [detect.py:341](app/detect.py#L341)).

## Step 1 — Perception: from pixels to `ActionEvent`s

`extract_events` ([detect.py:240](app/detect.py#L240)) is the heart of perception.

**Frame sampling.** The video is walked frame by frame, but only every `dt`
seconds is actually analyzed (`step = round(fps * dt)`, default `dt = 0.2s`, so
~5 samples/second). This keeps the run fast (~1.5s for a 4-person clip) while
still resolving short actions.

**Pose detection.** Each sampled frame goes through `_pose`
([detect.py:63](app/detect.py#L63)): letterbox to 640×640, run the ONNX network,
and for every person above `PERSON_CONF` (0.5) collect the bounding-box center
`cx`, box confidence, and the 17 COCO keypoints (nose, eyes, shoulders, elbows,
wrists, hips, knees, ankles). Overlapping boxes are removed with NMS.

**Tracking players by column.** Players are identified purely by horizontal
position — **left → right, player 1 is leftmost**. First the app finds the
typical player count (`num_players = mode of per-frame detection counts`), then
computes a stable column center for each player from frames that cleanly see
everyone ([detect.py:277](app/detect.py#L277)). In every sampled frame, each
detected person is assigned to the nearest column center
([detect.py:295](app/detect.py#L295)); if two people land on the same column,
the higher-confidence detection wins. This is a deliberately simple tracker: it
assumes players stay in their lanes and don't cross, which holds for the
"line up and perform" game format.

**Action classification.** For each player in each sampled frame, `_classify`
([detect.py:162](app/detect.py#L162)) maps the keypoint geometry to one action
label. It is pure geometry, normalized by shoulder width `sw` so it's
scale-invariant:

- `turn_around` — shoulders visible but face keypoints hidden (back to camera).
- `hands_on_head` — both wrists lifted to head level, near the head.
- `raise_one_arm` — exactly one wrist well above the shoulders.
- `arms_crossed` — forearms roughly horizontal across the chest, hands drawn in.
- `clap` — hands together at chest height, wrists raised above elbows (forearm V).
- `crouch` — knees drawn up toward the hips.
- `idle` — none of the above.

User-defined **custom gestures** (Test page only) are checked *first*, so they
override the built-ins ([detect.py:177](app/detect.py#L177)); each is a set of
AND-ed body-part relations (e.g. "left_wrist above head").

**Segmenting into events.** Per player, the frame-by-frame action stream is a
noisy timeline like `idle, idle, clap, clap, clap, idle, crouch, ...`.
`_segments` ([detect.py:311](app/detect.py#L311)) collapses each run of the same
action into a single `ActionEvent` with `t_start` and `t_end`. Two filters keep
it clean:

- Runs shorter than `min_samples` (default 2 samples) are dropped as flicker.
- `idle` runs are never emitted as events.

The output is a flat list of `ActionEvent(player_id, action, confidence,
t_start, t_end)` across all players.

## Step 2 — Validation: correct sequence, correct order, in time

`validate` ([validator.py:18](app/validator.py#L18)) groups events by player and
builds a `PlayerResult` for each. Per player:

1. Sort events by start time, then `_dedupe_consecutive` merges adjacent
   same-action events, and low-confidence / `idle` events are dropped
   ([validator.py:37](app/validator.py#L37)).
2. `seq` = the player's ordered list of action labels.
3. **Order check** — in strict mode, the first `len(expected)` detected actions
   must equal the expected sequence exactly
   (`seq[:len(expected)] == expected`, [validator.py:48](app/validator.py#L48)).
   In `lenient` mode, it's enough that the expected actions appear *in order* as
   a subsequence (longest-common-subsequence length equals the expected length).
4. **Time check** — `time_ok` is true when the player's *last* event ends within
   the limit: `evts[-1].t_end <= time_limit` (default 5s,
   [validator.py:42](app/validator.py#L42)).

A player **passes** only if `ok_order and time_ok`. Players who get the order
right but run over time still fail; a `score` (1.0 for a clean pass, otherwise a
partial LCS-based fraction) is recorded for everyone so near-misses are visible.

## Step 3 — Where "fastest" actually comes from

There are two things worth being precise about, because the phrase "fastest
person" is doing some work here:

**The time limit is a gate, not a stopwatch ranking.** The app does not compute
"who finished the sequence in the fewest seconds and rank by that number."
Instead, every player is measured against the same clock: performing the full
sequence correctly *before the time limit runs out* is what it means to be
"fast enough." Anyone whose final action lands after `time_limit` is
disqualified on `time_ok`. So the winner is not necessarily the person with the
smallest `t_end` — it's a person who satisfied both order and time.

**Winner selection is first-passing, in player order.** The winner is:

```python
winner = next((p for p in players if p.passed), None)   # app.py:165
```

`players` is ordered by `player_id`, i.e. left → right. So among all players who
passed, the app picks the **leftmost passing player**. If you need "fastest" to
mean strictly the smallest completion time, that logic is *not* in the code
today — it would be a one-line change here to instead pick
`min(passers, key=lambda p: p.events[-1].t_end)`.

In the common game setup (one required sequence, players racing the same clock)
these usually coincide: typically only the player who nailed the sequence in
time passes, and they become the winner. But with multiple passers the current
rule is positional, not temporal — worth knowing if results ever look
surprising.

## Step 4 — Presenting the winner

If there's a winner, `run_pipeline` ([app.py:167](app/app.py#L167)) calls
`winner_snapshot` ([detect.py:341](app/detect.py#L341)), which re-scans the video
for the frame where that player (by left→right rank) is most confidently
detected, draws a green box with their name, and saves a JPEG. The full
`GameResult` (expected sequence, every `PlayerResult`, raw events, winner,
snapshot filename) is returned to the template and persisted to the `games`
table in `movematch.db`.

## Key parameters

| Parameter | Location | Default | Effect |
|-----------|----------|---------|--------|
| `dt` | `extract_events` | 0.2s | Frame sampling interval (~5 fps analyzed) |
| `min_samples` | `extract_events` | 2 | Minimum run length to emit an event (flicker filter) |
| `PERSON_CONF` | `detect.py` | 0.5 | Minimum person-detection confidence |
| `TIME_LIMIT` | `app.py` / `validator.py` | 5.0s | Deadline for the last action |
| `CONFIDENCE_THRESHOLD` | `validator.py` | 0.5 | Events at/below this are ignored |
| `SEQUENCE_LENGTH` | `app.py` | 3 | Number of actions in a generated challenge |
| `lenient` | request form | off | Subsequence match instead of exact prefix match |

## Why it's built this way

The perception layer was deliberately switched from the VSS CV pipeline + LLM
adjudication to this local pose approach because the CV pipeline stalled with
zero detections and the VLM couldn't reliably ground *which* player did *what*
and *when*. Local pose geometry gives correct, repeatable per-player tracking,
action labels, and timing in about a second and a half — and, being fully
deterministic, it makes the winner defensible: the same video always yields the
same result.
