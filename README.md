# 🎮 Motion Sequence Challenge

> **A video-based game where players watch, remember, and reproduce sequences of movements — while AI checks whether they performed the correct actions in the correct order.**

## 🚀 Project Overview

**Motion Sequence Challenge** is a fun, interactive game inspired by sequence-based games such as *Taco, Chapeau, Gâteau, Cadeau, Pizza*.

The concept is simple:

1. Players are shown a sequence of movements on a screen.
2. They have a limited amount of time to reproduce the sequence.
3. A camera records their movements.
4. AI detects the players and analyzes their actions over time.
5. The system determines whether each player reproduced the **right movements, in the right order**.
6. Players receive a success/failure result and are ranked against the others.

The project combines **computer vision, temporal action understanding, and gamification** into a local web application.

---

## 🎯 Main Goal

The main goal is to build an AI-powered system capable of answering:

> **"Did this person correctly reproduce the requested sequence of movements?"**

This is more than simple pose or object detection. The system needs to understand **what happened over time** and verify that the required sequence of actions was followed.

For example:

```text
Displayed sequence:

👏  →  🙆  →  👇  →  👏

Player:

👏  →  🙆  →  👇  →  👏

             ✅ SUCCESS
```

Whereas:

```text
Displayed sequence:

👏  →  🙆  →  👇  →  👏

Player:

👏  →  👇  →  🙆  →  👏

             ❌ FAILED
             Wrong order
```

---

# 🏆 Hackathon MVP

Our objective is to build a functional end-to-end prototype during the hackathon.

### Core Features

* 📹 **Upload a video**

  * Upload recorded footage of players performing the challenge.

* 📝 **Display instructions**

  * Show the sequence of movements that players need to reproduce.

* 👤 **Person detection & tracking**

  * Detect participants in the video.
  * Assign a persistent ID to each person.
  * Track each participant throughout the sequence.

* 🕺 **Movement detection**

  * Determine which movement each participant performs.

* 🔢 **Sequence verification**

  * Check whether the participant performed:

    * the correct movements,
    * in the correct order,
    * within the allowed time.

* ⏱️ **5-second response window**

  * Players have **5 seconds** to reproduce the requested sequence.

* ✅ **Result**

  * Determine whether each participant succeeded or failed.

* 🏅 **Ranking**

  * Rank participants based on their performance.

* 💾 **Persistent IDs**

  * Player IDs should remain consistent between sessions.

* 🗄️ **Local data storage**

  * Keep all relevant data locally during the hackathon.

* 🌐 **Local web interface**

  * Provide a simple web interface to run and interact with the game locally.

---

# 🧠 AI Strategy

The core challenge is understanding **actions over time**, rather than recognizing a single frame.

Our initial strategy is divided into two main components.

### 1. Person Detection & Tracking

First, we detect all people appearing in the video.

Each detected person receives an ID:

```text
Frame 1       Frame 2       Frame 3       Frame 4

Person A      Person A      Person A      Person A
   ID: 01         ↓             ↓             ↓

Person B      Person B      Person B      Person B
   ID: 02         ↓             ↓             ↓
```

The system then follows each person throughout the video so that their movements can be evaluated independently.

### 2. Sequence Verification

For each participant, we analyze their actions chronologically.

Given an expected sequence:

```text
A → B → C → D
```

we want to determine whether the participant actually performed:

```text
A → B → C → D     ✅
```

rather than:

```text
A → C → B → D     ❌
```

or:

```text
A → B → D        ❌
```

The important aspect is therefore **temporal understanding**: not just *what movement happened*, but **when it happened and in which order**.

---

# 🎁 Bonus Features

If the MVP is completed, we would like to extend the game with additional features.

### 📷 Real-Time Gameplay

Move from uploaded videos to a **live camera experience**, allowing players to play the game in real time.

### 📈 Difficulty Levels

Introduce different difficulty levels based on:

* Number of movements
* Time available
* Sequence complexity

For example:

```text
Easy       👏 → 🙆

Medium     👏 → 🙆 → 👇 → 👏

Hard       👏 → 🙆 → 👇 → 🤸 → 👏 → 🙆
```

### 🎨 Visual Representation

Display a drawing or animation representing the required movement.

### ✋ Raise Hand to Participate

Automatically detect when someone raises their hand to indicate that they want to participate.

### 🏆 Keep the Top 80%

Maintain a leaderboard and allow the best-performing participants to continue to the next round.

### 🔊 Audio Instructions

Provide spoken instructions alongside the visual instructions.

### 🔔 Synchronization Beeps

Use audio cues or beeps to synchronize participants and make the game easier to follow.

---

# 🏗️ High-Level Architecture

```text
                  ┌─────────────────────┐
                  │   Web Interface     │
                  │                     │
                  │ Instructions        │
                  │ Video Upload        │
                  │ Results / Ranking   │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │    Video Pipeline   │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Person Detection &  │
                  │       Tracking      │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Movement / Action   │
                  │     Detection       │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Sequence Verification│
                  │                     │
                  │ Correct movement?   │
                  │ Correct order?      │
                  │ Correct timing?     │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │       Results       │
                  │                     │
                  │  ✅ Success         │
                  │  ❌ Failure         │
                  │  🏆 Ranking         │
                  └─────────────────────┘
```

---

# 🎯 Hackathon Challenge Alignment

Our project directly addresses the **Visual Compliance Inspector** challenge.

The original challenge asks whether AI can watch a video and determine if a procedure was followed correctly.

Our game applies the same concept to a playful setting:

> **Instead of checking whether a worker followed a maintenance procedure, we check whether a player followed a sequence of movements.**

Both problems require the system to:

* Detect people
* Track individuals over time
* Recognize actions
* Understand temporal sequences
* Compare observed actions against expected actions
* Identify missing or incorrect steps
* Produce a final compliance result

This makes the game a **gamified visual compliance inspector**.

---

# 🔬 Why This Is Interesting

Traditional computer vision can answer questions such as:

> "Is there a person in this image?"

Our project aims to answer a more complex question:

> **"Did this particular person perform the correct sequence of actions within the required time?"**

This introduces several interesting AI challenges:

* **Multi-person tracking**
* **Action recognition**
* **Temporal reasoning**
* **Sequence matching**
* **Identity persistence**
* **Real-time video analysis**

The same underlying technology could eventually be applied beyond games — for example to **sports training, industrial procedures, education, rehabilitation, or workplace safety**.

---

# 🗺️ Roadmap

### Phase 1 — MVP

* [ ] Local web interface
* [ ] Display movement instructions
* [ ] Upload video
* [ ] Detect people
* [ ] Assign persistent IDs
* [ ] Track participants
* [ ] Detect movements
* [ ] Validate movement sequence
* [ ] Apply 5-second time limit
* [ ] Display success/failure
* [ ] Generate ranking
* [ ] Store data locally

### Phase 2 — Gameplay Improvements

* [ ] Difficulty levels
* [ ] Audio instructions
* [ ] Synchronization sounds
* [ ] Visual movement illustrations
* [ ] "Raise hand to participate"
* [ ] Top 80% qualification system

### Phase 3 — Real-Time

* [ ] Live camera input
* [ ] Real-time person tracking
* [ ] Real-time movement detection
* [ ] Real-time sequence validation
* [ ] Live leaderboard

---

# 🎮 Vision

Our long-term vision is to turn computer vision into a **game master**.

The system should be able to look at a group of people, understand what each person is doing, determine whether they followed the rules, and automatically decide who wins.

**Watch. Move. Get recognized. Climb the leaderboard. 🏆**

---

## 🏁 Hackathon Goal

By the end of the hackathon, we aim to have a working local prototype demonstrating the complete loop:

```text
Instruction
     ↓
Player performs movements
     ↓
Video analysis
     ↓
Person detection & tracking
     ↓
Action recognition
     ↓
Sequence verification
     ↓
Success / Failure
     ↓
Leaderboard
```

**The goal is not just to recognize movement — it is to understand the movement sequence and determine whether the player followed the instructions.**
