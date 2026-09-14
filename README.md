# 🎮 MoveMatch

> **Watch the sequence. Reproduce the moves. Let AI decide who got it right.**

**Motion Sequence Challenge** is an AI-powered movement game where players reproduce a sequence of actions shown on screen.

A camera records the players, and the app uses computer vision to determine **who performed the correct movements, in the correct order, within the time limit**.

## 🎯 How It Works

**1. Watch**
The app displays a movement sequence.

👏 → 🙆 → 👇 → 👏

**2. Perform**
Players have **5 seconds** to reproduce it.

**3. Analyze**
The AI detects and tracks each player, recognizes their movements, and reconstructs the sequence.

**4. Verify**
Expected:

👏 → 🙆 → 👇 → 👏

Player:

👏 → 🙆 → 👇 → 👏 ✅

or:

👏 → 👇 → 🙆 → 👏 ❌

**5. Rank**
The app shows success/failure and updates the leaderboard.

---

## 🧠 What the AI Does

The key challenge is not simply detecting a person or a movement.

The AI needs to understand **actions over time**:

**Video → Player Detection → Tracking → Action Recognition → Sequence Matching → Result**

For every player, the system checks:

* Did they perform the correct movements?
* Were they performed in the correct order?
* Were they completed within the allowed time?

---

## 🚀 Hackathon MVP

Our goal is a working end-to-end local web app with:

* 📹 Video upload
* 🎯 Movement sequence display
* 👤 Multi-person detection & tracking
* 🕺 Movement recognition
* 🔢 Sequence verification
* ⏱️ 5-second challenge timer
* ✅ Success / failure results
* 🏆 Player leaderboard
* 💾 Local data storage

### MVP Flow

**Challenge → Perform → Record → Analyze → Verify → Score → Leaderboard**

---

## 🏗️ Architecture

```text
┌─────────────────────┐
│      Web App        │
│ Challenge + Video   │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│   Video Analysis    │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ Person Detection &  │
│      Tracking       │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ Action Recognition  │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ Sequence Validation │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ Results + Ranking   │
└─────────────────────┘
```

---

## 🏆 Why It Fits the Hackathon

The project turns the **Visual Compliance Inspector** challenge into a game.

Instead of asking:

> “Did a worker correctly follow this procedure?”

we ask:

> **“Did this player correctly follow this movement sequence?”**

The underlying AI problem is the same: detect people, recognize actions, understand their order, and compare what happened against an expected procedure.

---

## ✨ If We Have Time

After the MVP:

* 📷 Live camera gameplay
* 📈 Difficulty levels
* 🔊 Audio instructions and synchronization
* ✋ Raise-hand player registration
* 🏆 Tournament rounds / top 80% qualification
* ⚡ Real-time movement validation

---

## 🔭 Beyond the Hackathon

The same technology could be applied to **sports training, industrial procedures, education, rehabilitation, and workplace safety**.

### The Vision

Turn computer vision into an **AI game master** that can watch multiple people, understand what they are doing, check whether they followed the rules, and automatically determine the winner.

**Watch. Move. Get recognized. Climb the leaderboard. 🏆**
