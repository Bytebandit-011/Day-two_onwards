# 🎓 Day 4 – Teach-the-Tutor: Active Recall Coach  
Part of the **Murf AI Voice Agents Challenge**

This agent helps users learn using one of the most powerful retention techniques: **learn → quiz → teach back**.  
The agent uses *different voices* for each mode, powered by **Murf Falcon**.

---

## 🧠 Overview

The agent supports **three modes**, each with its own behavior and voice:

| Mode        | Behavior | Voice (Murf Falcon) |
|-------------|----------|----------------------|
| **learn**       | Explains a concept using summary text | Matthew |
| **quiz**        | Asks questions from the content file | Alicia |
| **teach_back**  | User teaches the concept back; agent gives qualitative feedback | Ken |

The user can switch modes at any time by saying things like:
- “Switch to quiz mode”
- “Let’s learn about loops”
- “I want to teach back variables”

---

## 📚 Content File (JSON)

All course content lives in:

Example:

```json
[
  {
    "id": "variables",
    "title": "Variables",
    "summary": "Variables store values so you can reuse them later...",
    "sample_question": "What is a variable and why is it useful?"
  },
  {
    "id": "loops",
    "title": "Loops",
    "summary": "Loops let you repeat an action multiple times...",
    "sample_question": "Explain the difference between a for loop and a while loop."
  }
]
