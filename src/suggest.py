"""
suggest.py
Rule-based AI suggestion engine for student improvement tips.
"""

from typing import List, Dict


def generate_suggestions(study_hours: float, attendance: float, previous_marks: float) -> List[Dict]:
    """
    Return a list of suggestion dicts based on input values.

    Each dict has:
        - icon    (str): emoji icon
        - title   (str): short heading
        - detail  (str): actionable advice
        - level   (str): "warning" | "error" | "success"
    """
    suggestions = []

    # ── Study Hours ─────────────────────────────────────────────────────────
    if study_hours < 2:
        suggestions.append({
            "icon": "📚",
            "title": "Very Low Study Time",
            "detail": (
                "You're studying less than 2 hours/day. "
                "Try to dedicate at least 4–5 hours of focused study. "
                "Use the Pomodoro technique (25-min focus + 5-min break) to stay consistent."
            ),
            "level": "error",
        })
    elif study_hours < 4:
        suggestions.append({
            "icon": "⏱️",
            "title": "Increase Study Time",
            "detail": (
                "Aim for at least 4–6 hours of study per day. "
                "Create a daily timetable and avoid distractions during study sessions."
            ),
            "level": "warning",
        })
    else:
        suggestions.append({
            "icon": "✅",
            "title": "Good Study Habit",
            "detail": "You're putting in solid study hours. Keep it up and stay consistent!",
            "level": "success",
        })

    # ── Attendance ───────────────────────────────────────────────────────────
    if attendance < 50:
        suggestions.append({
            "icon": "🚨",
            "title": "Critical Attendance Issue",
            "detail": (
                "Attendance below 50% is a serious concern. "
                "You may be missing key lectures and evaluations. "
                "Contact your academic advisor immediately."
            ),
            "level": "error",
        })
    elif attendance < 70:
        suggestions.append({
            "icon": "🏫",
            "title": "Improve Attendance",
            "detail": (
                "Attendance below 70% negatively impacts learning and grades. "
                "Try to attend all scheduled classes — even recorded ones don't replace live sessions."
            ),
            "level": "warning",
        })
    else:
        suggestions.append({
            "icon": "✅",
            "title": "Strong Attendance",
            "detail": "Great — you're showing up consistently. Keep maintaining this habit.",
            "level": "success",
        })

    # ── Previous Marks ───────────────────────────────────────────────────────
    if previous_marks < 35:
        suggestions.append({
            "icon": "🆘",
            "title": "Urgent: Academic Support Needed",
            "detail": (
                "Marks below 35 indicate a critical gap. "
                "Seek tutoring, revisit foundational concepts, and consider extra coaching classes."
            ),
            "level": "error",
        })
    elif previous_marks < 50:
        suggestions.append({
            "icon": "📝",
            "title": "Revise Core Concepts",
            "detail": (
                "Focus on revising fundamental topics and solving past exam papers. "
                "Practice regularly and aim for conceptual clarity rather than rote learning."
            ),
            "level": "warning",
        })
    elif previous_marks < 70:
        suggestions.append({
            "icon": "📈",
            "title": "Room for Improvement",
            "detail": (
                "You're doing okay but there's room to push higher. "
                "Work on time management during exams and review mistakes from past tests."
            ),
            "level": "warning",
        })
    else:
        suggestions.append({
            "icon": "🌟",
            "title": "Excellent Academic Record",
            "detail": (
                "Your previous marks are excellent! "
                "Stay consistent, challenge yourself with advanced problems, and mentor peers."
            ),
            "level": "success",
        })

    # ── Overall Summary ──────────────────────────────────────────────────────
    all_good = all(s["level"] == "success" for s in suggestions)
    if all_good:
        suggestions.append({
            "icon": "🏆",
            "title": "Outstanding Performance!",
            "detail": (
                "All indicators are positive. You're on track for excellent results. "
                "Keep your momentum and inspire those around you!"
            ),
            "level": "success",
        })

    return suggestions
