"""
predict.py
Handles single-student prediction using the trained model.
"""

import numpy as np


def predict_student(model, scaler, study_hours: float, attendance: float, previous_marks: float):
    """
    Predict PASS/FAIL for a single student.

    Returns:
        label       (str)  : "PASS" or "FAIL"
        confidence  (float): probability of the predicted class (0–1)
        proba_pass  (float): raw probability of PASS
    """
    input_data = np.array([[study_hours, attendance, previous_marks]])
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    label = "PASS" if prediction == 1 else "FAIL"

    # Confidence / probability
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(input_scaled)[0]   # [P(FAIL), P(PASS)]
        proba_pass = probas[1]
        confidence = probas[int(prediction)]
    else:
        proba_pass = float(prediction)
        confidence = 1.0

    return label, round(confidence, 4), round(proba_pass, 4)
