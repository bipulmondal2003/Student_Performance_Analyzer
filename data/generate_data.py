import pandas as pd
import numpy as np

np.random.seed(42)
n = 200

study_hours = np.random.uniform(0, 10, n)
attendance = np.random.uniform(30, 100, n)
previous_marks = np.random.uniform(20, 100, n)

# Result logic: pass if weighted score is high enough
score = (study_hours / 10) * 0.3 + (attendance / 100) * 0.3 + (previous_marks / 100) * 0.4
result = (score >= 0.5).astype(int)

df = pd.DataFrame({
    "study_hours": np.round(study_hours, 1),
    "attendance": np.round(attendance, 1),
    "previous_marks": np.round(previous_marks, 1),
    "result": result
})

df.to_csv("data.csv", index=False)
print("data.csv generated.")
