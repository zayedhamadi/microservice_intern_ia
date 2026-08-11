"""Génère un dataset synthétique COHÉRENT avec les poids canoniques
(model.WEIGHTS), utilisé par train.py comme fallback tant qu'il n'y a pas
assez de vraies décisions RH dans MlTrainingData.

Avant, ce fichier avait ses propres poids (0.40/0.20/0.15/...), différents
de ceux de model.py (0.35/0.20/0.10/...) — les deux divergeaient en
silence. Maintenant il n'y a plus qu'une seule source de vérité.
"""
import random

from model import WEIGHTS

random.seed(42)

MAX_EDUCATION_LEVEL = 5
ACCEPTANCE_THRESHOLD = 45  # score (sur 100) à partir duquel on considère "accepté"


def _random_row() -> dict:
    row = {
        "skillsMatch": round(random.randint(0, 100) / 100, 2),
        "experienceMatch": round(random.uniform(0, 1), 2),
        "educationLevel": random.randint(0, MAX_EDUCATION_LEVEL),
        "contractMatch": random.choice([True, False]),
        "languageMatch": random.choice([True, False]),
        "certificationMatch": random.choice([True, False]),
        "locationMatch": random.choice([True, False]),
        "salaryExpectationMatch": random.choice([True, False]),
    }

    score = 0.0
    for key, weight in WEIGHTS.items():
        val = row[key]
        if key == "educationLevel":
            val = val / MAX_EDUCATION_LEVEL
        elif isinstance(val, bool):
            val = 1.0 if val else 0.0
        score += weight * val

    row["accepted"] = 1 if score * 100 >= ACCEPTANCE_THRESHOLD else 0
    return row


SYNTHETIC_DATA = [_random_row() for _ in range(500)]


if __name__ == "__main__":
    import pandas as pd

    df = pd.DataFrame(SYNTHETIC_DATA)
    print(df["accepted"].value_counts())
    df.to_excel("cv_screening_dataset_enhanced.xlsx", index=False)
    print("Dataset généré avec 500 lignes !")