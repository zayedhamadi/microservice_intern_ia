
WEIGHTS = {
    "skillsMatch": 0.35,
    "experienceMatch": 0.20,
    "educationLevel": 0.10,
    "languageMatch": 0.10,
    "locationMatch": 0.10,
    "contractMatch": 0.05,
    "certificationMatch": 0.05,
    "salaryExpectationMatch": 0.05,
}

# Doit correspondre au niveau max défini dans feature_engineering.NIVEAU_ETUDE_ORDER
MAX_EDUCATION_LEVEL = 5


def rule_based_score(features: dict) -> float:
    """Fallback transparent utilisé quand aucun modèle ML n'est chargé
    (premier lancement, avant le premier /retrain)."""
    score = 0.0
    for key, weight in WEIGHTS.items():
        val = features.get(key, 0)
        if key == "educationLevel":
            val = val / MAX_EDUCATION_LEVEL
        elif isinstance(val, bool):
            val = 1.0 if val else 0.0
        score += weight * val
    return round(score * 100, 2)