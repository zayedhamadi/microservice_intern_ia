import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

FEATURES = [
    "skills_match", "experience_match", "education_level",
    "contract_match", "language_match", "certification_match",
    "location_match", "salary_expectation_match"
]

# ✅ experience_match = score relatif (exp_candidat/exp_requise * 10)
# ex: 3 ans requis 5 → 3/5*10 = 6.0
# ex: 5 ans requis 5 → 5/5*10 = 10.0
# ex: 8 ans requis 5 → 10.0 (capped)

synthetic = [
    # ── Parfaits ──
    {"skills_match": 95, "experience_match": 10, "education_level": 5, "contract_match": 1, "language_match": 1, "certification_match": 1, "location_match": 1, "salary_expectation_match": 1, "accepted": 1},
    {"skills_match": 90, "experience_match": 10, "education_level": 4, "contract_match": 1, "language_match": 1, "certification_match": 1, "location_match": 1, "salary_expectation_match": 1, "accepted": 1},
    {"skills_match": 85, "experience_match": 10, "education_level": 4, "contract_match": 1, "language_match": 1, "certification_match": 0, "location_match": 1, "salary_expectation_match": 1, "accepted": 1},
    {"skills_match": 80, "experience_match": 10, "education_level": 4, "contract_match": 1, "language_match": 1, "certification_match": 0, "location_match": 1, "salary_expectation_match": 1, "accepted": 1},

    # ── Bons ──
    {"skills_match": 75, "experience_match": 10, "education_level": 3, "contract_match": 1, "language_match": 1, "certification_match": 0, "location_match": 1, "salary_expectation_match": 1, "accepted": 1},
    {"skills_match": 70, "experience_match": 8,  "education_level": 4, "contract_match": 1, "language_match": 1, "certification_match": 0, "location_match": 1, "salary_expectation_match": 1, "accepted": 1},
    {"skills_match": 65, "experience_match": 8,  "education_level": 4, "contract_match": 1, "language_match": 1, "certification_match": 0, "location_match": 1, "salary_expectation_match": 1, "accepted": 1},

    # ── Cas Aziz (69% skills, 6/10 exp) → BON mais pas Excellent ──
    # ✅ accepted=1 mais score sera ~60-70%
    {"skills_match": 69, "experience_match": 6, "education_level": 4, "contract_match": 1, "language_match": 1, "certification_match": 0, "location_match": 1, "salary_expectation_match": 1, "accepted": 1},
    {"skills_match": 70, "experience_match": 6, "education_level": 4, "contract_match": 1, "language_match": 1, "certification_match": 0, "location_match": 1, "salary_expectation_match": 1, "accepted": 1},
    {"skills_match": 68, "experience_match": 6, "education_level": 4, "contract_match": 1, "language_match": 0, "certification_match": 0, "location_match": 1, "salary_expectation_match": 1, "accepted": 1},
    {"skills_match": 65, "experience_match": 6, "education_level": 3, "contract_match": 1, "language_match": 1, "certification_match": 0, "location_match": 1, "salary_expectation_match": 1, "accepted": 1},

    # ── Limite basse — skills ok mais exp vraiment insuffisante ──
    {"skills_match": 69, "experience_match": 4, "education_level": 3, "contract_match": 1, "language_match": 1, "certification_match": 0, "location_match": 1, "salary_expectation_match": 1, "accepted": 0},
    {"skills_match": 65, "experience_match": 3, "education_level": 3, "contract_match": 0, "language_match": 1, "certification_match": 0, "location_match": 1, "salary_expectation_match": 0, "accepted": 0},

    # ── Moyens ──
    {"skills_match": 55, "experience_match": 6, "education_level": 3, "contract_match": 1, "language_match": 0, "certification_match": 0, "location_match": 1, "salary_expectation_match": 1, "accepted": 0},
    {"skills_match": 50, "experience_match": 4, "education_level": 2, "contract_match": 0, "language_match": 1, "certification_match": 0, "location_match": 1, "salary_expectation_match": 0, "accepted": 0},
    {"skills_match": 45, "experience_match": 3, "education_level": 2, "contract_match": 1, "language_match": 0, "certification_match": 0, "location_match": 1, "salary_expectation_match": 1, "accepted": 0},

    # ── Faibles ──
    {"skills_match": 40, "experience_match": 2, "education_level": 2, "contract_match": 1, "language_match": 0, "certification_match": 0, "location_match": 1, "salary_expectation_match": 1, "accepted": 0},
    {"skills_match": 30, "experience_match": 2, "education_level": 1, "contract_match": 0, "language_match": 1, "certification_match": 0, "location_match": 0, "salary_expectation_match": 0, "accepted": 0},
    {"skills_match": 20, "experience_match": 1, "education_level": 1, "contract_match": 0, "language_match": 0, "certification_match": 0, "location_match": 0, "salary_expectation_match": 0, "accepted": 0},
    {"skills_match": 10, "experience_match": 0, "education_level": 1, "contract_match": 0, "language_match": 0, "certification_match": 0, "location_match": 0, "salary_expectation_match": 0, "accepted": 0},
    {"skills_match": 0,  "experience_match": 0, "education_level": 1, "contract_match": 0, "language_match": 0, "certification_match": 0, "location_match": 0, "salary_expectation_match": 0, "accepted": 0},
]

df = pd.DataFrame(synthetic)
X  = df[FEATURES]
y  = df["accepted"]

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    min_samples_leaf=2,
    max_depth=5,
)
model.fit(X, y)

joblib.dump({"model": model, "features": FEATURES}, "model.pkl")
print("✅ model.pkl regenerated successfully")

# ── Test rapide ──
import pandas as pd
test_cases = [
    {"label": "Aziz vs Cloud (69% skills, 6/10 exp)", "skills_match": 69, "experience_match": 6.0, "education_level": 4, "contract_match": 1, "language_match": 1, "certification_match": 0, "location_match": 1, "salary_expectation_match": 1},
    {"label": "Parfait (90% skills, 10/10 exp)",       "skills_match": 90, "experience_match": 10,  "education_level": 4, "contract_match": 1, "language_match": 1, "certification_match": 1, "location_match": 1, "salary_expectation_match": 1},
    {"label": "Faible (30% skills, 2/10 exp)",         "skills_match": 30, "experience_match": 2,   "education_level": 2, "contract_match": 0, "language_match": 0, "certification_match": 0, "location_match": 0, "salary_expectation_match": 0},
]

for tc in test_cases:
    label = tc.pop("label")
    row   = pd.DataFrame([tc], columns=FEATURES)
    proba = model.predict_proba(row)[0][1]
    print(f"{label} → {int(proba * 100)}%")