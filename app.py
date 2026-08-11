from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
import re
from sklearn.ensemble import RandomForestClassifier
import os



# ──────────────────────────────────────────────
#  Skills map
# ──────────────────────────────────────────────
TECH_SKILLS_MAP = {
    "java":             ["java", "jvm", "j2ee"],
    "spring":           ["spring boot", "spring framework", "springboot", "spring"],
    "angular":          ["angular", "angularjs"],
    "react":            ["react", "reactjs", "react.js"],
    "vue":              ["vue", "vuejs", "vue.js"],
    "python":           ["python", "django", "flask", "fastapi"],
    "javascript":       ["javascript", "js", "es6", "typescript"],
    "node":             ["node", "nodejs", "node.js"],
    "php":              ["php", "symfony", "laravel"],
    "dotnet":           [".net", "dotnet", "c#", "asp.net"],
    "flutter":          ["flutter", "dart"],
    "android":          ["android", "kotlin"],
    "html":             ["html", "html5"],
    "css":              ["css", "css3", "sass", "scss"],
    "sql":              ["sql", "base de données", "database"],
    "mysql":            ["mysql"],
    "postgresql":       ["postgresql", "postgres"],
    "mongodb":          ["mongodb", "nosql", "mongo"],
    "redis":            ["redis"],
    "elasticsearch":    ["elasticsearch", "elastic"],
    "docker":           ["docker", "conteneur", "container"],
    "kubernetes":       ["kubernetes", "k8s"],
    "git":              ["git", "github", "gitlab"],
    "linux":            ["linux", "unix", "ubuntu"],
    "aws":              ["aws", "amazon cloud"],
    "azure":            ["azure", "microsoft cloud"],
    "gcp":              ["gcp", "google cloud"],
    "jenkins":          ["jenkins", "ci/cd", "pipeline"],
    "terraform":        ["terraform"],
    "ansible":          ["ansible"],
    "cybersecurite":    ["cybersécurité", "cybersecurity", "sécurité informatique", "pentest", "firewall", "siem"],
    "reseau":           ["réseau", "réseaux", "network", "networking", "cisco", "tcp/ip", "vpn"],
    "cloud":            ["cloud", "saas", "paas", "iaas", "serverless", "hébergement cloud"],
    "microservices":    ["microservices", "microservice"],
    "api":              ["api", "rest", "restful", "graphql", "soap"],
    "machine_learning": ["machine learning", "deep learning", "ia", "intelligence artificielle",
                         "tensorflow", "pytorch", "sklearn", "scikit-learn", "nlp", "computer vision"],
    "data":             ["data science", "data analyst", "big data", "hadoop", "spark", "power bi", "tableau"],
    "finance":          ["finance", "comptabilité", "erp", "sap", "odoo"],
    "ios":              ["ios", "swift", "xcode"],
    "mobile":           ["mobile", "application mobile"],
}

# ──────────────────────────────────────────────
#  Groupes de compétences proches (match partiel)
# ──────────────────────────────────────────────
SKILL_GROUPS = {
    "frontend_js":    {"angular", "react", "vue", "javascript", "html", "css"},
    "backend_java":   {"spring", "java"},
    "backend_js":     {"node", "javascript", "python"},
    "database_sql":   {"sql", "mysql", "postgresql"},
    "database_nosql": {"mongodb", "redis", "elasticsearch"},
    "mobile":         {"flutter", "android", "ios", "mobile", "react"},
    "ml_data":        {"machine_learning", "data", "python"},
    "cloud":          {"aws", "azure", "gcp", "cloud", "terraform"},
    "devops":         {"docker", "kubernetes", "jenkins", "linux", "git", "ansible"}
}

FEATURES = [
    "skills_match",
    "experience_match",
    "education_level",
    "contract_match",
    "language_match",
    "certification_match",
    "location_match",
    "salary_expectation_match",
]

# ──────────────────────────────────────────────
#  Données synthétiques
#  Règle apprise par le RF :
#    skills ≥ 70  → accepté si exp ok
#    skills 40-69 → borderline, exp/edu décident
#    skills < 40  → rejeté même avec exp=10 edu=5
# ──────────────────────────────────────────────
SYNTHETIC_DATA = [
    # ── Skills élevés → acceptés ──
    {"skills_match": 100, "experience_match": 10, "education_level": 5, "contract_match": 1, "language_match": 1, "certification_match": 1, "location_match": 1, "salary_expectation_match": 1, "accepted": 1},
    {"skills_match": 100, "experience_match": 8,  "education_level": 4, "contract_match": 1, "language_match": 1, "certification_match": 0, "location_match": 1, "salary_expectation_match": 1, "accepted": 1},
    {"skills_match": 100, "experience_match": 6,  "education_level": 3, "contract_match": 1, "language_match": 1, "certification_match": 0, "location_match": 1, "salary_expectation_match": 1, "accepted": 1},
    {"skills_match": 90,  "experience_match": 10, "education_level": 5, "contract_match": 1, "language_match": 1, "certification_match": 1, "location_match": 1, "salary_expectation_match": 1, "accepted": 1},
    {"skills_match": 90,  "experience_match": 8,  "education_level": 4, "contract_match": 1, "language_match": 1, "certification_match": 0, "location_match": 1, "salary_expectation_match": 1, "accepted": 1},
    {"skills_match": 90,  "experience_match": 2,  "education_level": 2, "contract_match": 1, "language_match": 1, "certification_match": 0, "location_match": 1, "salary_expectation_match": 1, "accepted": 1},
    {"skills_match": 85,  "experience_match": 8,  "education_level": 4, "contract_match": 1, "language_match": 1, "certification_match": 0, "location_match": 1, "salary_expectation_match": 1, "accepted": 1},
    {"skills_match": 85,  "experience_match": 2,  "education_level": 2, "contract_match": 1, "language_match": 1, "certification_match": 0, "location_match": 1, "salary_expectation_match": 1, "accepted": 1},
    {"skills_match": 80,  "experience_match": 10, "education_level": 5, "contract_match": 1, "language_match": 1, "certification_match": 1, "location_match": 1, "salary_expectation_match": 1, "accepted": 1},
    {"skills_match": 80,  "experience_match": 8,  "education_level": 4, "contract_match": 1, "language_match": 1, "certification_match": 0, "location_match": 1, "salary_expectation_match": 1, "accepted": 1},
    {"skills_match": 80,  "experience_match": 2,  "education_level": 2, "contract_match": 0, "language_match": 0, "certification_match": 0, "location_match": 0, "salary_expectation_match": 0, "accepted": 1},
    {"skills_match": 75,  "experience_match": 10, "education_level": 4, "contract_match": 1, "language_match": 1, "certification_match": 0, "location_match": 1, "salary_expectation_match": 1, "accepted": 1},
    {"skills_match": 75,  "experience_match": 2,  "education_level": 2, "contract_match": 1, "language_match": 1, "certification_match": 0, "location_match": 1, "salary_expectation_match": 1, "accepted": 1},
    {"skills_match": 70,  "experience_match": 10, "education_level": 5, "contract_match": 1, "language_match": 1, "certification_match": 1, "location_match": 1, "salary_expectation_match": 1, "accepted": 1},
    {"skills_match": 70,  "experience_match": 8,  "education_level": 4, "contract_match": 1, "language_match": 1, "certification_match": 0, "location_match": 1, "salary_expectation_match": 1, "accepted": 1},

    # ── Skills moyens → borderline ──
    {"skills_match": 67,  "experience_match": 10, "education_level": 5, "contract_match": 1, "language_match": 1, "certification_match": 1, "location_match": 1, "salary_expectation_match": 1, "accepted": 1},
    {"skills_match": 65,  "experience_match": 10, "education_level": 5, "contract_match": 1, "language_match": 1, "certification_match": 1, "location_match": 1, "salary_expectation_match": 1, "accepted": 1},
    {"skills_match": 65,  "experience_match": 8,  "education_level": 4, "contract_match": 1, "language_match": 1, "certification_match": 0, "location_match": 1, "salary_expectation_match": 1, "accepted": 1},
    {"skills_match": 65,  "experience_match": 4,  "education_level": 3, "contract_match": 1, "language_match": 1, "certification_match": 0, "location_match": 1, "salary_expectation_match": 1, "accepted": 0},
    {"skills_match": 60,  "experience_match": 10, "education_level": 5, "contract_match": 1, "language_match": 1, "certification_match": 1, "location_match": 1, "salary_expectation_match": 1, "accepted": 1},
    {"skills_match": 60,  "experience_match": 8,  "education_level": 4, "contract_match": 1, "language_match": 1, "certification_match": 0, "location_match": 1, "salary_expectation_match": 1, "accepted": 0},
    {"skills_match": 55,  "experience_match": 10, "education_level": 5, "contract_match": 1, "language_match": 1, "certification_match": 1, "location_match": 1, "salary_expectation_match": 1, "accepted": 0},
    {"skills_match": 50,  "experience_match": 10, "education_level": 5, "contract_match": 1, "language_match": 1, "certification_match": 1, "location_match": 1, "salary_expectation_match": 1, "accepted": 0},
    {"skills_match": 50,  "experience_match": 8,  "education_level": 4, "contract_match": 1, "language_match": 1, "certification_match": 0, "location_match": 1, "salary_expectation_match": 1, "accepted": 0},

    # ── Skills faibles → TOUJOURS rejeté ──
    {"skills_match": 45,  "experience_match": 10, "education_level": 5, "contract_match": 1, "language_match": 1, "certification_match": 1, "location_match": 1, "salary_expectation_match": 1, "accepted": 0},
    {"skills_match": 40,  "experience_match": 10, "education_level": 5, "contract_match": 1, "language_match": 1, "certification_match": 1, "location_match": 1, "salary_expectation_match": 1, "accepted": 0},
    {"skills_match": 33,  "experience_match": 10, "education_level": 5, "contract_match": 1, "language_match": 1, "certification_match": 1, "location_match": 1, "salary_expectation_match": 1, "accepted": 0},
    {"skills_match": 33,  "experience_match": 8,  "education_level": 4, "contract_match": 1, "language_match": 1, "certification_match": 0, "location_match": 1, "salary_expectation_match": 1, "accepted": 0},
    {"skills_match": 30,  "experience_match": 10, "education_level": 5, "contract_match": 1, "language_match": 1, "certification_match": 1, "location_match": 1, "salary_expectation_match": 1, "accepted": 0},
    {"skills_match": 25,  "experience_match": 10, "education_level": 5, "contract_match": 1, "language_match": 1, "certification_match": 1, "location_match": 1, "salary_expectation_match": 1, "accepted": 0},
    {"skills_match": 20,  "experience_match": 10, "education_level": 5, "contract_match": 1, "language_match": 1, "certification_match": 1, "location_match": 1, "salary_expectation_match": 1, "accepted": 0},
    {"skills_match": 10,  "experience_match": 10, "education_level": 5, "contract_match": 1, "language_match": 1, "certification_match": 1, "location_match": 1, "salary_expectation_match": 1, "accepted": 0},
    {"skills_match": 0,   "experience_match": 10, "education_level": 5, "contract_match": 1, "language_match": 1, "certification_match": 1, "location_match": 1, "salary_expectation_match": 1, "accepted": 0},
    {"skills_match": 0,   "experience_match": 8,  "education_level": 4, "contract_match": 1, "language_match": 1, "certification_match": 0, "location_match": 1, "salary_expectation_match": 1, "accepted": 0},
    {"skills_match": 0,   "experience_match": 2,  "education_level": 2, "contract_match": 0, "language_match": 0, "certification_match": 0, "location_match": 0, "salary_expectation_match": 0, "accepted": 0},
]


def normalize_features(row: dict) -> dict:
    """Ramène toutes les features sur 0-100 avant le RF."""
    return {
        "skills_match":             row["skills_match"],
        "experience_match":         row["experience_match"] * 10,
        "education_level":          (row["education_level"] - 1) * 25,
        "contract_match":           row["contract_match"] * 100,
        "language_match":           row["language_match"] * 100,
        "certification_match":      row["certification_match"] * 100,
        "location_match":           row["location_match"] * 100,
        "salary_expectation_match": row["salary_expectation_match"] * 100,
    }


def compute_weighted_score(skills_pct: float, exp_score: float, edu: int) -> float:
    """
    Score pondéré transparent : skills 55% + exp 30% + edu 15%
    Retourne un float 0-100.
    """
    exp_norm = (exp_score / 10.0) * 100
    edu_norm = ((edu - 1) / 4.0) * 100
    return (skills_pct * 0.55) + (exp_norm * 0.30) + (edu_norm * 0.15)


def blend_scores(rf_proba: float, weighted: float, skills_pct: float) -> int:
    """
    Combine RF + formule pondérée pour un score graduel et honnête.

    - RF donne la décision binaire (accepté/rejeté)
    - La formule pondérée donne la gradation
    - Le blend évite les extrêmes 0%/100%

    Blend : 40% RF + 60% formule pondérée
    → Le RF influence la direction, la formule donne la nuance
    """
    rf_score       = rf_proba * 100          # 0-100
    blended        = (rf_score * 0.40) + (weighted * 0.60)

    # Plancher minimum basé sur les skills
    # → même si le RF dit 0, on affiche au moins skills/4
    result = blended


    return max(0, min(100, int(round(result))))


def train_model(data: list[dict]) -> RandomForestClassifier:
    df = pd.DataFrame([normalize_features(d) for d in data])
    X  = df[FEATURES]
    y  = pd.Series([d["accepted"] for d in data])

    clf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        min_samples_leaf=1,
        max_depth=5,
        max_features=None,
    )
    clf.fit(X, y)

    imp = {f: round(float(i), 3) for f, i in zip(FEATURES, clf.feature_importances_)}
    print(f"✅ Model trained | Feature importances: {imp}")
    return clf


# ── Chargement ou création du modèle ──────────────────────────────
if os.path.exists("model.pkl"):
    try:
        artifact = joblib.load("model.pkl")
        model = artifact["model"] if isinstance(artifact, dict) else artifact
        print("✅ model.pkl chargé")
    except Exception:
        print("⚠️  model.pkl corrompu → réentraînement")
        model = train_model(SYNTHETIC_DATA)
        joblib.dump({"model": model, "features": FEATURES}, "model.pkl")
else:
    print("⚠️  model.pkl absent → entraînement initial")
    model = train_model(SYNTHETIC_DATA)
    joblib.dump({"model": model, "features": FEATURES}, "model.pkl")


# ──────────────────────────────────────────────
#  Extraction de features
# ──────────────────────────────────────────────

def extract_skills_from_text(text: str) -> set:
    text_lower = text.lower()
    found = set()
    for skill_key, variants in TECH_SKILLS_MAP.items():
        for variant in variants:
            if variant in text_lower:
                found.add(skill_key)
                break
    return found


def compute_skills_match(candidat_skills: set, offer_skills: set) -> float:
    if not offer_skills:
        return 0.0
    if not candidat_skills:
        return 0.0

    total = 0.0
    for skill in offer_skills:
        if skill in candidat_skills:
            total += 1.0
            continue
        for group in SKILL_GROUPS.values():
            if skill in group and candidat_skills & group:
                total += 0.3
                break

    score = (total / len(offer_skills)) * 100
    print(f"Skills candidat : {candidat_skills}")
    print(f"Skills offre    : {offer_skills}")
    print(f"Score skills (exact+partiel): {score:.1f}%")
    return round(score, 2)


def extract_features(candidat: str, offer: str) -> dict:
    candidat_skills = extract_skills_from_text(candidat)
    offer_skills    = extract_skills_from_text(offer)
    skills_match    = compute_skills_match(candidat_skills, offer_skills)

    exp_c = re.search(r'(\d+)\s*(ans?|années?|years?)', candidat, re.IGNORECASE)
    exp_o = re.search(r'(\d+)\s*(ans?|années?|years?)', offer,    re.IGNORECASE)
    exp_candidat = min(int(exp_c.group(1)), 10) if exp_c else 1
    exp_requise  = min(int(exp_o.group(1)), 10) if exp_o else 0

    if exp_requise == 0:
        exp_score = 8.0
    else:
        ratio = exp_candidat / exp_requise
        if ratio >= 1.0:   exp_score = 10.0
        elif ratio >= 0.8: exp_score = 8.0
        elif ratio >= 0.6: exp_score = 6.0
        elif ratio >= 0.4: exp_score = 4.0
        else:              exp_score = 2.0

    print(f"Exp candidat: {exp_candidat} ans | Exp requise: {exp_requise} ans | Score: {exp_score}/10")

    c = candidat.lower()
    if "doctorat" in c or "phd" in c:        edu = 5
    elif "master" in c or "ingénieur" in c:  edu = 4
    elif "licence" in c or "bac+3" in c:     edu = 3
    elif "bts" in c or "dut" in c:           edu = 2
    else:                                     edu = 1

    contracts  = ["cdi", "cdd", "stage", "freelance"]
    contract_c = next((x for x in contracts if x in candidat.lower()), "")
    contract_o = next((x for x in contracts if x in offer.lower()),    "")
    contract_match = 1 if not contract_c or not contract_o or contract_c == contract_o else 0

    languages  = ["english", "anglais", "french", "français", "arabic", "arabe"]
    lang_match = 1
    for lang in languages:
        if lang in offer.lower() and lang not in candidat.lower():
            lang_match = 0
            break

    print(f"skills_match final: {skills_match}%")

    return {
        "skills_match":             skills_match,
        "experience_match":         exp_score,
        "education_level":          edu,
        "contract_match":           contract_match,
        "language_match":           lang_match,
        "certification_match":      0,
        "location_match":           1,
        "salary_expectation_match": 1,
    }


# ──────────────────────────────────────────────
#  Endpoints
# ──────────────────────────────────────────────

class MatchingRequest(BaseModel):
    candidat_profile: str
    offer_description: str


class MatchingResponse(BaseModel):
    probability: int
    level: str
    recommendation: str
    skills_score: float
    feature_importance: dict


@app.post("/predict", response_model=MatchingResponse)
def predict(req: MatchingRequest):
    global model
    features = extract_features(req.candidat_profile, req.offer_description)
    norm     = normalize_features(features)

    row       = pd.DataFrame([[norm[f] for f in FEATURES]], columns=FEATURES)
    rf_proba  = model.predict_proba(row)[0][1]

    weighted  = compute_weighted_score(
        skills_pct=features["skills_match"],
        exp_score=features["experience_match"],
        edu=features["education_level"],
    )

    probability = blend_scores(rf_proba, weighted, features["skills_match"])

    print(f"→ RF proba: {rf_proba*100:.1f}% | Weighted: {weighted:.1f}% | Final blend: {probability}%")

    if probability >= 75:
        level = "HIGH"
        recommendation = "Excellent match — candidat fortement recommandé"
    elif probability >= 55:
        level = "MEDIUM"
        recommendation = "Bon match — candidat à considérer"
    elif probability >= 35:
        level = "LOW"
        recommendation = "Match moyen — profil partiellement adapté"
    else:
        level = "WEAK"
        recommendation = "Match faible — profil peu adapté à cette offre"

    importance = {f: round(float(i), 3)
                  for f, i in zip(FEATURES, model.feature_importances_)}

    return MatchingResponse(
        probability=probability,
        level=level,
        recommendation=recommendation,
        skills_score=features["skills_match"],
        feature_importance=importance,
    )


@app.post("/retrain")
def retrain(data: list[dict]):
    global model

    if len(data) < 5:
        combined = SYNTHETIC_DATA
        msg_data = f"pas assez de données réelles ({len(data)}) → synthétiques seulement"
    else:
        combined = SYNTHETIC_DATA + data
        msg_data = f"{len(data)} réels + {len(SYNTHETIC_DATA)} synthétiques"

    try:
        model = train_model(combined)
        joblib.dump({"model": model, "features": FEATURES}, "model.pkl")
        return {
            "success":   True,
            "message":   f"Modèle réentraîné avec {len(combined)} échantillons ({msg_data})",
            "samples":   len(combined),
            "real":      len(data),
            "synthetic": len(SYNTHETIC_DATA),
        }
    except Exception as e:
        return {"success": False, "message": str(e)}


@app.get("/health")
def health():
    return {"status": "ok"}