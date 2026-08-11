# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib, os

from feature_engineering import build_feature_vector, to_vector, FEATURE_ORDER
from model import rule_based_score
from schemas import CandidatDto, PosteDto

app = FastAPI(title="ML Ranking Service")

MODEL_PATH = "models/ranking_model.pkl"

model = None
if os.path.exists(MODEL_PATH):
    artifact = joblib.load(MODEL_PATH)
    if artifact["features"] != FEATURE_ORDER:
        raise RuntimeError(
            "Le modèle a été entraîné avec un ordre de features différent — relance train.py"
        )
    model = artifact["model"]
else:
    print("⚠️  Aucun modèle entraîné trouvé — mode rule_based_score en attendant le premier /retrain")


def _predict_score(features: dict) -> float:
    """Un seul point de calcul du score — évite de dupliquer la logique
    ml vs rule_based entre /score et /score/batch."""
    if model is not None:
        vector = to_vector(features)  # ordre garanti + bool → float
        proba = model.predict_proba([vector])[0][1]
        return round(proba * 100, 2)
    return rule_based_score(features)


class ScoreRequest(BaseModel):
    candidat: CandidatDto
    poste: PosteDto


class CandidatAvecId(CandidatDto):
    keycloakId: str


class BatchScoreRequest(BaseModel):
    candidats: List[CandidatAvecId]
    poste: PosteDto


@app.post("/score")
def score(req: ScoreRequest):
    features = build_feature_vector(req.candidat, req.poste)
    return {
        "score": _predict_score(features),
        "mode": "ml" if model is not None else "rule_based",
        "details": features,
    }


@app.post("/score/batch")
def score_batch(req: BatchScoreRequest):
    """Utilisé par le RH pour classer toutes les candidatures d'un poste."""
    results = [
        {
            "candidatKeycloakId": c.keycloakId,
            "score": _predict_score(build_feature_vector(c, req.poste)),
        }
        for c in req.candidats
    ]
    return sorted(results, key=lambda x: x["score"], reverse=True)


@app.get("/health")
def health():
    return {"status": "ok", "mode": "ml" if model is not None else "rule_based"}