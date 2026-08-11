import unicodedata
from difflib import SequenceMatcher

from model import WEIGHTS
from schemas import CandidatDto, PosteDto

# NiveauEtude : hypothèse de mapping — à corriger avec les vraies valeurs de
# l'enum Java user.service.Entity.Enum.NiveauEtude (non fourni dans le code
# partagé). Vérifie les noms exacts (ex: "BAC_PLUS_2" vs "BAC_2") côté Java.
NIVEAU_ETUDE_ORDER = {
    "BAC": 1,
    "BAC_2": 2,
    "BTS": 2,
    "DUT": 2,
    "LICENCE": 3,
    "MASTER": 4,
    "INGENIEUR": 4,
    "DOCTORAT": 5,
}

FUZZY_MATCH_THRESHOLD = 0.8
FEATURE_ORDER = list(WEIGHTS.keys())


def _normalize(text: str) -> str:
    """minuscule + suppression des accents (pour comparer 'Réseau' et 'reseau')."""
    text = text.strip().lower()
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c))


def jaccard_similarity(list_a: list[str], list_b: list[str]) -> float:
    a = {_normalize(x) for x in list_a}
    b = {_normalize(x) for x in list_b}
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def compute_skills_match(candidat_competences: list[str], poste_requises: list[str]) -> float:
    """Matching exact + flou pour capter les variantes ('Spring Boot' vs
    'SpringBoot' vs 'spring-boot'). Contrairement à l'ancien app.py, on
    compare des listes structurées (déjà extraites côté Java/CvParsedData),
    donc plus de bug de sous-chaîne ('java' matchant 'javascript')."""
    if not poste_requises:
        return 1.0
    if not candidat_competences:
        return 0.0

    candidat_norm = [_normalize(c) for c in candidat_competences]
    matched = 0
    for req in poste_requises:
        req_norm = _normalize(req)
        best = max(
            (SequenceMatcher(None, req_norm, c).ratio() for c in candidat_norm),
            default=0.0,
        )
        if best >= FUZZY_MATCH_THRESHOLD:
            matched += 1
    return round(matched / len(poste_requises), 4)


def compute_experience_match(annees_candidat: int | None, annees_min_requis: int | None) -> float:
    annees_candidat = annees_candidat or 0
    annees_min_requis = annees_min_requis or 0
    if annees_min_requis == 0:
        return 1.0
    ratio = annees_candidat / annees_min_requis
    # cap à 1.5x le minimum requis pour ne pas sur-favoriser la sur-qualification
    return round(min(ratio, 1.5) / 1.5, 4)


def compute_education_level(niveau_etude: str | None) -> int:
    if not niveau_etude:
        return 0
    return NIVEAU_ETUDE_ORDER.get(niveau_etude.upper(), 0)


def compute_language_match(candidat_langues: list[str], poste_langues: list[str]) -> bool:
    if not poste_langues:
        return True
    return jaccard_similarity(candidat_langues, poste_langues) > 0.5


def build_feature_vector(candidat: CandidatDto, poste: PosteDto) -> dict:
    return {
        "skillsMatch": compute_skills_match(candidat.competences, poste.competencesRequises),
        "experienceMatch": compute_experience_match(candidat.anneesExperience, poste.anneesExperienceMin),
        "educationLevel": compute_education_level(candidat.niveauEtude),
        "contractMatch": (
            candidat.typeContratSouhaite is None
            or poste.typeContrat is None
            or candidat.typeContratSouhaite == poste.typeContrat
        ),
        "languageMatch": compute_language_match(candidat.langues, poste.languesRequises),
        "certificationMatch": len(candidat.certifications) > 0,
        "locationMatch": (
            poste.workType == "DISTANCE"
            or candidat.lieu is None
            or poste.lieu is None
            or _normalize(candidat.lieu) == _normalize(poste.lieu)
        ),
        "salaryExpectationMatch": (
            candidat.salaireAttendu is None
            or poste.salaire is None
            or candidat.salaireAttendu <= poste.salaire
        ),
    }


def to_vector(features: dict) -> list[float]:
    """Convertit le dict de features (avec bool) en vecteur numérique dans
    l'ordre attendu par le modèle — utilisé à l'entraînement ET à l'inférence,
    pour être sûr que l'ordre des colonnes ne diverge jamais."""
    row = []
    for key in FEATURE_ORDER:
        val = features.get(key, 0)
        row.append(1.0 if val is True else 0.0 if val is False else float(val))
    return row