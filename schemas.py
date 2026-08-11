from typing import List, Optional
from pydantic import BaseModel, Field


class CandidatDto(BaseModel):
    """Profil candidat assemblé côté Java à partir de CvParsedData +
    Application (voir service.recrutement.Entity)."""

    competences: List[str] = Field(default_factory=list)
    langues: List[str] = Field(default_factory=list)
    anneesExperience: int = 0
    niveauEtude: Optional[str] = None          # valeur de l'enum NiveauEtude (à vérifier)
    typeContratSouhaite: Optional[str] = None  # valeur de l'enum TypeContrat, si dispo
    lieu: Optional[str] = None
    salaireAttendu: Optional[float] = None
    certifications: List[str] = Field(default_factory=list)


class PosteDto(BaseModel):
    """Reflète service.recrutement.Entity.PosteRecrutement."""

    competencesRequises: List[str] = Field(default_factory=list)
    languesRequises: List[str] = Field(default_factory=list)
    anneesExperienceMin: int = 0
    niveauEtudeRequis: Optional[str] = None
    typeContrat: Optional[str] = None   # valeur de l'enum TypeContrat
    workType: Optional[str] = None      # valeur de l'enum WorkType (SUR_SITE/HYBRIDE/DISTANCE)
    lieu: Optional[str] = None
    salaire: Optional[float] = None


class ScoreRequest(BaseModel):
    candidat: CandidatDto
    poste: PosteDto
    candidatKeycloakId: Optional[str] = None  # utile pour /score/batch


class MlTrainingRecord(BaseModel):
    """Reflète service.recrutement.Entity.MlTrainingData — une candidature déjà
    tranchée par le RH (accepté/rejeté), utilisée pour le ré-entraînement réel."""

    skillsMatch: float
    experienceMatch: float
    educationLevel: int
    contractMatch: bool
    languageMatch: bool
    certificationMatch: bool
    locationMatch: bool
    salaryExpectationMatch: bool
    accepted: bool