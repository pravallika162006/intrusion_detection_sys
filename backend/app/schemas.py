"""
Pydantic schemas for Phase 3 API endpoints.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# Dataset schemas
class DatasetValidationResponse(BaseModel):
    filename: str
    file_format: str
    row_count: int
    column_count: int
    has_labels: bool
    label_column: Optional[str] = None
    has_multiclass_labels: bool
    multiclass_column: Optional[str] = None
    features_present_42: bool
    missing_features_42: List[str]
    features_present_19: bool
    missing_features_19: List[str]
    validation_status: str
    message: str

class DatasetPredictionRequest(BaseModel):
    file_id: str
    feature_mode: str = Field(..., description="42 or 19")
    model_name: str = Field(..., description="e.g., decision_tree, xgboost_dt, ann, knn, logistic_regression, svm")
    prediction_task: str = Field("binary", description="binary or multiclass")

class DatasetPredictionSummary(BaseModel):
    total_records: int
    normal_count: int
    attack_count: int
    attack_percentage: float
    attack_categories: Optional[Dict[str, int]] = None
    confidence_avg: Optional[float] = None

class EvaluationMetrics(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    macro_f1: float
    weighted_f1: float
    confusion_matrix: List[List[int]]
    classification_report: Dict[str, Any]

class DatasetPredictionResponse(BaseModel):
    file_id: str
    feature_mode: str
    model_name: str
    prediction_task: str
    summary: DatasetPredictionSummary
    evaluation: Optional[EvaluationMetrics] = None
    sample_predictions: List[Dict[str, Any]]

# Live Monitoring schemas
class NetworkInterfaceInfo(BaseModel):
    name: str
    description: str
    ip_address: Optional[str] = None
    mac_address: Optional[str] = None
    is_active: bool

class StartCaptureRequest(BaseModel):
    interface_name: str
    test_mode: bool = False
    binary_model: str = "xgboost_dt"
    multiclass_model: str = "ann"
    threshold: float = 0.80

class SetThresholdRequest(BaseModel):
    threshold: float = Field(0.80, ge=0.10, le=0.99, description="Calibrated operating threshold (0.10 - 0.99)")

class AIRecommendationRequest(BaseModel):
    flow_id: str
    src_ip: str
    dst_ip: str
    service: str
    attack_category: str
    confidence: float
    flow_info: Dict[str, Any]

class AIRecommendationResponse(BaseModel):
    threat_summary: str
    explanation: str
    why_flagged: Optional[str] = None
    severity: str
    recommended_actions: List[str]
    investigation_guidance: List[str]
