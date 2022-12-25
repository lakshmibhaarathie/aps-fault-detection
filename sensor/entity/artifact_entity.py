from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    feature_store_path: str
    train_file_path: str
    test_file_path: str


@dataclass
class DataValidationArtifact:
    report_file_path: str


@dataclass
class DataTransformationArtifact:
    transformed_train_path: str
    transformed_test_path: str
    transformer_object_path: str
    target_encoder_path: str


@dataclass
class ModelTrainerArtifact:
    trained_model_path: str
    f1_train_score: float
    f1_test_score: float


@dataclass
class ModelEvaluationArtifact:
    is_model_accepted:bool
    improved_accuracy:float


@dataclass
class ModelPusherArtifact:
    pusher_model_dir:str
    model_registry_dir:str