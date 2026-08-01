export interface PredictionValue {
  value: string;
  confidence: number;
  breakdown?: Record<string, number>;
}

export interface PredictionResult {
  prediction_id: number;
  image_url: string;
  filename: string;
  predictions: {
    nationality: PredictionValue;
    emotion: PredictionValue;
    age?: PredictionValue | null;
    dress_color?: PredictionValue | null;
  };
  conditional_rule: string;
  display_fields: {
    show_nationality: boolean;
    show_emotion: boolean;
    show_age: boolean;
    show_dress_color: boolean;
  };
}

export interface LayerSpec {
  layer: string;
  shape: string;
  params: number;
}

export interface ModelArchSpec {
  num_classes: number;
  classes: string[];
  architecture?: LayerSpec[];
  total_params: number;
  weight_initialization: string;
}

export interface ModelMetric {
  model_name: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  loss: number;
  epochs_trained: number;
  updated_at?: string;
  confusion_matrix?: number[][];
}

export interface TrainingEpoch {
  id: number;
  model_name: string;
  epoch: number;
  train_acc: number;
  val_acc: number;
  train_loss: number;
  val_loss: number;
  lr: number;
  timestamp?: string;
}

export interface SampleImage {
  name: string;
  path: string;
  expected: string;
}

export interface HistoryRecord {
  id: number;
  filename: string;
  image_path: string;
  nationality: string;
  nationality_confidence: number;
  emotion: string;
  emotion_confidence: number;
  age?: string | null;
  age_confidence?: number | null;
  dress_color?: string | null;
  dress_color_confidence?: number | null;
  conditional_rule: string;
  created_at: string;
}
