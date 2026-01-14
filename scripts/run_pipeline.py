"""
Run ML Pipeline
---------------
Main script to run the complete ML pipeline.
"""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data.ingestion import DataIngestion
from src.data.validation import DataValidation
from src.data.preprocessing import DataPreprocessing
from src.models.train import ModelTrainer
from src.models.evaluate import ModelEvaluator
from src.utils.logger import get_logger
from src.utils.config import Config

logger = get_logger(__name__)


def run_pipeline():
    """Run the complete ML pipeline."""
    
    print("="*60)
    print("       🚀 ML PIPELINE - Starting Execution")
    print("="*60)
    
    # Load config
    config = Config()
    
    # =====================
    # STEP 1: Data Ingestion
    # =====================
    print("\n📥 STEP 1: Data Ingestion")
    print("-" * 40)
    
    ingestion = DataIngestion()
    
    # Use Forest Covtype dataset - a large real-world classification dataset
    # 581,012 samples, 54 features, 7 forest cover type classes
    from sklearn.datasets import fetch_covtype
    import pandas as pd
    
    print("   ⏳ Downloading Forest Covtype dataset (this may take a moment)...")
    covtype = fetch_covtype(as_frame=True)
    data = covtype.frame  # Already includes target column
    
    # Rename target column for consistency
    data = data.rename(columns={'Cover_Type': 'target'})
    
    # For faster iteration during development, you can use a subset:
    # Uncomment the line below to use 50,000 samples instead of full dataset
    # data = data.sample(n=50000, random_state=42).reset_index(drop=True)
    
    # Save raw data
    raw_path = "data/raw/covtype_data.csv"
    os.makedirs("data/raw", exist_ok=True)
    data.to_csv(raw_path, index=False)
    
    logger.info(f"Data loaded: {data.shape}")
    print(f"   ✅ Loaded {len(data):,} samples with {len(data.columns)} columns")
    print(f"   📊 Features: {len(data.columns) - 1}, Target classes: {data['target'].nunique()}")
    
    # =====================
    # STEP 2: Data Validation
    # =====================
    print("\n🔍 STEP 2: Data Validation")
    print("-" * 40)
    
    validator = DataValidation()
    
    # Get first feature column name for validation check
    feature_cols = [col for col in data.columns if col != 'target']
    validation_result = validator.validate_all(
        data,
        required_columns=[feature_cols[0], 'target'],  # Check first feature and target exist
        missing_threshold=0.3
    )
    
    if validation_result['overall_valid']:
        print("   ✅ Validation PASSED")
    else:
        print("   ❌ Validation FAILED")
        print(f"   Details: {validation_result['details']}")
    
    # =====================
    # STEP 3: Data Preprocessing
    # =====================
    print("\n⚙️ STEP 3: Data Preprocessing")
    print("-" * 40)
    
    preprocessor = DataPreprocessing()
    X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline(
        data, 
        target_column='target',
        test_size=config.get('data', 'test_size', default=0.2)
    )
    
    print(f"   ✅ Training samples: {len(X_train)}")
    print(f"   ✅ Testing samples: {len(X_test)}")
    
    # =====================
    # STEP 4: Model Training (ALL 4 MODELS!)
    # =====================
    print("\n🎯 STEP 4: Model Training - Training ALL 4 Models!")
    print("-" * 40)
    
    # Define all models to train
    models_to_train = ['random_forest', 'logistic_regression', 'decision_tree', 'gradient_boosting']
    trained_models = {}
    
    for model_type in models_to_train:
        print(f"\n   📦 Training: {model_type}...")
        
        trainer = ModelTrainer()
        
        try:
            # Different params for different models
            if model_type in ['random_forest', 'gradient_boosting']:
                model = trainer.train(
                    X_train, y_train,
                    model_name=model_type,
                    n_estimators=100
                )
            else:
                model = trainer.train(
                    X_train, y_train,
                    model_name=model_type
                )
            
            print(f"      ✅ {model_type} trained in {trainer.training_info['duration_seconds']:.2f}s")
            
            # Save each model with its name
            model_path = trainer.save_model(f"model_{model_type}.pkl")
            print(f"      💾 Saved to: {model_path}")
            
            trained_models[model_type] = {
                'trainer': trainer,
                'path': model_path,
                'duration': trainer.training_info['duration_seconds']
            }
            
        except Exception as e:
            print(f"      ❌ Failed to train {model_type}: {e}")
    
    # Also save the default model.pkl as random_forest for backwards compatibility
    if 'random_forest' in trained_models:
        import shutil
        shutil.copy(
            trained_models['random_forest']['path'],
            "artifacts/model.pkl"
        )
        print(f"\n   📌 Default model (model.pkl) = random_forest")
    
    print(f"\n   ✅ Successfully trained {len(trained_models)} models!")
    
    # Use random_forest for evaluation (best performance)
    trainer = trained_models['random_forest']['trainer']
    model = trainer.model
    
    # =====================
    # STEP 5: Model Evaluation (Compare ALL Models!)
    # =====================
    print("\n📊 STEP 5: Model Evaluation - Comparing All Models")
    print("-" * 40)
    
    evaluator = ModelEvaluator()
    all_reports = {}
    
    print("\n   +---------------------------+----------+----------+")
    print("   | Model                     | Accuracy | Time (s) |")
    print("   +---------------------------+----------+----------+")
    
    for model_type, model_data in trained_models.items():
        report = evaluator.evaluate(
            model_data['trainer'].model, 
            X_test, y_test, 
            model_name=model_type
        )
        all_reports[model_type] = report
        
        accuracy = report['metrics']['accuracy']
        duration = model_data['duration']
        print(f"   | {model_type:<25} | {accuracy:>7.2%} | {duration:>8.2f} |")
    
    print("   +---------------------------+----------+----------+")
    
    # Find best model
    best_model = max(all_reports.keys(), key=lambda k: all_reports[k]['metrics']['accuracy'])
    print(f"\n   BEST Model: {best_model} ({all_reports[best_model]['metrics']['accuracy']:.2%} accuracy)")
    
    # Save the comparison report
    report = all_reports['random_forest']  # Main report
    
    # =====================
    # STEP 6: Save Model
    # =====================
    print("\n💾 STEP 6: Saving Model")
    print("-" * 40)
    
    # Save as model.pkl for API to load
    model_path = trainer.save_model("model.pkl")
    print(f"   ✅ Model saved to: {model_path}")
    
    # =====================
    # DONE!
    # =====================
    print("\n" + "="*60)
    print("       ✅ ML PIPELINE - Completed Successfully!")
    print("="*60)
    print("\n📌 Next Steps:")
    print("   1. Start the API: python -m src.api.main")
    print("   2. Open docs: http://localhost:8000/docs")
    print("   3. Make predictions via API!")
    print("")
    
    return report


if __name__ == "__main__":
    run_pipeline()
