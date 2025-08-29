"""
Model Organization Script
=========================
This script helps you organize models in a folder and shows how to update your code accordingly.
"""

import os
import shutil
import glob
from pathlib import Path

def organize_models():
    """Move existing model files to models folder and show how to update code"""
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    print("🗂️  MODEL ORGANIZATION GUIDE")
    print("=" * 40)
    
    # Find existing model files
    model_files = glob.glob("*.pkl") + glob.glob("*.joblib") + glob.glob("*model*")
    
    if model_files:
        print(f"\n📁 Found {len(model_files)} model files:")
        for file in model_files:
            print(f"  • {file}")
        
        print(f"\n🚚 Moving files to 'models/' folder...")
        for file in model_files:
            if os.path.exists(file):
                new_path = models_dir / file
                try:
                    shutil.move(file, new_path)
                    print(f"  ✅ Moved: {file} → {new_path}")
                except Exception as e:
                    print(f"  ❌ Failed to move {file}: {e}")
    else:
        print("\n📁 No existing model files found in current directory")
    
    print(f"\n🔧 CODE UPDATE REQUIREMENTS:")
    print("When organizing models in folders, update these patterns:")
    print()
    
    print("❌ OLD CODE:")
    print("joblib.dump(model, 'salary_model.pkl')")
    print("model = joblib.load('salary_model.pkl')")
    print()
    
    print("✅ NEW CODE:")
    print("import os")
    print("os.makedirs('models', exist_ok=True)")
    print("joblib.dump(model, 'models/salary_model.pkl')")
    print("model = joblib.load('models/salary_model.pkl')")
    print()
    
    print("🏗️  RECOMMENDED FOLDER STRUCTURE:")
    print("foai_assignment_outputs/")
    print("├── models/")
    print("│   ├── salary_model_advanced.pkl")
    print("│   ├── salary_model_simple.pkl")
    print("│   └── salary_model_regularized.pkl")
    print("├── data/")
    print("│   └── FOAI-assignment2-1.csv")
    print("├── figures/")
    print("│   ├── figure1_salary_distribution.png")
    print("│   └── ...")
    print("└── notebooks/")
    print("    └── AI.ipynb")
    print()
    
    print("💡 BENEFITS OF ORGANIZING MODELS:")
    print("• Cleaner main directory")
    print("• Easy to find and compare models")
    print("• Better version control")
    print("• Professional project structure")
    print("• Easier to share and deploy")
    
    return models_dir

def create_model_loading_example():
    """Create example code for loading models from folder"""
    
    example_code = '''
# Example: How to load models from the models folder

import joblib
import os
from pathlib import Path

def load_model(model_name):
    """Load a model from the models folder"""
    model_path = Path("models") / f"{model_name}.pkl"
    
    if model_path.exists():
        model = joblib.load(model_path)
        print(f"✅ Loaded model: {model_path}")
        return model
    else:
        print(f"❌ Model not found: {model_path}")
        return None

def list_available_models():
    """List all available models"""
    models_dir = Path("models")
    if models_dir.exists():
        model_files = list(models_dir.glob("*.pkl"))
        print("📋 Available models:")
        for model_file in model_files:
            print(f"  • {model_file.stem}")
        return [f.stem for f in model_files]
    else:
        print("❌ Models folder not found")
        return []

# Usage examples:
if __name__ == "__main__":
    # List available models
    available_models = list_available_models()
    
    # Load specific model
    model = load_model("salary_model_advanced")
    
    # Load with error handling
    try:
        model = joblib.load("models/salary_model_advanced.pkl")
        print("Model loaded successfully!")
    except FileNotFoundError:
        print("Model file not found - check the path")
    except Exception as e:
        print(f"Error loading model: {e}")
'''
    
    with open("model_loading_example.py", "w") as f:
        f.write(example_code)
    
    print("📝 Created 'model_loading_example.py' with usage examples")

if __name__ == "__main__":
    # Organize existing models
    models_dir = organize_models()
    
    # Create example code
    create_model_loading_example()
    
    print(f"\n🎉 ORGANIZATION COMPLETE!")
    print(f"Models are now organized in: {models_dir.absolute()}")
    print(f"Check 'model_loading_example.py' for usage examples")
