#!/usr/bin/env python3
"""
Environment checker for TypeScript SLM
Verifies that all dependencies and paths work correctly on Mac and Colab
"""

import os
import sys
from pathlib import Path

def check_environment():
    """Check if environment is properly configured"""

    print("="*70)
    print("TypeScript SLM - Environment Check")
    print("="*70)

    # Detect platform
    is_colab = 'COLAB_GPU' in os.environ or '/content/' in os.getcwd()
    is_mac = sys.platform == 'darwin'

    print(f"\n✓ Platform: {'Google Colab' if is_colab else 'macOS' if is_mac else 'Linux/Other'}")
    print(f"✓ Python: {sys.version.split()[0]}")
    print(f"✓ Working directory: {os.getcwd()}")

    # Check PyTorch
    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")

        if torch.cuda.is_available():
            print(f"✓ CUDA: {torch.cuda.get_device_name(0)}")
            print(f"  - VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        elif torch.backends.mps.is_available():
            print(f"✓ MPS: Apple Metal Performance Shaders available")
        else:
            print(f"⚠ No GPU detected - will use CPU (slow)")
    except ImportError as e:
        print(f"✗ PyTorch not installed: {e}")
        return False

    # Check key dependencies
    required_packages = {
        'transformers': 'Transformers',
        'datasets': 'Datasets',
        'peft': 'PEFT',
        'trl': 'TRL',
        'accelerate': 'Accelerate',
    }

    print("\nDependencies:")
    all_ok = True
    for package, name in required_packages.items():
        try:
            mod = __import__(package)
            version = getattr(mod, '__version__', 'unknown')
            print(f"✓ {name}: {version}")
        except ImportError:
            print(f"✗ {name}: NOT INSTALLED")
            all_ok = False

    # Check directory structure
    print("\nDirectory Structure:")
    required_dirs = ['data', 'scripts', 'models']
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"✓ {dir_name}/")
        else:
            print(f"⚠ {dir_name}/ - missing (will be created)")
            dir_path.mkdir(parents=True, exist_ok=True)

    # Check data files
    print("\nData Files:")
    data_dir = Path('data/processed')
    if data_dir.exists():
        data_files = list(data_dir.glob('train*.jsonl'))
        if data_files:
            for f in sorted(data_files):
                size_mb = f.stat().st_size / 1e6
                print(f"✓ {f.name}: {size_mb:.1f} MB")
        else:
            print(f"⚠ No training data found in data/processed/")
    else:
        print(f"⚠ data/processed/ directory not found")

    # Check scripts
    print("\nScripts:")
    script_files = ['training.py', 'filter_dataset.py', 'evaluation.py']
    scripts_dir = Path('scripts')
    for script in script_files:
        script_path = scripts_dir / script
        if script_path.exists():
            print(f"✓ scripts/{script}")
        else:
            print(f"✗ scripts/{script} - missing")
            all_ok = False

    print("\n" + "="*70)
    if all_ok:
        print("✅ Environment check passed! Ready for training.")
    else:
        print("⚠️  Some issues detected. Please fix before training.")
    print("="*70)

    return all_ok


if __name__ == '__main__':
    success = check_environment()
    sys.exit(0 if success else 1)
