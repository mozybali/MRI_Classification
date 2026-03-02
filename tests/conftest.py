"""
pytest configuration and shared fixtures.
"""

import sys
import pytest
import numpy as np
from pathlib import Path
from PIL import Image
import tempfile
import shutil

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "model"))

# Mock model ayarlar modülü için gerekli değişkenleri ekle
@pytest.fixture(autouse=True, scope="session")
def setup_model_paths():
    """Model test'leri için gerekli path'leri ayarla."""
    import model.ayarlar as ayarlar
    # Ayarlar zaten doğru import ediliyor, bu fixture sadece güvence için


@pytest.fixture
def test_image():
    """Create a test MRI image (grayscale 256x256)."""
    img_array = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
    return Image.fromarray(img_array, mode='L')


@pytest.fixture
def test_image_path(tmp_path, test_image):
    """Save test image to temporary file."""
    img_path = tmp_path / "test_image.jpg"
    test_image.save(img_path)
    return img_path


@pytest.fixture
def test_dataset_structure(tmp_path):
    """Create a minimal test dataset structure."""
    dataset_path = tmp_path / "test_dataset"
    classes = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]
    
    for class_name in classes:
        class_path = dataset_path / class_name
        class_path.mkdir(parents=True, exist_ok=True)
        
        # Create 3 test images per class
        for i in range(3):
            img = Image.fromarray(
                np.random.randint(0, 256, (256, 256), dtype=np.uint8),
                mode='L'
            )
            img.save(class_path / f"test_{i}.jpg")
    
    return dataset_path


@pytest.fixture
def sample_features_df():
    """Create a sample features DataFrame."""
    import pandas as pd
    
    data = {
        'dosya_adi': [f"img_{i}.png" for i in range(10)],
        'genislik': [256] * 10,
        'yukseklik': [256] * 10,
        'ortalama_yogunluk': np.random.uniform(50, 200, 10),
        'std_yogunluk': np.random.uniform(10, 50, 10),
        'min_yogunluk': np.random.uniform(0, 50, 10),
        'max_yogunluk': np.random.uniform(200, 255, 10),
        'medyan_yogunluk': np.random.uniform(50, 200, 10),
        'sinif': ['NonDemented'] * 3 + ['VeryMildDemented'] * 3 + 
                 ['MildDemented'] * 2 + ['ModerateDemented'] * 2,
        'etiket': [0] * 3 + [1] * 3 + [2] * 2 + [3] * 2,
        'tam_yol': [f"/tmp/img_{i}.png" for i in range(10)],
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary output directory."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture(scope="session")
def test_config():
    """Test configuration settings."""
    return {
        'image_width': 256,
        'image_height': 256,
        'batch_size': 4,
        'random_seed': 42
    }
