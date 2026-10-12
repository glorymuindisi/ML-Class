import sys
from importlib.metadata import version, PackageNotFoundError

packages = [
    "numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "seaborn",
    "torch", "torchvision", "torchinfo", "thop", "onnx", "onnxruntime",
    "mlflow", "memory-profiler", "psutil", "codecarbon", "fastapi",
    "uvicorn", "pytest", "httpx", "locust", "requests", "pyarrow",
    "joblib", "tqdm",
]

print("Python", sys.version.split()[0])
for name in packages:
    try:
        print(f"{name}=={version(name)}")
    except PackageNotFoundError:
        print(f"{name}: NOT INSTALLED")