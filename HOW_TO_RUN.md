# How to Run the Project

These instructions are for Windows PowerShell. Run commands from the repository root unless a step says otherwise.

## 1. Open the project

```powershell
cd "C:\path\to\way2agri-price-forecasting-engine"
```

In VS Code, open this folder, not just the notebook file. The dataset is in `data\price_data.csv`.

## 2. Create and prepare the Python environment

Use Python 3.13 for this project. Python 3.15 may not have prebuilt wheels for all dependencies. Create a fresh environment if `.venv` was created with a different Python version:

```powershell
py -3.13 -m venv .venv313
.\.venv313\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install jupyter ipykernel numpy pandas matplotlib seaborn statsmodels scikit-learn xgboost joblib Django djangorestframework django-cors-headers fastapi "uvicorn[standard]"
```

Python 3.13 is available through the Windows Python launcher (`py -3.13`). Keep using this new environment for the project instead of activating the old Python 3.15 `.venv`.

If PowerShell blocks activation, allow it for the current terminal only, then activate:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv313\Scripts\Activate.ps1
```

## 3. Run the forecasting notebook

Start Jupyter from the repository root so the notebook can find the data file:

```powershell
python -m jupyter notebook notebooks\AgriPrice_Forecasting_Way2Agri.ipynb
```

In VS Code, you can instead open `notebooks\AgriPrice_Forecasting_Way2Agri.ipynb`, select the `.venv313` Python kernel, and choose **Run All**.

The notebook loads `data\price_data.csv`, trains and evaluates the forecasting models, and displays the forecast and charts. Running all cells recomputes the outputs.

## 4. Run the Django API (optional)

Open a second terminal, activate `.venv313`, and run:

```powershell
$repo = "C:\path\to\way2agri-price-forecasting-engine"
Set-Location $repo
.\.venv313\Scripts\Activate.ps1
cd backend\django_agri
python manage.py migrate
python manage.py runserver
```

The API is available at `http://localhost:8000/api/`; the admin page is at `http://localhost:8000/admin/`.

## 5. Run the FastAPI service (optional)

Open another terminal and run:

```powershell
$repo = "C:\path\to\way2agri-price-forecasting-engine"
Set-Location $repo
.\.venv313\Scripts\Activate.ps1
cd fastapi_service
python -m uvicorn main:app --reload --port 8001
```

Open `http://localhost:8001/docs` for the interactive API documentation. The health endpoint is `http://localhost:8001/health`.

The repository includes model artifacts under `ml_pipeline\artifacts`. If they are missing and you want to regenerate them, run the training script from the repository root:

```powershell
python ml_pipeline\train_and_forecast.py
```

## Troubleshooting

- **Dataset not found:** Open Jupyter from the repository root, or open the repository root as the VS Code workspace.
- **Module not found:** Confirm the `.venv313` kernel is selected in the notebook and install the missing package in that environment.
- **Port already in use:** Stop the other process using port 8000 or 8001, or choose another port in the server command.
