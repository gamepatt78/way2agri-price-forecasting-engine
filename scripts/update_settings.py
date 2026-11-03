from pathlib import Path


settings_path = (
    Path(__file__).resolve().parents[1]
    / "backend"
    / "django_agri"
    / "agri_core"
    / "settings.py"
)
content = settings_path.read_text(encoding="utf-8")

if "import os" not in content:
    content = content.replace("from pathlib import Path", "from pathlib import Path\nimport os")

target = """INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
]"""

replacement = """INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'corsheaders',
    'forecasting',
]"""

if target in content:
    content = content.replace(target, replacement)
elif replacement not in content:
    raise RuntimeError("Could not locate INSTALLED_APPS block in settings.py.")

target_mid = """MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',"""

replacement_mid = """MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.security.SecurityMiddleware',"""

if target_mid in content:
    content = content.replace(target_mid, replacement_mid)
elif replacement_mid not in content:
    raise RuntimeError("Could not locate MIDDLEWARE block in settings.py.")

if "CORS_ALLOWED_ORIGINS =" not in content:
    content += """
CORS_ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("DJANGO_CORS_ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
]
"""

if "REST_FRAMEWORK = {" not in content:
    content += """
REST_FRAMEWORK = {
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': 24,
}
"""

settings_path.write_text(content, encoding="utf-8")
print("Updated settings.py successfully!")
