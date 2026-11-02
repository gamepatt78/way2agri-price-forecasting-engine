with open(r'd:\Agri\backend\django_agri\agri_core\settings.py', 'r', encoding='utf-8') as f:
    content = f.read()

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

content = content.replace(target, replacement)

target_mid = """MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',"""

replacement_mid = """MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.security.SecurityMiddleware',"""

content = content.replace(target_mid, replacement_mid)

content += """

CORS_ALLOW_ALL_ORIGINS = True

REST_FRAMEWORK = {
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': 24,
}
"""

with open(r'd:\Agri\backend\django_agri\agri_core\settings.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated settings.py successfully!")
