# Stage 1: Download dependencies
FROM python:3.8-slim as dependencies
WORKDIR /app
COPY requirements.txt .
RUN python -m pip download -d /app/deps -r requirements.txt

# Stage 2: Linting and Security Checks
FROM python:3.8-slim as lint-security
WORKDIR /app
COPY --from=dependencies /app/deps /app/deps
COPY requirements.txt .
RUN pip install --no-index --find-links=/app/deps flake8 bandit
COPY . .
RUN flake8 main.py --statistics --count || true
RUN bandit -r . || true

# Stage 3: Unit Tests
FROM python:3.8-slim as unittests
WORKDIR /app
COPY --from=dependencies /app/deps /app/deps
COPY requirements.txt .
RUN pip install --no-index --find-links=/app/deps -r requirements.txt
COPY . .
RUN python -m unittest discover -v tests || true

# Stage 4: Build Final Image
FROM python:3.8-slim as final
WORKDIR /app
COPY --from=dependencies /app/deps /app/deps
COPY requirements.txt .
RUN pip install --no-index --find-links=/app/deps -r requirements.txt
COPY . .
CMD ["python", "main.py"]
