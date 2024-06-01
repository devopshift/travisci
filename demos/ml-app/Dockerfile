# Stage 1: Download dependencies
FROM python:3.8-slim as dependencies
WORKDIR /app
COPY requirements.txt .
RUN python -m pip download -d /app/deps -r requirements.txt
RUN echo "Dependencies stage completed"

# Stage 2: Linting and Security Checks
FROM python:3.8-slim as lint-security
WORKDIR /app
RUN mkdir /report
COPY --from=dependencies /app/deps /app/deps
COPY requirements.txt .
RUN pip install --no-index --find-links=/app/deps flake8 bandit --progress-bar off
COPY . .
RUN echo "Running flake8 and bandit"
RUN flake8 main.py --statistics --count -v --show-source --exit-zero > /report/flake8.log  2>&1
RUN echo "Flake8 completed"
RUN bandit -r . -v -s B101 > /report/bandit.log  2>&1
RUN echo "Bandit completed"

# Stage 3: Unit Tests
FROM python:3.8-slim as unittests
WORKDIR /app
RUN mkdir /report
COPY --from=dependencies /app/deps /app/deps
COPY requirements.txt .
RUN pip install --no-index --find-links=/app/deps -r requirements.txt --progress-bar off
COPY . .
RUN echo "Running unit tests"
RUN python -m unittest discover -v tests > /report/unittests.log 2>&1 || true 
RUN echo "Unit tests completed"

# Stage 4: Build Final Image
FROM python:3.8-slim as final
WORKDIR /app
RUN mkdir /report
COPY --from=dependencies /app/deps /app/deps
COPY requirements.txt .
RUN pip install --no-index --find-links=/app/deps -r requirements.txt --progress-bar off
# Copy reports from previous stages
COPY --from=lint-security /report /report
COPY --from=unittests /report /report
COPY . .
CMD ["python", "main.py"]
