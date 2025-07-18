# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Install curl for downloading uv
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Set the working directory in the container
WORKDIR /app

# Copy project configuration and source code
COPY pyproject.toml poetry.lock* langgraph.json ./
# poetry.lock* is included in case it exists, for uv to potentially use it.
# If only pyproject.toml is present, uv will resolve dependencies from it.

COPY src/ ./src/

# Copy requirements.txt and install dependencies
COPY requirements.txt ./
RUN uv pip install --system -r requirements.txt

# Copy .env.example to .env
# In a production Cloud Run environment, environment variables should be set directly
# in the service configuration rather than relying on .env files for secrets.
# DO NOT COPY .env.example to .env in production Cloud Run environments.
# Environment variables should be set directly in the Cloud Run service configuration.

# Cloud Run sets the PORT environment variable, so we don't need to EXPOSE it here,
# but it's good practice for documenting which port the application intends to use.
# The actual port binding will be handled by the CMD.
EXPOSE 8080

# Define the command to run the application
# We use sh -c to allow for environment variable expansion for $PORT.
# Cloud Run will set the PORT environment variable.
# The --allow-blocking flag was present in the original README command.
# --config langgraph.json explicitly points to the config.
# Add src directory to PYTHONPATH so that the 'agent' module can be found
ENV PYTHONPATH="/app/src:${PYTHONPATH}"

# Call langgraph directly, assuming it's on PATH after system-wide installation.
CMD ["sh", "-c", "langgraph dev --host 0.0.0.0 --port ${PORT:-8080} --config langgraph.json --allow-blocking"]
