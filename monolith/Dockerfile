########################
# 1) Build-time stage:  #
#    install all deps   #
########################
FROM python:3.11-slim AS build
WORKDIR /build

# Use Poetry for reproducible installs; switch to requirements.txt if you prefer
RUN pip install --no-cache-dir poetry==1.8.3

# Copy lock files first to leverage Docker layer caching
COPY app/pyproject.toml app/poetry.lock ./

# Install only runtime deps (no dev/test extras)
RUN poetry install --only main --no-interaction --no-ansi

########################
# 2) Runtime stage      #
########################
FROM python:3.11-slim

# Put Poetry’s virtual-env on $PATH
ENV PATH="/root/.local/bin:$PATH" \
    PYTHONUNBUFFERED=1

WORKDIR /app

########################
# Copy built site-pkgs #
########################
COPY --from=build /root/.local /root/.local

###################################
#  Bake the model into the image  #
###################################
# ./model contains a single file, e.g. msg_cls.pt or MLflow directory
COPY model/ ./model

#################
# Copy the code #
#################
COPY app/ ./app

########################################
#   Start the FastAPI server via Uvicorn
########################################
CMD ["uvicorn", "app.main:app",
     "--host", "0.0.0.0", "--port", "8000"]
