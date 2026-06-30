FROM continuumio/miniconda3:23.10.0-1

LABEL maintainer="Utkarsh Patel"
LABEL description="Adenosine Selectivity Model — QSAR platform with conformal prediction"

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libxrender1 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install RDKit via conda (pip install rdkit fails on many Linux systems)
RUN conda install -y -c conda-forge rdkit && conda clean -afy

# Install Python dependencies (layer cached until requirements change)
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Verify critical imports
RUN python -c "from rdkit import Chem; print('RDKit OK')" && \
    python -c "from mapie.regression import CrossConformalRegressor; print('MAPIE OK')"

# Copy application code
COPY . .

RUN adduser --disabled-password --gecos "" appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8501', timeout=5)" || exit 1

ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLE_CORS=false

CMD ["streamlit", "run", "streamlit_app.py"]
