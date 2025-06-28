FROM python:3.11-slim-bullseye

WORKDIR /app

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copiar e instalar dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo o projeto
COPY . .

# Expor porta do Jupyter
EXPOSE 8888

# Comando padrão (Jupyter Lab)
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--notebook-dir=/app"]
