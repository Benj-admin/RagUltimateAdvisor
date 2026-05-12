#!/bin/bash

# Exit on error
set -e

echo "Starting RAG with Apptainer"

# Load .env variables at the start so all services can use them
set -a
source .env
set +a

# 1. Create data directories for persistence
PG_DATA_DIR="pgdata_${APP_EMBEDDING_MODEL:-default}"
mkdir -p "$PG_DATA_DIR" ollama_data

if [ ! -f "pg-advisor.sif" ]; then
    echo "Building custom PostgreSQL image to bypass Docker entrypoint..."
    cat <<EOF > PgAdvisor.def
Bootstrap: docker
From: pgvector/pgvector:pg17

%startscript
    exec /usr/lib/postgresql/17/bin/postgres -D /var/lib/postgresql/data -p ${APP_PG_PORT:-5432} -c unix_socket_directories=/var/lib/postgresql/data -c listen_addresses='*' > /var/lib/postgresql/data/postgres.log 2>&1

%runscript
    exec /usr/lib/postgresql/17/bin/postgres "\$@"
EOF
    apptainer build pg-advisor.sif PgAdvisor.def
fi

# 2. Start PostgreSQL / pgvector Instance
echo "1. Setting up PostgreSQL"
if [ ! -s "$PG_DATA_DIR/PG_VERSION" ]; then
    echo "Initializing new database cluster"
    apptainer exec \
        --bind "$PG_DATA_DIR:/var/lib/postgresql/data" \
        pg-advisor.sif \
        /usr/lib/postgresql/17/bin/initdb -D /var/lib/postgresql/data --auth=trust
fi

echo "2. Starting PostgreSQL instance"
if ! apptainer instance list | grep -q "pg-advisor"; then
    apptainer instance start \
        --bind "$PG_DATA_DIR:/var/lib/postgresql/data" \
        pg-advisor.sif pg-advisor
        
    echo "   Waiting for PostgreSQL to start..."
    for i in {1..10}; do
        apptainer exec instance://pg-advisor /usr/lib/postgresql/17/bin/pg_isready -h /var/lib/postgresql/data -p "${APP_PG_PORT:-5432}" -U postgres >/dev/null 2>&1 && break
        sleep 1
    done

    apptainer exec instance://pg-advisor /usr/lib/postgresql/17/bin/pg_isready -h /var/lib/postgresql/data -p "${APP_PG_PORT:-5432}" -U postgres >/dev/null 2>&1 || { echo "PostgreSQL failed to start. Logs:"; cat "$PG_DATA_DIR/postgres.log"; exit 1; }
    
    if [ ! -f "$PG_DATA_DIR/.db_initialized" ]; then
        echo "   Creating database and user from .env..."
        apptainer exec instance://pg-advisor /usr/lib/postgresql/17/bin/psql -h /var/lib/postgresql/data -p ${APP_PG_PORT:-5432} -d postgres -c "CREATE USER $APP_PG_USER WITH PASSWORD '$APP_PG_PASSWORD';" || true
        apptainer exec instance://pg-advisor /usr/lib/postgresql/17/bin/psql -h /var/lib/postgresql/data -p ${APP_PG_PORT:-5432} -d postgres -c "CREATE DATABASE $APP_PG_DATABASE OWNER $APP_PG_USER;" || true
        apptainer exec instance://pg-advisor /usr/lib/postgresql/17/bin/psql -h /var/lib/postgresql/data -p ${APP_PG_PORT:-5432} -d $APP_PG_DATABASE -c "CREATE EXTENSION IF NOT EXISTS vector;" || true
        touch "$PG_DATA_DIR/.db_initialized"
    fi
else
    echo "   Instance 'pg-advisor' is already running."
fi

# 3. Start Ollama Instance
echo "3. Starting Ollama instance"
if [ ! -f "ollama-advisor.sif" ]; then
    echo "Building custom Ollama image to ensure proper background execution..."
    cat <<EOF > OllamaAdvisor.def
Bootstrap: docker
From: ollama/ollama

%startscript
    export OLLAMA_HOST=0.0.0.0:11434
    exec ollama serve > /root/.ollama/ollama.log 2>&1

%runscript
    exec ollama "\$@"
EOF
    apptainer build ollama-advisor.sif OllamaAdvisor.def
fi

# Note: Add --nv before --bind if you want NVIDIA GPU support (e.g., apptainer instance start --nv ...)
if ! apptainer instance list | grep -q "ollama-advisor"; then
    apptainer instance start \
        --bind ollama_data:/root/.ollama \
        ollama-advisor.sif ollama-advisor
        
    echo "   Waiting for Ollama to start..."
    for i in {1..15}; do
        apptainer exec instance://ollama-advisor ollama list >/dev/null 2>&1 && break
        sleep 1
    done
    
    apptainer exec instance://ollama-advisor ollama list >/dev/null 2>&1 || { echo "Ollama failed to start. Logs:"; cat ollama_data/ollama.log; exit 1; }
    
    echo "   Pulling required AI models (this might take time if not cached)..."
    apptainer exec instance://ollama-advisor ollama pull ${APP_CHAT_MODEL:-gemma3:4b}
    apptainer exec instance://ollama-advisor ollama pull ${APP_EMBEDDING_MODEL:-embeddinggemma}
else
    echo "   Instance 'ollama-advisor' is already running."
fi

# Optional: Wait a few seconds for DB and Ollama to boot
sleep 5

# 4. Build Apptainer image for the app if it doesn't exist
if [ ! -f "ultimate-advisor.sif" ]; then
    echo "4. Building Application Image "
    apptainer build ultimate-advisor.sif UltimateAdvisor.def
fi

# Create a fast wrapper image just to inject the proper background startscript
if [ ! -f "ultimate-advisor-bg.sif" ]; then
    echo "   Creating background execution wrapper for Application..."
    cat <<'EOF' > AppWrapper.def
Bootstrap: localimage
From: ultimate-advisor.sif

%startscript
    HOST_DIR=$PWD
    cd /app && exec .venv/bin/fastapi run src/main.py --host 0.0.0.0 --port 8000 > "$HOST_DIR/app.log" 2>&1
EOF
    apptainer build ultimate-advisor-bg.sif AppWrapper.def
fi

# 5. Start the Application Instance
echo "5. Starting Application"
if ! apptainer instance list | grep -q "app-advisor"; then
    apptainer instance start --writable-tmpfs ultimate-advisor-bg.sif app-advisor
    
    echo "   Waiting for FastAPI to start..."
    sleep 3
else
    echo "   Instance 'app-advisor' is already running."
fi

echo "All services started!"
echo ""
echo "To view running instances: apptainer instance list"
echo "To stop everything: apptainer instance stop --all"
echo "To view app logs: cat app.log"
echo ""
echo "App is accessible at http://localhost:8000"