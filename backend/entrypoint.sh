#!/bin/bash
set -e

cd /app/backend

echo "🛠 Running Alembic migrations..."
alembic upgrade head
echo "Alembic migrations completed"

echo "Starting app..."
exec uvicorn main:app --host 0.0.0.0 --port 8000
