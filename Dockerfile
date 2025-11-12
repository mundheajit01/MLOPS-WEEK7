# 1. Use official Python base image
FROM python:3.11-slim

# 2. Set working directory
WORKDIR /app

# 3. Copy files
# We copy requirements first to leverage Docker layer caching
COPY requirements.txt .

# 4. Install dependencies
# Using your existing requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of the application code
# --- ADDED: Copy the model.joblib file ---
COPY model.joblib .
COPY iris_pipeline_app.py .

# 6. Expose port
EXPOSE 8200

# 7. Command to run the server
# We point to the new python file: iris_pipeline_app
CMD ["uvicorn", "iris_pipeline_app:app", "--host", "0.0.0.0", "--port", "8200"]