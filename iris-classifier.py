from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError
import logging
import time
import json
import joblib # <-- Added for model loading
import numpy as np # <-- Added for array conversion
from pathlib import Path # <-- Added to find model file

# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter

# --- Setup Tracer ---
try:
    trace.set_tracer_provider(TracerProvider())
    tracer = trace.get_tracer(__name__)
    span_processor = BatchSpanProcessor(CloudTraceSpanExporter())
    trace.get_tracer_provider().add_span_processor(span_processor)
    print("OpenTelemetry Tracer initialized.")
except Exception as e:
    print(f"Failed to initialize OpenTelemetry Tracer: {e}")
    tracer = trace.get_tracer("fallback-tracer")

# --- Setup Structured Logging ---
logger = logging.getLogger("iris-ml-service")
logger.setLevel(logging.INFO)

if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter(json.dumps({
        "severity": "%(levelname)s",
        "message": "%(message)s",
        "timestamp": "%(asctime)s"
    }))
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# --- NEW: ML Model Loading ---
# This variable simulate a heavier model's workload.
# **IMPORTANT**: A real Iris model is too fast (<1ms).
# We keep this sleep to make sure the CPU stress test
# works for your autoscaling assignment.
SIMULATED_WORKLOAD_SECONDS = 0.1 

# Find the model file relative to this python file
MODEL_FILE_PATH = Path(__file__).parent / "model.joblib"
model = None # Global variable to hold the loaded model

# --- FastAPI app ---
app = FastAPI(
    title="Iris Classification Pipeline",
    description="A scalable API for Iris classification, built to demonstrate autoscaling."
)

# --- NEW: Real ML Model Function ---
def run_model_inference(features: dict):
    """
    Runs inference using the loaded model and simulates workload.
    """
    global model
    if model is None:
        raise RuntimeError("Model is not loaded.")
    
    # 1. Convert Pydantic features into the 2D array
    # scikit-learn expects (e.g., [[5.1, 3.5, 1.4, 0.2]])
    input_data = [
        features["sepal_length_cm"],
        features["sepal_width_cm"],
        features["petal_length_cm"],
        features["petal_width_cm"]
    ]
    input_array = np.array([input_data])
    
    # 2. Run real model prediction
    prediction = model.predict(input_array)
    
    # 3. **CRITICAL**: Simulate workload for stress test
    # This is what makes autoscaling observable.
    time.sleep(SIMULATED_WORKLOAD_SECONDS) 
    
    # Return the prediction (convert numpy int to standard int)
    return {"predicted_class": int(prediction[0]), "confidence": 0.99} # Confidence is still dummy

# --- Input/Output Schemas ---
class IrisFeatures(BaseModel):
    sepal_length_cm: float = Field(..., example=5.1)
    sepal_width_cm: float = Field(..., example=3.5)
    petal_length_cm: float = Field(..., example=1.4)
    petal_width_cm: float = Field(..., example=0.2)


# --- Probes & State ---
app_state = {"is_ready": False, "is_alive": True}

@app.on_event("startup")
async def startup_event():
    # --- UPDATED: Load the real model ---
    global model
    logger.info(json.dumps({"event": "startup", "status": "loading_model", "path": str(MODEL_FILE_PATH)}))
    
    try:
        model = joblib.load(MODEL_FILE_PATH)
        app_state["is_ready"] = True
        logger.info(json.dumps({"event": "startup", "status": "model_ready_and_loaded"}))
    except Exception as e:
        logger.exception(json.dumps({"event": "startup", "status": "model_load_failed", "error": str(e)}))
        app_state["is_alive"] = False # Mark as not alive if model fails to load

@app.get("/live_check", tags=["Health Probes"])
async def liveness_probe():
    if app_state["is_alive"]:
        return {"status": "alive"}
    return Response(status_code=500)

@app.get("/ready_check", tags=["Health Probes"])
async def readiness_probe():
    if app_state["is_ready"]:
        return {"status": "ready"}
    return Response(status_code=503)


# --- Middleware ---
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = round((time.time() - start_time) * 1000, 2)
    response.headers["X-Process-Time-ms"] = str(duration)
    return response

# --- Exception Handler ---
@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception):
    span = trace.get_current_span()
    trace_id = format(span.get_span_context().trace_id, "032x") if span.is_recording() else "no_trace"
    
    logger.exception(json.dumps({
        "event": "unhandled_exception",
        "trace_id": trace_id,
        "path": str(request.url),
        "error": str(exc)
    }))
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error", "trace_id": trace_id},
    )

# --- Prediction Endpoint ---
@app.post("/predict", tags=["ML Inference"])
async def predict(features: IrisFeatures, request: Request):
    """
    Run inference using the loaded Iris model. This endpoint is
    designed to be stress-tested.
    """
    with tracer.start_as_current_span("model_inference") as span:
        start_time = time.time()
        trace_id = format(span.get_span_context().trace_id, "032x")

        try:
            input_data = features.dict()
            span.set_attributes({
                "ml.model.input": json.dumps(input_data)
            })
            
            # --- UPDATED: Call the real model function ---
            result = run_model_inference(input_data)
            
            latency = round((time.time() - start_time) * 1000, 2)
            
            logger.info(json.dumps({
                "event": "prediction",
                "trace_id": trace_id,
                "input": input_data,
                "result": result,
                "latency_ms": latency,
                "status": "success"
            }))
            
            span.set_attribute("ml.model.output", json.dumps(result))
            return result

        except Exception as e:
            logger.exception(json.dumps({
                "event": "prediction_error",
                "trace_id": trace_id,
                "error": str(e)
            }))
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR))
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # This allows running the app locally for testing
    uvicorn.run(app, host="0.0.0.0", port=8200)