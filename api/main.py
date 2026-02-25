from fastapi import FastAPI, HTTPException
from api.schemas import PredictRequest, PredictResponse
from api.model_store import build_store

app = FastAPI()
store = build_store()

@app.on_event("startup")
def startup():
    store.load()

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.get("/readyz")
def readyz():
    return {"ready": store.ready()}

@app.post("/model/reload")
def reload_model():
    store.load()
    return {"reloaded": store.ready()}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not store.ready():
        raise HTTPException(status_code=503, detail="Model not ready")
    features = [
        req.age,
        req.income,
        req.account_balance,
        req.transactions_last_30d,
        req.is_premium,
    ]
    pred, proba = store.predict(features)
    return PredictResponse(prediction=pred, probability=proba)