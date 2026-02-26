from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from api.model_store import build_store
from api.schemas import PredictRequest, PredictResponse

store = build_store()


@asynccontextmanager
async def lifespan(_: FastAPI):
    store.load()
    yield


app = FastAPI(lifespan=lifespan)


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