from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    age: float = Field(..., ge=0)
    income: float = Field(..., ge=0)
    account_balance: float = Field(..., ge=0)
    transactions_last_30d: float = Field(..., ge=0)
    is_premium: float = Field(..., ge=0, le=1)

class PredictResponse(BaseModel):
    prediction: int
    probability: float