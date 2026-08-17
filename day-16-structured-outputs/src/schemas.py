from pydantic import BaseModel, Field
from typing import Optional

class CustomerInfo(BaseModel):
    name: str = Field(..., description="Full name of customer")
    company: str = Field(..., description="Customer company name")
    budget: float = Field(..., description="Estimated numerical budget in USD")
    intent: str = Field(..., description="Customer intent or tier interest")
