from pydantic import BaseModel, ConfigDict, Field


class ProductTexts(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str | None = ""
    description: str | None = ""
    model: str | None = ""
    type_prefix: str | None = ""
    vendor: str | None = ""


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    url: str = ""
    texts: ProductTexts
    image_url: str = ""


class PredictResponse(BaseModel):
    category_ind: int = Field(ge=0)


class HealthResponse(BaseModel):
    status: str
    model_ready: bool
    redis_ready: bool
