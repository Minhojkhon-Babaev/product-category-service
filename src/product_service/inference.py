from typing import Protocol

import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException


class Vectorizer(Protocol):
    def transform(self, raw_documents: list[str]): ...


class TritonClassifier:
    def __init__(
        self,
        vectorizer: Vectorizer,
        url: str,
        model_name: str,
        timeout: float = 10.0,
    ) -> None:
        self.vectorizer = vectorizer
        self.model_name = model_name
        self.timeout = timeout
        self.client = httpclient.InferenceServerClient(
            url=url,
            connection_timeout=timeout,
            network_timeout=timeout,
        )

    def is_ready(self) -> bool:
        return self.client.is_server_ready() and self.client.is_model_ready(self.model_name)

    def predict(self, text: str) -> int:
        features = self.vectorizer.transform([text]).astype(np.float32).toarray()
        infer_input = httpclient.InferInput("features", features.shape, "FP32")
        infer_input.set_data_from_numpy(features)
        output = httpclient.InferRequestedOutput("label")
        try:
            result = self.client.infer(
                model_name=self.model_name,
                inputs=[infer_input],
                outputs=[output],
            )
        except InferenceServerException as error:
            raise RuntimeError("Triton inference failed") from error
        labels = result.as_numpy("label")
        if labels is None or labels.size != 1:
            raise RuntimeError("Triton returned an invalid label tensor")
        return int(labels[0])
