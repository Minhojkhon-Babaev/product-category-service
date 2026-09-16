from unittest.mock import Mock

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from tritonclient.utils import InferenceServerException

from product_service.inference import TritonClassifier


@pytest.fixture
def classifier(mocker):
    vectorizer = Mock()
    vectorizer.transform.return_value = csr_matrix([[0.0, 1.5, 0.0]], dtype=np.float64)
    client = mocker.patch("product_service.inference.httpclient.InferenceServerClient").return_value
    instance = TritonClassifier(vectorizer, "triton:8000", "product_category", timeout=3.0)
    return instance, vectorizer, client


def test_is_ready_requires_server_and_model(classifier):
    instance, _, client = classifier
    client.is_server_ready.return_value = True
    client.is_model_ready.return_value = True

    assert instance.is_ready() is True
    client.is_model_ready.assert_called_once_with("product_category")


def test_is_ready_short_circuits_when_server_is_down(classifier):
    instance, _, client = classifier
    client.is_server_ready.return_value = False

    assert instance.is_ready() is False
    client.is_model_ready.assert_not_called()


def test_predict_vectorizes_and_calls_triton(classifier):
    instance, vectorizer, client = classifier
    client.infer.return_value.as_numpy.return_value = np.array([42], dtype=np.int64)

    assert instance.predict("товар") == 42

    vectorizer.transform.assert_called_once_with(["товар"])
    call = client.infer.call_args
    assert call.kwargs["model_name"] == "product_category"
    assert call.kwargs["inputs"][0].shape() == (1, 3)
    assert call.kwargs["inputs"][0].datatype() == "FP32"


@pytest.mark.parametrize("value", [None, np.array([], dtype=np.int64), np.array([1, 2])])
def test_predict_rejects_invalid_output(classifier, value):
    instance, _, client = classifier
    client.infer.return_value.as_numpy.return_value = value

    with pytest.raises(RuntimeError, match="invalid label"):
        instance.predict("товар")


def test_predict_wraps_triton_error(classifier):
    instance, _, client = classifier
    client.infer.side_effect = InferenceServerException("failed")

    with pytest.raises(RuntimeError, match="Triton inference failed"):
        instance.predict("товар")
