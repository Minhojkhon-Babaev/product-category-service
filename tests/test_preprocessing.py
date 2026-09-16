from product_service.preprocessing import cache_key, combine_text
from product_service.schemas import PredictRequest


def request(**texts):
    return PredictRequest(url="https://example.test", image_url="", texts=texts)


def test_combine_text_uses_training_field_order():
    payload = request(
        name="Name",
        description=" Description ",
        model=None,
        type_prefix="Type",
        vendor="Vendor",
    )

    assert combine_text(payload) == "Type Vendor Name Description"


def test_combine_text_handles_empty_fields():
    assert combine_text(request()) == ""


def test_cache_key_is_stable_and_payload_specific():
    first = request(name="A")
    same = request(name="A")
    different = request(name="B")

    assert cache_key(first) == cache_key(same)
    assert cache_key(first) != cache_key(different)
    assert cache_key(first).startswith("prediction:")
