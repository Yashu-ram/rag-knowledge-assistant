from app.api.schemas import SignupRequest


def test_signup_password_allows_short_arbitrary_value():
    payload = SignupRequest(full_name="A", email="a@example.com", password="x")
    assert payload.password == "x"
