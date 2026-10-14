from app.auth.security import hash_password, normalize_email, verify_password


def test_password_hash_is_not_plaintext():
    hashed = hash_password("correct horse battery")
    assert hashed != "correct horse battery"
    assert verify_password("correct horse battery", hashed)
    assert not verify_password("wrong password", hashed)


def test_email_normalization():
    assert normalize_email("  PERSON@Example.COM ") == "person@example.com"

