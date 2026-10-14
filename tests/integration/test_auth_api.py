from tests.conftest import auth_headers, signup


def test_signup_login_logout_and_protected_access(client):
    user = signup(client)

    me = client.get("/auth/me", headers=auth_headers(user["access_token"]))
    assert me.status_code == 200
    assert me.json()["email"] == "person@example.com"

    login = client.post(
        "/auth/login",
        json={"email": "person@example.com", "password": "strong-password-123"},
    )
    assert login.status_code == 200
    token = login.json()["access_token"]

    logout = client.post("/auth/logout", headers=auth_headers(token))
    assert logout.status_code == 200

    denied = client.get("/auth/me", headers=auth_headers(token))
    assert denied.status_code == 401


def test_duplicate_signup_and_bad_login(client):
    signup(client)
    duplicate = client.post(
        "/auth/signup",
        json={"full_name": "Test Person", "email": "PERSON@example.com", "password": "strong-password-123"},
    )
    assert duplicate.status_code == 409

    bad_login = client.post(
        "/auth/login",
        json={"email": "person@example.com", "password": "wrong-password"},
    )
    assert bad_login.status_code == 401


def test_protected_documents_require_auth(client):
    response = client.get("/documents")
    assert response.status_code == 401

