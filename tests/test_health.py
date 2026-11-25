from src.config import get_db_path


def test_health_returns_ok(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_responses_include_request_id_header(client):
    response = client.get("/health")

    assert "X-Request-ID" in response.headers


def test_tests_use_temporary_database(test_db_path):
    assert get_db_path() == test_db_path
    assert get_db_path().name == "test_app.db"
