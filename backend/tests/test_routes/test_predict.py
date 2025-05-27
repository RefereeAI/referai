import pytest
from main import app
from httpx import AsyncClient
import uuid
import pytest
from asgi_lifespan import LifespanManager
import pytest_asyncio

# ---------------- Fixtures ---------------- #

@pytest_asyncio.fixture
async def client():
    async with LifespanManager(app):
        async with AsyncClient(app=app, base_url="http://test") as ac:
            yield ac

@pytest_asyncio.fixture
async def test_user(client: AsyncClient):
    email = f"user_{uuid.uuid4().hex[:6]}@example.com"
    password = "Test1234!"
    await client.post("/register", json={"email": email, "password": password, "confirm_password": password})
    response = await client.post("/login", data={"username": email, "password": password}, headers={"Content-Type": "application/x-www-form-urlencoded"})
    token = response.json()["access_token"]
    return {"email": email, "token": token}

@pytest_asyncio.fixture
def override_get_db():
    from app.db.database import SessionLocal
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ---------------- Tests ---------------- #

@pytest.mark.asyncio
async def test_upload_and_predict(client, test_user, override_get_db):
    test_user_data = test_user
    headers = {"Authorization": f"Bearer {test_user_data['token']}"}

    # Cargar archivos reales
    with open("tests/assets/videos/clip_0.mp4", "rb") as f1, open("tests/assets/videos/clip_1.mp4", "rb") as f2:
        files = [
            ("files", ("clip_0.mp4", f1, "video/mp4")),
            ("files", ("clip_1.mp4", f2, "video/mp4")),
        ]
        response = await client.post("/upload", files=files, headers=headers)

    assert response.status_code == 200
    action_id = response.json()["action_id"]

    response = await client.post(f"/predict/{action_id}", headers=headers)
    assert response.status_code == 200
    assert "results" in response.json()


@pytest.mark.asyncio
async def test_get_prediction(client: AsyncClient, test_user, override_get_db):
    test_user_data = test_user
    headers = {"Authorization": f"Bearer {test_user_data['token']}"}

    # Cargar archivos reales
    with open("tests/assets/videos/clip_0.mp4", "rb") as f1, open("tests/assets/videos/clip_1.mp4", "rb") as f2:
        files = [
            ("files", ("clip_0.mp4", f1, "video/mp4")),
            ("files", ("clip_1.mp4", f2, "video/mp4")),
        ]
        response = await client.post("/upload", files=files, headers=headers)

    assert response.status_code == 200
    action_id = response.json()["action_id"]

    response = await client.post(f"/predict/{action_id}", headers=headers)
    assert response.status_code == 200

    response = await client.get(f"/predict/{action_id}", headers=headers)
    assert response.status_code == 200
    assert "results" in response.json()
    assert len(response.json()["results"]) == 1