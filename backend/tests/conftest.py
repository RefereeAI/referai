import os
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.database import get_db
from app.db.models import Base
from main import app

# Set env vars early
os.environ["ENV"] = "test"
os.environ["DB_URL"] = "sqlite:///./tests/test.db"

# Setup DB
TEST_DATABASE_URL = os.environ["DB_URL"]
engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create schema
@pytest.fixture(scope="session", autouse=True)
def setup_database():
    # For SQLite, just remove and recreate the tables
    if os.path.exists("./tests/test.db"):
        os.remove("./tests/test.db")
    Base.metadata.create_all(bind=engine)
    yield
    engine.dispose()

# Provide DB session to tests
@pytest.fixture()
def db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

# Provide HTTP client with overridden DB
@pytest_asyncio.fixture()
async def client(db):
    def override_get_db():
        yield db

    app.dependency_overrides[get_db] = override_get_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
