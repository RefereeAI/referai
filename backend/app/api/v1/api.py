# app/api/v1/api.py
from fastapi import APIRouter
from app.api.v1.endpoints import user

api_router = APIRouter()
api_router.include_router(user.router, prefix="/users", tags=["users"])
