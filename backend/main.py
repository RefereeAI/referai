from fastapi import FastAPI, File, UploadFile
import os
from app.auth.routes import router as auth_router
from app.action.routes import router as action_router
from app.predict.routes import router as predict_router
from fastapi.middleware.cors import CORSMiddleware
from app.models.predictor import load_models,load_severity_models
import os


# Models
FOUL_MODELS = None
SEVERITY_MODELS = None

def print_ascii_art():
    logo = r"""
     ____  _____ _____ _____ ____      _    ___ 
    |  _ \| ____|  ___| ____|  _ \    / \  |_ _|
    | |_) |  _| | |_  |  _| | |_) |  / _ \  | | 
    |  _ <| |___|  _| | |___|  _ <  / ___ \ | | 
    |_| \_\_____|_|   |_____|_| \_\/_/   \_\___|
                    REFERAI API
    """
    print(logo)

def lifespan(app: FastAPI):
    if os.environ.get("ENV") == "test":
        yield
        return

    print_ascii_art()

    foul_models = load_models(os.path.join(os.path.dirname(__file__), "app/models/foul"))
    if len(foul_models) != 3:
        raise RuntimeError("3 foul models were expected.")

    severity_models = load_severity_models(os.path.join(os.path.dirname(__file__), "app/models/severity"))
    if len(severity_models) != 3:
        raise RuntimeError("3 severity models were expected.")

    app.state.foul_models = foul_models
    app.state.severity_models = severity_models

    print("Models loaded successfully.")
    yield

    print("Shutting down...")

app = FastAPI(title="Referai API", lifespan=lifespan)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(action_router)
app.include_router(predict_router)