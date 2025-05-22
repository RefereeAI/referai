from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from typing import List
from sqlalchemy.orm import Session
from app.db.models import Prediction, User, Action, Clip


from app.db.database import get_db
from app.auth.jwt_utils import get_current_user

from app.models.predictor import predict

import os
import tempfile

router = APIRouter()

class PredictRequest(BaseModel):
    video_filenames: List[str]

class SinglePrediction(BaseModel):
    filename: str
    is_foul: bool
    foul_confidence: float
    no_foul_confidence: float
    severity: dict
    foul_model_results: List[dict]
    severity_model_results: List[dict]

class PredictResponse(BaseModel):
    results: List[SinglePrediction]


@router.post("/predict/{action_id}", response_model=PredictResponse)
async def predict_endpoint(action_id: int,request: Request, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    # Verify the action exists and belongs to the current user
    action = db.query(Action).filter(Action.id == action_id, Action.user_id == current_user.id).first()
    if not action:
        raise HTTPException(status_code=404, detail="Action not found or you do not have permission to access it.")

    # Obtains the clips associated with the action
    clips = db.query(Clip).filter(Clip.action_id == action.id).all()
    if not clips:
        raise HTTPException(status_code=404, detail="No clips found for this action.")
    
    # Delete the previous prediction if it exists
    existing_prediction = db.query(Prediction).filter(Prediction.action_id == action.id).first()
    if existing_prediction:
        db.delete(existing_prediction)
        db.commit()

    # Creates temporary files for all clips
    video_paths = []
    try:
        for clip in clips:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
                temp_video.write(clip.content)
                video_paths.append(temp_video.name)
        
        # Prediction call
        prediction_results = predict(video_paths,request)

        # Save prediction to the database
        prediction = Prediction(
            action_id=action.id,
            is_foul=prediction_results["is_foul"],
            foul_confidence=prediction_results["foul_confidence"],
            no_foul_confidence=prediction_results["no_foul_confidence"],
            foul_model_results=prediction_results["foul_model_results"],  # Ensure this is serialized as JSON
            no_card_confidence=prediction_results["severity"]["no_card"],
            red_card_confidence=prediction_results["severity"]["red_card"],
            yellow_card_confidence=prediction_results["severity"]["yellow_card"],
            severity_model_results=prediction_results["severity_model_results"]  # Ensure this is serialized as JSON
        )
        db.add(prediction)
        db.commit()
        db.refresh(prediction)

        results = [{
            "filename": f"clip_{clip.id}.mp4",
            "is_foul": prediction_results["is_foul"],
            "foul_confidence": prediction_results["foul_confidence"],
            "no_foul_confidence": prediction_results["no_foul_confidence"],
            "severity": prediction_results["severity"],
            "foul_model_results": prediction_results["foul_model_results"],
            "severity_model_results": prediction_results["severity_model_results"]
        }]

    finally:
        # Delete temporary files
        for path in video_paths:
            if os.path.exists(path):
                os.remove(path)

    return {"results": results}

@router.get("/predict/{action_id}", response_model=PredictResponse)
async def get_prediction(
    action_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # Verificar que la acción existe y pertenece al usuario
    action = db.query(Action).filter(Action.id == action_id, Action.user_id == current_user.id).first()
    if not action:
        raise HTTPException(status_code=404, detail="Action not found or access denied")

    # Obtener predicción única asociada a esta acción
    prediction = db.query(Prediction).filter(Prediction.action_id == action_id).first()
    if not prediction:
        raise HTTPException(status_code=404, detail="No prediction found for this action")

    # Construir la respuesta en el mismo formato que el POST
    result = {
        "filename": f"action_{action_id}.mp4",  # o algún nombre simbólico
        "is_foul": prediction.is_foul,
        "foul_confidence": prediction.foul_confidence,
        "no_foul_confidence": prediction.no_foul_confidence,
        "severity": {
            "no_card": prediction.no_card_confidence,
            "red_card": prediction.red_card_confidence,
            "yellow_card": prediction.yellow_card_confidence
        },
        "foul_model_results": prediction.foul_model_results,
        "severity_model_results": prediction.severity_model_results
    }

    return {"results": [result]}

