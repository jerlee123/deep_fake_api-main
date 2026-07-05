# Render Deploy Bundle

This folder is the minimal deployment entrypoint for Render.

Keep here:
- `app.py`
- `requirements.txt`
- `static/`

Do not copy:
- `__pycache__/`
- notebooks
- training scripts
- datasets
- benchmark outputs
- local secrets or private keys

Render settings for this folder:
- Root Directory: `render_deploy`
- Build Command: `pip install -r requirements.txt`
- Start Command: `uvicorn app:app --host 0.0.0.0 --port $PORT`

Notes:
- `__pycache__` is not needed.
- If you want live inference to work, make sure the checkpoint file is available to the app, such as `ekyc_checkpoint.pth` or `checkpoint.pth`.
- If you move this bundle to a separate repository, also copy the runtime Python files it imports, especially `apiForAppNewOne.py` and `vivit_bilstm_model.py`.
