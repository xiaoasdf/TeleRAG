from bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

import uvicorn


if __name__ == "__main__":
    uvicorn.run("src.main:app", host="127.0.0.1", port=8000, reload=False)
