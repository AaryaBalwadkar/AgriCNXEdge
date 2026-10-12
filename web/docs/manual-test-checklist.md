# Manual Test Checklist

Run these yourself after implementation.

## Backend

```bash
cd agricnxedge-web/backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy ..\..\AgriCNXEdge\app\src\main\assets\adc_student_full.onnx models\adc_student_full.onnx
pytest
uvicorn app.main:app --reload --port 8000
```

Open:

```text
http://localhost:8000/api/health
http://localhost:8000/api/model/status
http://localhost:8000/docs
```

## Frontend

```bash
cd agricnxedge-web/frontend
npm install
npm run build
npm run dev
```

Open the Vite URL printed in the terminal.

## GitHub Actions

1. Create a GitHub repository.
2. Push the `agricnxedge-web` folder contents.
3. Check the Actions tab for the `build-test` job.
4. Add production secrets only when you have a server ready.
