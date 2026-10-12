# Pipeline Diagram

```text
Developer push / pull request
        |
        v
GitHub repository
        |
        v
GitHub Actions: build-test
        |-- checkout
        |-- setup Python 3.11
        |-- install FastAPI dependencies
        |-- run pytest
        |-- setup Node.js 22
        |-- install React dependencies
        |-- build Vite frontend
        |
        v
GitHub Actions: deploy, main branch only
        |-- SSH to provisioned server
        |-- pull latest code
        |-- install backend dependencies
        |-- build frontend
        |-- restart FastAPI systemd service
        |-- reload Nginx
        |
        v
Live AgriCNXEdge Web service
```
