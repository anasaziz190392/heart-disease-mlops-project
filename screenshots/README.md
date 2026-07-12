# Screenshots Folder

This folder holds evidence screenshots for the written report (Section 9 of
the assignment). Since actual cloud/K8s deployment and GitHub Actions runs
require live infrastructure and repository secrets that only exist once you
push this repo to your own GitHub account and cloud project, capture the
following after you complete the deployment steps in the README:

1. `01_mlflow_ui.png` — MLflow UI showing both experiment runs (Logistic
   Regression + Random Forest) with logged params/metrics/artifacts.
2. `02_pytest_pass.png` — terminal output of `pytest tests/ -v` passing.
3. `03_github_actions_run.png` — GitHub Actions workflow run, all jobs green
   (lint → unit-tests → build-and-push-image → deploy).
4. `04_docker_build.png` — `docker build` completing successfully.
5. `05_docker_run_curl.png` — `docker run` + a `curl /predict` call and
   response.
6. `06_k8s_pods.png` — `kubectl get pods,svc` showing the running
   `heart-disease-api` pods and the LoadBalancer/NodePort service.
7. `07_grafana_dashboard.png` — Grafana dashboard with live request-rate /
   latency panels after sending some test traffic.
8. `08_swagger_docs.png` — the FastAPI `/docs` Swagger UI.

Reference these filenames from `reports/final_report.docx` once captured.
