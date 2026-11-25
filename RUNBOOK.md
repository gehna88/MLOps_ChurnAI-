# ChurnAI runbook

How to check, redeploy and roll back the live ChurnAI API.

| | |
|---|---|
| **Live URL** | https://churnai-api-g8a9fnb8ajhsbhhx.centralindia-01.azurewebsites.net |
| **Platform** | Azure App Service (Web App for Containers), app `churnai-api`, resource group `churnai-rg`, region Central India, plan Basic B1 |
| **Image** | `ghcr.io/gehna88/churnai`, tagged `sha-<7-char commit>` (one per push to `main`), `latest`, and `v*` release tags |
| **Deploys** | Automatic on every push to `main`, through [`.github/workflows/ci.yml`](.github/workflows/ci.yml) |
| **Rollback** | [`.github/workflows/rollback.yml`](.github/workflows/rollback.yml): *Actions → Rollback → Run workflow* |
| **GitHub settings it needs** | Secret `AZURE_WEBAPP_PUBLISH_PROFILE`; variables `AZURE_WEBAPP_NAME` (`churnai-api`) and `SERVICE_URL` (the live URL) |

The model is **baked into the image**. Deploying or rolling back an image tag always brings
its own model with it, so code and model can never get out of step.

---

## 1. Is it healthy?

```bash
URL=https://churnai-api-g8a9fnb8ajhsbhhx.centralindia-01.azurewebsites.net
curl $URL/health
curl $URL/stats
```

**Healthy** looks like this:

```json
{"status": "ok", "model_loaded": true, "model_version": "xgb_tuned-v1-afa3870f",
 "build": "sha-a170d9f", "uptime_seconds": 85.3}
```

- **`build`** is the image tag that is live. Compare it with the latest successful CI run
  on `main` to confirm the newest version is deployed.
- **`/stats`** shows `requests_total`, `errors_total` and `latency.avg_ms`. A rising
  `errors_total` for `/predict` means requests are failing.
- **Logs:** Azure portal → `churnai-api` → **Monitoring → Log stream**. There is one JSON line
  per request, and at start-up `"message": "Model loaded"`, then
  `Uvicorn running on http://0.0.0.0:8000`.
- The first request after a long idle period can take up to a minute while Azure starts the
  container.

---

## 2. Redeploy

### Normal deploy: merge to `main`

1. Open a pull request into `main` and get it approved.
2. Wait until **Lint**, **Test** and **Build** are green, then merge.
3. The CI run on `main` then:
   1. builds and smoke-tests the image
   2. pushes `ghcr.io/gehna88/churnai:sha-xxxxxxx` and `:latest`
   3. deploys `sha-xxxxxxx` to Azure (`azure/webapps-deploy`)
   4. polls `/health` every 15 s for up to 10 minutes, until `build` = `sha-xxxxxxx`
4. **Done when** the step *Wait until Azure serves the new build* is green. The run summary
   shows `Deployed ghcr.io/gehna88/churnai:sha-xxxxxxx to Azure`.

### Redeploy without a code change

- **Same commit again:** Actions → the CI run on `main` → **Re-run all jobs**. This rebuilds
  and redeploys.
- **Same image again, no rebuild:** run the **Rollback** workflow (section 3) with the tag
  that is already live.
- **Just restart the container:** Azure portal → `churnai-api` → **Overview → Restart**. The
  same image starts again.

---

## 3. Roll back to a previous image tag

Use this when a deploy breaks the service (errors in `/stats`, `/health` not `ok`, wrong
predictions) and the fix will take longer than a few minutes.

### 3.1 Find the last good tag

Any of these lists the tags:
- **GitHub → Packages → churnai**: every `sha-…` tag with its date
  (https://github.com/users/gehna88/packages/container/package/churnai)
- **Actions → CI → a green run on `main`**: the summary says `Pushed …:sha-xxxxxxx`
- **Git history:** the tag is `sha-` plus the first 7 characters of the commit, e.g. commit
  `a170d9f…` gives `sha-a170d9f`. List them with `git log --oneline origin/main`.

### 3.2 Run the rollback

1. GitHub → **Actions** → **Rollback** (left sidebar) → **Run workflow**.
2. Branch: `main`. **Image tag:** the last good tag, e.g. `sha-a170d9f`.
3. Click **Run workflow**. It will:
   1. check the Azure secret and variables are set
   2. check the image tag exists in GHCR (it stops here if you mistyped it)
   3. deploy that exact image to Azure
   4. wait until `/health` reports `build` = that tag
4. **Done when** the run is green and the summary says `Rolled back to …:sha-xxxxxxx on Azure`.

No rebuild and no retraining happen: the same image that passed CI earlier is redeployed.

### 3.3 Verify

```bash
curl $URL/health      # "build" must be the tag you rolled back to
curl $URL/stats       # errors_total should stop rising
```

### 3.4 Roll forward again

Once the bug is fixed on `main`, the next merge deploys a new tag automatically. To return
to a specific newer tag instead, run **Rollback** again with that tag.

### 3.5 If GitHub Actions is unavailable

Azure portal → `churnai-api` → **Deployment Center** (or **Containers**) → container
`main` → change the image tag to the last good tag (e.g. `gehna88/churnai:sha-a170d9f`) →
**Save**. Azure pulls and restarts it. Verify with `/health` as in 3.3.

### 3.6 Run an older image tag locally

Every tag in GHCR still runs on its own, with the model it was built with. Use this to check a
tag starts correctly before rolling back to it, or to reproduce a bug from an older release.

```bash
TAG=sha-a27f762                                   # any tag from 3.1
docker pull ghcr.io/gehna88/churnai:$TAG
docker run --rm -p 8000:8000 ghcr.io/gehna88/churnai:$TAG
```

In a second terminal:

```bash
curl http://localhost:8000/health   # "build" must be the tag you ran, "model_loaded": true
docker ps                           # STATUS shows (healthy) after about a minute
```

- **Apple Silicon Macs:** the images are built for `linux/amd64`, so add `--platform linux/amd64`
  to `docker pull` and `docker run`. They run under emulation, a little more slowly.
- **Port 8000 in use:** map another port on your side, e.g. `-p 8001:8000`.
- `docker images ghcr.io/gehna88/churnai` lists the tags you have pulled.

---

## 4. Rollback drill

A rehearsal of section 3 on the live service, to prove the procedure works before it's needed.

1. Note the current build: `curl $URL/health` gives `build` = **new tag**.
2. Run **Rollback** with the **previous** tag (e.g. `sha-a170d9f`).
3. Check `/health` shows the previous tag, and that `/predict` works.
4. Run **Rollback** with the **new** tag to roll forward.
5. Check `/health` shows the new tag again.

| Date | Live before | Rolled back to | Rolled forward to | Result |
|---|---|---|---|---|
| 2026-11-24 | `sha-f93473e` | `sha-a27f762` | `sha-f93473e` | ✅ Passed |

What we checked on the live service at each step:

| Step | `/health` → `build` | `/health` → `model_version` | Other checks |
|---|---|---|---|
| Before | `sha-f93473e` | `xgb_tuned-v1-27b17d2c` | `status: ok`, model loaded |
| After rollback | `sha-a27f762` | `xgb_tuned-v1-ad112305` | `/predict` returned a prediction (0.664, churn); uptime reset to 148 s, so Azure restarted on the older image. `/stats` did not exist in that build yet; `/metrics` did |
| After roll-forward | `sha-f93473e` | `xgb_tuned-v1-27b17d2c` | `/stats` available again, `errors_total: 0`; uptime reset to 61 s |

**What the drill showed:**
- The Rollback workflow switched the live service to an older image and back, with no rebuild
  and no retraining. Each run finished only once `/health` reported the requested build, and
  each took **between 1 m 28 s and 2 m 12 s** (GitHub Actions run durations), so recovery
  from a bad deploy takes about two minutes.
- The **model rolls back with the code**: `model_version` changed with the image, because
  each image ships its own trained model.
- `build` in `/health` is a reliable way to confirm which version is live.

---

## 5. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Browser shows **"Application Error"** or `/health` times out right after a deploy | Container still starting (pulling a ~390 MB image) | Wait 2–5 minutes; watch **Log stream** |
| Logs say the container **didn't respond on port 8000** | Port setting lost | App setting `WEBSITES_PORT=8000`; container port 8000 in Deployment Center |
| Logs show an **image pull error** | Wrong tag, or the GHCR package was made private | Check the tag exists on the package page; the package must stay **Public** |
| CI step **Deploy to Azure App Service** fails with an auth error | Publish profile missing, expired or reset | Azure → Overview → **Download publish profile** again; update the `AZURE_WEBAPP_PUBLISH_PROFILE` secret. SCM Basic Auth Publishing Credentials must be **On** |
| CI step **Wait until Azure serves the new build** times out | Azure is still running the old container, or the new one crashed on start | Check **Log stream**; if the new image is broken, roll back (section 3) |
| `/health` returns `"status": "unavailable"` | The model failed to load inside the image | Roll back to the last good tag, then check the CI Build job logs for that commit |
| `/stats` shows `insufficient_data` under `drift` | Fewer than 100 requests since start-up | Normal; drift needs at least 100 requests |
| `POST /drift/study/run` returns **409** | A study is already running | Wait; follow it at `/drift/study/status` |
| `/drift/study/status` shows `failed` | The error is in the `error` field (e.g. no model) | Fix the cause, run again. The previous results stay in place until a study succeeds |

---

## 6. Drift study on Azure

The image ships the study CI ran before the build, so `/drift/study` works right after a deploy.

* **Run a new study on the live service:** `POST /drift/study/run` (body optional, e.g.
  `{"n_slices": 10}`), then poll `GET /drift/study/status` until `succeeded`. On the B1 plan
  (1 core) a 5-slice study takes a few minutes; predictions keep being served meanwhile, a little
  slower. Only one study runs at a time.
* **MLflow on Azure** writes to `/app/mlflow/mlflow.db` inside the container, which is reset on
  every restart or deploy. To keep runs, set the app setting
  `CHURNAI_MLFLOW_TRACKING_URI=sqlite:////home/mlflow/mlflow.db` (and
  `CHURNAI_MLFLOW_ARTIFACT_DIR=/home/mlflow/artifacts`) with
  `WEBSITES_ENABLE_APP_SERVICE_STORAGE=true`, so they live on the persistent `/home` share. The
  same applies to new study results: `CHURNAI_DRIFT_STUDY_DIR=/home/drift_study`.
* **See the runs:** `GET /drift/study/mlflow-runs`.

---

## 7. Shutting down to save credit

The Basic B1 plan costs about 13 USD/month from the Azure for Students credit.
- **Pause:** Azure portal → `churnai-api` → **Stop**. Billing for the plan continues; the app stops serving.
- **Remove completely after grading:** **Resource groups → `churnai-rg` → Delete resource group.**
  Then note in the README that the service was shut down.
Last verified on the live service: 2026-11-25, by Gehna Bhatia and M. Zayed Nauman
