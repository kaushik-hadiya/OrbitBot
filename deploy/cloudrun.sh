#!/usr/bin/env bash
# Deploy OrbitBot to Google Cloud Run: builds the Dockerfile with Cloud Build and serves it at a public URL.
#
#   bash deploy/cloudrun.sh            # first deploy and every update after it
#
# Run it from the repo root in Google Cloud Shell (gcloud is preinstalled there), or anywhere gcloud is
# logged in, with a project selected:  gcloud config set project <PROJECT_ID>
# The first run asks for the OpenRouter key once (hidden input) and keeps it in Secret Manager; it is never
# written to the repo or the image.
#
# Settings, chosen to stay inside the free tier:
# - request-based billing (CPU only while answering) and scale to zero when idle: no cost between visits,
#   at the price of a cold start (~10-30 s) for the first visitor
# - max 1 instance: the rate limits and conversations live in memory, so one instance keeps them consistent,
#   and it caps how much the demo can ever use
# - old images are deleted automatically, so storage stays under Artifact Registry's free 0.5 GB
set -euo pipefail

SERVICE=orbitbot
REGION="${REGION:-us-central1}"      # a Tier 1 region: the free tier is priced at Tier 1 rates
SECRET=openrouter-demo-key
PROJECT="$(gcloud config get-value project 2>/dev/null)"
[ -n "$PROJECT" ] || { echo "No project selected. Run: gcloud config set project <PROJECT_ID>"; exit 1; }
[ -f Dockerfile ] || { echo "Run this from the repo root (where the Dockerfile is)."; exit 1; }
echo "Deploying $SERVICE to project $PROJECT, region $REGION"

echo "1/4 Enabling the APIs (first run only takes a minute)..."
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com \
  secretmanager.googleapis.com --quiet

echo "2/4 Checking the OpenRouter key in Secret Manager..."
if ! gcloud secrets describe "$SECRET" >/dev/null 2>&1; then
  read -rsp "Paste the DEMO OpenRouter key (the one with a credit limit), then Enter: " KEY; echo
  [ -n "$KEY" ] || { echo "Empty key, stopping."; exit 1; }
  printf %s "$KEY" | gcloud secrets create "$SECRET" --replication-policy=automatic --data-file=- --quiet
  unset KEY
fi
# Cloud Run runs as the project's default compute service account: let it read this one secret
NUMBER="$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')"
gcloud secrets add-iam-policy-binding "$SECRET" \
  --member="serviceAccount:${NUMBER}-compute@developer.gserviceaccount.com" \
  --role=roles/secretmanager.secretAccessor --quiet >/dev/null

echo "3/4 Building and deploying (about 5-10 minutes the first time)..."
gcloud run deploy "$SERVICE" \
  --source . \
  --region "$REGION" \
  --allow-unauthenticated \
  --cpu 1 --memory 2Gi \
  --cpu-throttling --cpu-boost \
  --min-instances 0 --max-instances 1 \
  --concurrency 10 --timeout 300 \
  --set-secrets "OPENROUTER_API_KEY=${SECRET}:latest" \
  --set-env-vars "PROXY_HOPS=1" \
  --quiet

echo "4/4 Keeping only the 2 newest images (storage stays in the free tier)..."
POLICY="$(mktemp)"
cat > "$POLICY" <<'JSON'
[
  {"name": "keep-newest", "action": {"type": "Keep"}, "mostRecentVersions": {"keepCount": 2}},
  {"name": "delete-older", "action": {"type": "Delete"}, "condition": {"tagState": "ANY"}}
]
JSON
gcloud artifacts repositories set-cleanup-policies cloud-run-source-deploy \
  --location="$REGION" --policy="$POLICY" --no-dry-run --quiet >/dev/null
rm -f "$POLICY"

URL="$(gcloud run services describe "$SERVICE" --region "$REGION" --format='value(status.url)')"
echo
echo "Done: $URL"
echo "Logs: gcloud run services logs read $SERVICE --region $REGION --limit 50"
