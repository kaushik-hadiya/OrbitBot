#!/usr/bin/env bash
# Deploy OrbitBot to Azure Container Apps, from the image GitHub Actions publishes to ghcr.io.
#
#   bash deploy/azure.sh                 # first deploy, and every update after a new image is built
#   NEW_KEY=1 bash deploy/azure.sh       # replace the OpenRouter key (asks for it, hidden) and reload
#   LOCATION=eastus bash deploy/azure.sh # another region (see "Region" below)
#
# Run it in Azure Cloud Shell (Bash): az is preinstalled and already logged in. The first run asks for the
# OpenRouter key once (hidden input) and stores it as a Container Apps secret, never in the repo or image.
#
# Only free things are created:
# - no Azure Container Registry (paid even on Basic): the image comes from ghcr.io, which must be public
# - no Log Analytics workspace (billed per GB of logs): live logs still work with `az containerapp logs show`
# - scale to zero when idle, at most 1 replica: inside Container Apps' free monthly grant, and the rate
#   limits and conversations (kept in memory) stay consistent on one replica
# Region: Azure for Students subscriptions only allow a few regions. If creation fails with
# "RequestDisallowedByAzure" / "disallowed by policy", the error lists the allowed ones: rerun with LOCATION=<one>.
set -euo pipefail

RG=orbitbot-rg
LOCATION="${LOCATION:-centralindia}"
ENV_NAME=orbitbot-env
APP=orbitbot
IMAGE="${IMAGE:-ghcr.io/kaushik-hadiya/orbitbot:latest}"

az account show --query "{subscription:name, id:id}" -o table
echo "Deploying $APP ($IMAGE) to $LOCATION"

echo "1/4 Preparing the subscription (first run only takes a minute)..."
az extension add --name containerapp --upgrade --only-show-errors
az provider register --namespace Microsoft.App --wait --only-show-errors
az group create --name "$RG" --location "$LOCATION" --only-show-errors -o none

echo "2/4 Container Apps environment..."
if ! az containerapp env show --name "$ENV_NAME" --resource-group "$RG" -o none 2>/dev/null; then
  az containerapp env create --name "$ENV_NAME" --resource-group "$RG" --location "$LOCATION" \
    --logs-destination none --only-show-errors -o none
fi

ask_key() {
  # hidden input; paste once. A wrong-looking length usually means it was pasted twice.
  read -rsp "Paste the DEMO OpenRouter key (the one with a credit limit) ONCE, then Enter: " KEY; echo
  [ -n "$KEY" ] || { echo "Empty key, stopping."; exit 1; }
  echo "Key length: ${#KEY} (OpenRouter keys are about 73 characters)"
}

if ! az containerapp show --name "$APP" --resource-group "$RG" -o none 2>/dev/null; then
  echo "3/4 Creating the app..."
  ask_key
  az containerapp create --name "$APP" --resource-group "$RG" --environment "$ENV_NAME" \
    --image "$IMAGE" \
    --ingress external --target-port 7860 \
    --cpu 1.0 --memory 2.0Gi \
    --min-replicas 0 --max-replicas 1 \
    --secrets "openrouter-key=$KEY" \
    --env-vars "OPENROUTER_API_KEY=secretref:openrouter-key" "PROXY_HOPS=1" \
    --only-show-errors -o none
  unset KEY
else
  if [ -n "${NEW_KEY:-}" ]; then
    echo "3/4 Replacing the OpenRouter key..."
    ask_key
    az containerapp secret set --name "$APP" --resource-group "$RG" \
      --secrets "openrouter-key=$KEY" --only-show-errors -o none
    unset KEY
  else
    echo "3/4 Updating the app to the newest image..."
  fi
  # The tag is always :latest, so the image alone doesn't count as a change. Changing DEPLOYED_AT does:
  # it makes a new revision, which pulls the newest :latest and reads the current secret. (This
  # environment type has no "revision restart", so this is also how a new key takes effect.)
  az containerapp update --name "$APP" --resource-group "$RG" --image "$IMAGE" \
    --set-env-vars "DEPLOYED_AT=$(date -u +%Y%m%dT%H%M%SZ)" --only-show-errors -o none
fi

echo "4/4 Done."
FQDN="$(az containerapp show --name "$APP" --resource-group "$RG" --query properties.configuration.ingress.fqdn -o tsv)"
echo
echo "URL:  https://$FQDN   (the first visit after idle wakes it up: ~30-60 s)"
echo "Logs: az containerapp logs show --name $APP --resource-group $RG --follow"
echo "Key:  to replace it later: NEW_KEY=1 bash deploy/azure.sh"
