#!/usr/bin/env bash
# One-time Azure setup: resource group, Container Apps environment, the app itself, and an
# OIDC identity so GitHub Actions can deploy without stored passwords.
# Mirrors the manual deployment: zero-spend caps, no Log Analytics, public GHCR image.
# Requires: az login, gh auth login, and the image already on GHCR as a *public* package.
set -euo pipefail

RESOURCE_GROUP=${RESOURCE_GROUP:-talos-rg}
# Azure for Students only allows a few regions (norwayeast, switzerlandnorth, belgiumcentral,
# denmarkeast, spaincentral); Container Apps isn't offered in denmarkeast.
LOCATION=${LOCATION:-switzerlandnorth}
ENV_NAME=${ENV_NAME:-talos-env}
APP_NAME=${APP_NAME:-talos-api}
IDENTITY_NAME=${IDENTITY_NAME:-$APP_NAME-github-deploy}
REPO=${REPO:-$(gh repo view --json nameWithOwner -q .nameWithOwner)}
# CI pushes :latest and :<sha>; v1 is the first image pushed by hand.
IMAGE=${IMAGE:-ghcr.io/$(echo "$REPO" | tr '[:upper:]' '[:lower:]'):v1}

# Fail early instead of letting Azure report a misleading "authentication failed" pull error.
IMAGE_PATH=${IMAGE#ghcr.io/}
IMAGE_PATH=${IMAGE_PATH%:*}
if [ "$(curl -s -o /dev/null -w '%{http_code}' "https://ghcr.io/token?scope=repository:$IMAGE_PATH:pull")" != 200 ]; then
  echo "error: $IMAGE is not publicly pullable; make the GHCR package public first" >&2
  exit 1
fi

az extension add --name containerapp --upgrade --only-show-errors
az provider register --namespace Microsoft.App --wait
az provider register --namespace Microsoft.ManagedIdentity --wait

az group create --name "$RESOURCE_GROUP" --location "$LOCATION" -o none
# No Log Analytics workspace (billed per GB); the Portal's Log stream still works.
az containerapp env create --name "$ENV_NAME" --resource-group "$RESOURCE_GROUP" --location "$LOCATION" \
  --logs-destination none -o none

# Zero-spend caps: scale to zero when idle, at most one replica, and a size that stretches the
# monthly free grant (180k vCPU-s, 360k GiB-s) to ~100 active hours.
az containerapp create \
  --name "$APP_NAME" --resource-group "$RESOURCE_GROUP" --environment "$ENV_NAME" \
  --image "$IMAGE" --target-port 8000 --ingress external \
  --cpu 0.5 --memory 1.0Gi --min-replicas 0 --max-replicas 1 \
  --env-vars BACKEND=onnx THREADS=1 -o none

# GitHub Actions -> Azure via OIDC. A user-assigned managed identity lives in your subscription,
# so it works on university tenants that block app registrations (az ad app create).
az identity create --name "$IDENTITY_NAME" --resource-group "$RESOURCE_GROUP" -o none
CLIENT_ID=$(az identity show -n "$IDENTITY_NAME" -g "$RESOURCE_GROUP" --query clientId -o tsv)
PRINCIPAL_ID=$(az identity show -n "$IDENTITY_NAME" -g "$RESOURCE_GROUP" --query principalId -o tsv)
# Least privilege: manage this one app (and join its environment), nothing else. Not Contributor
# on the resource group, which could create any resource or edit this identity's own credentials.
APP_RES_ID=$(az containerapp show -n "$APP_NAME" -g "$RESOURCE_GROUP" --query id -o tsv)
ENV_RES_ID=$(az containerapp env show -n "$ENV_NAME" -g "$RESOURCE_GROUP" --query id -o tsv)
for SCOPE in "$APP_RES_ID" "$ENV_RES_ID"; do
  az role assignment create --assignee-object-id "$PRINCIPAL_ID" --assignee-principal-type ServicePrincipal \
    --role "Container Apps Contributor" --scope "$SCOPE" -o none
done
# Only workflow jobs running in this repo's `production` environment can get a token. GitHub may
# put the owner/repo IDs in the token subject (repo:owner@123/name@456), so ask it for the prefix.
SUB_PREFIX=$(gh api "repos/$REPO/actions/oidc/customization/sub" --jq .sub_claim_prefix 2>/dev/null || true)
SUB_PREFIX=${SUB_PREFIX:-repo:$REPO}
az identity federated-credential create --name github-production \
  --identity-name "$IDENTITY_NAME" --resource-group "$RESOURCE_GROUP" \
  --issuer https://token.actions.githubusercontent.com \
  --subject "${SUB_PREFIX}:environment:production" \
  --audiences api://AzureADTokenExchange -o none

# The `production` environment only accepts deployments from main, and holds the Azure IDs, so
# a workflow on any other branch can neither read them nor get an Azure token.
gh api -X PUT "repos/$REPO/environments/production" --silent --input - <<'JSON'
{"deployment_branch_policy": {"protected_branches": false, "custom_branch_policies": true}}
JSON
if ! gh api "repos/$REPO/environments/production/deployment-branch-policies" \
  --jq '.branch_policies[].name' | grep -qx main; then
  gh api -X POST "repos/$REPO/environments/production/deployment-branch-policies" \
    -f name=main -f type=branch --silent
fi
gh secret set AZURE_CLIENT_ID --env production --body "$CLIENT_ID"
gh secret set AZURE_TENANT_ID --env production --body "$(az account show --query tenantId -o tsv)"
gh secret set AZURE_SUBSCRIPTION_ID --env production --body "$(az account show --query id -o tsv)"
gh variable set AZURE_RESOURCE_GROUP --body "$RESOURCE_GROUP"
gh variable set AZURE_CONTAINERAPP_NAME --body "$APP_NAME"

FQDN=$(az containerapp show -n "$APP_NAME" -g "$RESOURCE_GROUP" --query properties.configuration.ingress.fqdn -o tsv)
echo "Deployed: https://$FQDN/docs"
echo "Pushes to main roll out automatically once MODEL_RELEASE is set (scripts/publish_model.sh)."
echo "Tear everything down with: az group delete -n $RESOURCE_GROUP --yes"
