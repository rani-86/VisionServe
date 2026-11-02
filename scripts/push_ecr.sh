#!/usr/bin/env bash
# Build the serving image and push it to Amazon ECR.
#   AWS_REGION=ap-south-1 ./scripts/push_ecr.sh [tag]
set -euo pipefail

REGION="${AWS_REGION:-ap-south-1}"
REPO="${ECR_REPO:-docserve}"
TAG="${1:-$(git rev-parse --short HEAD 2>/dev/null || date +%Y%m%d%H%M)}"
ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
REGISTRY="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"

aws ecr describe-repositories --repository-names "$REPO" --region "$REGION" >/dev/null 2>&1 \
  || aws ecr create-repository --repository-name "$REPO" --region "$REGION" \
       --image-scanning-configuration scanOnPush=true >/dev/null

aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$REGISTRY"
docker build -t "$REPO:$TAG" .
docker tag "$REPO:$TAG" "$REGISTRY/$REPO:$TAG"
docker push "$REGISTRY/$REPO:$TAG"
echo "pushed $REGISTRY/$REPO:$TAG"
