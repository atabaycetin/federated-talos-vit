set -euo pipefail
CONFIG=${1:?config path}
TAG=${2:?release tag, e.g. model-v1}

rm -rf model_store && talos package --config "$CONFIG" --store model_store
tar -czf model_store.tar.gz --exclude 'model.onnx' model_store
gh release create "$TAG" model_store.tar.gz --title "$TAG" \
  --notes "Model store for the serving image. Bundle meta: $(ls model_store)."
gh variable set MODEL_RELEASE --body "$TAG"
echo "Published $TAG and set MODEL_RELEASE; the next push to main builds an image with it."
