set -euo pipefail
[ $# -ge 2 ] || { echo "usage: $0 CONFIG [CONFIG...] TAG" >&2; exit 1; }
TAG=${*: -1}
CONFIGS=("${@:1:$#-1}")
command -v talos >/dev/null || { echo "talos not found: run 'conda activate talos-mlops' first" >&2; exit 1; }

rm -rf model_store
for CONFIG in "${CONFIGS[@]}"; do
  talos package --config "$CONFIG" --store model_store
done
tar -czf model_store.tar.gz --exclude 'model.onnx' model_store
gh release create "$TAG" model_store.tar.gz --title "$TAG" \
  --notes "Model store for the serving image. Tasks: $(ls model_store | xargs)."
gh variable set MODEL_RELEASE --body "$TAG"
echo "Published $TAG and set MODEL_RELEASE; the next push to main builds an image with it."
