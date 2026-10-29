set -euo pipefail
BASE=$(sed -n 's/^FROM \(python:[^ ]*\) AS base$/\1/p' Dockerfile)
docker run --rm -v "$PWD/requirements:/req" -w /req "$BASE" sh -c '
  pip install -q pip-tools &&
  pip-compile -q --generate-hashes --allow-unsafe --strip-extras -o serve.txt serve.in &&
  pip-compile -q --generate-hashes --allow-unsafe --strip-extras -o taskvector.txt taskvector.in'
