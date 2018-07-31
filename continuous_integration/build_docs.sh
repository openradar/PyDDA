set -e

cd "$TRAVIS_BUILD_DIR"

echo "Building Docs"
conda install -q sphinx pillow


