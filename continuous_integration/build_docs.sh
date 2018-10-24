set -e

echo "Building Docs"
conda install -q sphinx doctr

cd docs
make clean
make html
cd ..
doctr deploy



