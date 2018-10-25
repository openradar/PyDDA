set -e

echo "Building Docs"
conda install -c conda-forge -q sphinx doctr

cd doc
make clean
make html
cd ..
doctr deploy



