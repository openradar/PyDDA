set -e

echo "Building Docs"
conda install -c conda-forge -q sphinx doctr
pip install sphinx_gallery
cd doc
make clean
make html
cd ..
doctr deploy



