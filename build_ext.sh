python setup.py clean
rm -r build
rm ikg_native*.so
python setup.py build_ext --inplace -j 16