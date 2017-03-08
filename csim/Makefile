all: clean
	python setup.py build_ext --inplace

test: all
	python ../run.py C

clean:
	rm -f *.c
	rm -rf __pycache__
	rm -f *.so
	rm -rf build
