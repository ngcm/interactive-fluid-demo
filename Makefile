all: clean
	cd csim; python setup.py build_ext --inplace

test: all
	python run.py

clean:
	rm -f csim/*.c
	rm -rf csim/__pycache__
	rm -f csim/*.so
	rm -rf csim/build
