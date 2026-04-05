cleanall:
	rm -rf data/tmp

interactive:
	qlogin -n 20 --mem-per-cpu=4096 --time=00:30:00 --job-name=interactive

main:
	python main.py