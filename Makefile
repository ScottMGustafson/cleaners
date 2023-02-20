sources = cleaners

.PHONY: test format lint unittest coverage pre-commit clean open-coverage
test: format lint unittest

format:
	isort $(sources) tests
	black $(sources) tests

lint:
	flake8 $(sources) tests

unittest:
	pytest tests --doctest-modules

nbtest:
	pytest --nbmake notebooks

coverage:
	pytest tests --exitfirst --doctest-modules --cov=$(sources) --cov-branch --cov-report=html --cov-report=term-missing --run-regression --nbmake notebooks

open-coverage:
	xdg-open htmlcov/index.html

pre-commit:
	pre-commit run --all-files

clean:
	rm -rf .pytest_cache
	rm -rf *.egg-info
	rm -rf .tox dist site
	rm -rf coverage.xml .coverage
