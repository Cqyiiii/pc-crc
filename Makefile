PYTHON ?= python
OUT ?= output

.PHONY: reproduce fetch generate verify
reproduce: generate verify

fetch:
	$(PYTHON) -B fetch_data.py

generate: fetch
	$(PYTHON) -B reproduce.py --out "$(OUT)"

verify:
	$(PYTHON) -B verify.py --out "$(OUT)"
