PYTHON ?= python
OUT ?= output

.PHONY: reproduce fetch generate verify paper theory
reproduce: generate
	$(PYTHON) -B verify.py --out "$(OUT)"

fetch:
	$(PYTHON) -B fetch_data.py

generate: fetch
	$(PYTHON) -B reproduce.py --out "$(OUT)"

verify:
	$(PYTHON) -B verify.py --out "$(OUT)"

theory:
	$(PYTHON) -B -m unittest discover -v
	$(PYTHON) -B -m theory.verify_crc_theory
	$(PYTHON) -B -m theory.verify_score_theory
	$(PYTHON) -B -m theory.verify_rl_theory

paper: reproduce theory
	$(PYTHON) -B -m paper.tables --out "$(OUT)"
	$(PYTHON) -B -m paper.figures --out "$(OUT)"
