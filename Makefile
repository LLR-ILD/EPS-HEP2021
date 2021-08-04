LATEX=pdflatex --file-line-error --shell-escape --synctex=1
# This directory will be overwritten. Should be new.
BUILD_DIR=build
IMG_PATTERN=find img  \( -wholename "img/code/*" -o -wholename "img/extern/*" \)

PER_EVENT_DATA=code/tmp/per_event_data
HIGGSTABLES_CONFIG=code/data/higgstables-config.yaml
TIMESTAMPED_TABLES=code/tmp/timestamped_tables/$(shell date +%Y-%m-%d-%H%M%S)

TAB_FILES=$(BUILD_DIR)/files_tab.txt
TEX_FILES=$(BUILD_DIR)/files_tex.txt
PY_FILES=$(BUILD_DIR)/files_py.txt
IMG_FILES=$(BUILD_DIR)/files_img.txt


all : images presentation.pdf
tables-all : tables all

presentation.pdf : presentation.tex $(TEX_FILES) $(IMG_FILES)
	$(LATEX) --output-directory=$(BUILD_DIR) presentation.tex
	ln -sf $(BUILD_DIR)/presentation.pdf

presentation-only : presentation.tex $(TEX_FILES)
	$(LATEX) --output-directory=$(BUILD_DIR) presentation.tex
	ln -sf $(BUILD_DIR)/presentation.pdf

images : preselection fit

preselection : $(TAB_FILES) $(PY_FILES) $(HIGGSTABLES_CONFIG)
	python3 code/preselection_plots.py code/data $(HIGGSTABLES_CONFIG)

fit : $(TAB_FILES) $(PY_FILES)
	python3 code/fit_and_plot.py code/data

small-toys : $(TAB_FILES) $(PY_FILES)
	python3 code/fit_and_plot.py code/data 500

.PHONY: clean
clean :
	rm -f presentation.pdf
	rm -f $(TAB_FILES) $(TEX_FILES) $(PY_FILES) $(IMG_FILES)
	rm -f $(BUILD_DIR)/presentation.*
	rmdir $(BUILD_DIR)

	rm -rf img/code
	rm -rf code/.venv
	rm -rf code/tmp/timestamped_tables
	@echo INFO: $(PER_EVENT_DATA) can be expensive to obtain and is thus not removed.

$(TAB_FILES) : $(shell find code/data)
	mkdir -p $(BUILD_DIR)
	find code/data -type f | sort > $(TAB_FILES)

$(TEX_FILES) : $(shell find presentation)
	mkdir -p $(BUILD_DIR)
	find presentation -type f | sort > $(TEX_FILES)

$(PY_FILES) : $(shell find code -maxdepth 2 -name "*.py")
	mkdir -p $(BUILD_DIR)
	find code -maxdepth 2 -name "*.py" | sort > $(PY_FILES)

$(IMG_FILES) : $(shell $(IMG_PATTERN))
	mkdir -p $(BUILD_DIR)
	$(IMG_PATTERN) -type f | sort > $(IMG_FILES)

# -----------------------------------------------------------------------------
#
# This block is only needed when building new data tables.
#
TABLES_INPUT:=code/data/tables_e1e1 code/data/tables_e2e2
TABLES_INPUT+=code/data/presel_e2e2
# presel_e1e1 not used in presentation.
TABLES_INPUT+=$(HIGGSTABLES_CONFIG)

tables : $(TABLES_INPUT)

code/data/tables_* : $(PER_EVENT_DATA)/z_to_* $(HIGGSTABLES_CONFIG)
	echo "$?"
	python3 -c "import higgstables; print(higgstables._version_info)"  # Ensure that we use a good python interpreter before building anything.
	mkdir -p $(TIMESTAMPED_TABLES)
	higgstables $< --config $(HIGGSTABLES_CONFIG) \
		--data_dir $(TIMESTAMPED_TABLES)/tables_$(shell (basename $<) | rev | cut -d'_' -f 1 | rev)
	cp -r $(TIMESTAMPED_TABLES)/* code/data/
	# We need to change the folder's timestamp (the content changed, but the file names stayed).
	touch $@

code/data/presel_* : $(PER_EVENT_DATA)/z_to_* $(HIGGSTABLES_CONFIG) code/preselection_make_tables.py
	mkdir -p $(TIMESTAMPED_TABLES)
	python3 code/preselection_make_tables.py $< $(HIGGSTABLES_CONFIG) $(TIMESTAMPED_TABLES)
	cp -r $(TIMESTAMPED_TABLES)/* code/data/
	touch $@

$(PER_EVENT_DATA)/% :
ifeq ($(shell test -e $(PER_EVENT_DATA) || echo "not found"),not found)
	# The rootfiles are large and thus only available on the LLR servers.
	$(error Making new event count tables needs access to per-event rootfiles at $(PER_EVENT_DATA). Look into the init.sh options)
endif
