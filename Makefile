LATEX=pdflatex --file-line-error --shell-escape --synctex=1
# This directory will be overwritten. Should be new.
BUILD_DIR=build
PANIC_PREFIX=$(BUILD_DIR)/panic/PANIC2021_HiggsBRs_JonasKunath
IMG_PATTERN=find img  \( -wholename "img/code/*" -o -wholename "img/extern/*" \)

PER_EVENT_DATA=code/tmp/per_event_data
HIGGSTABLES_CONFIG=code/data/higgstables-config.yaml
TIMESTAMPED_TABLES=code/tmp/timestamped_tables/$(shell date +%Y-%m-%d-%H%M%S)

TAB_FILES=$(BUILD_DIR)/files_tab.txt
PRESENTATION_FILES=$(BUILD_DIR)/files_latex_presentation.txt
PY_FILES=$(BUILD_DIR)/files_py.txt
IMG_FILES=$(BUILD_DIR)/files_img.txt


all : images presentation.pdf poster.pdf panic2021.pdf
tables-all : tables all

# $(call make_pdf, tex_folder)
define make_pdf
  cd $1; $(LATEX) --output-directory=../$(BUILD_DIR) $1.tex
  ln -sf $(BUILD_DIR)/$1.pdf
endef

%.pdf : $(BUILD_DIR)/files_latex_%.txt $(IMG_FILES)
	$(call make_pdf,$*)

presentation-only : $(PRESENTATION_FILES)
	$(call make_pdf,presentation)

images : preselection fit

preselection : $(TAB_FILES) $(PY_FILES) $(HIGGSTABLES_CONFIG)
	python3 code/preselection_plots.py code/data $(HIGGSTABLES_CONFIG)

fit : $(TAB_FILES) $(PY_FILES)
	python3 code/fit_and_plot.py code/data

small-toys : $(TAB_FILES) $(PY_FILES)
	python3 code/fit_and_plot.py code/data 500

panic : panic2021.pdf
	@# It might be necessary in /etc/ImageMagick-6/policy.xml to comment out
	@# <!--policy domain="coder" rights="none" pattern="PDF" /-->
	@mkdir -p dir $(dir $(PANIC_PREFIX))
	@cp $^ $(PANIC_PREFIX).pdf
	convert -density 100 $(PANIC_PREFIX).pdf $(PANIC_PREFIX).jpg
	convert -density 50 $(PANIC_PREFIX).pdf $(PANIC_PREFIX)-small.jpg

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
	find code/data -type f | sort > $@

$(BUILD_DIR)/files_latex_presentation.txt : $(shell find presentation)
	mkdir -p $(BUILD_DIR)
	find presentation -type f | sort > $@

$(BUILD_DIR)/files_latex_poster.txt : $(shell find poster)
	mkdir -p $(BUILD_DIR)
	find poster -type f | sort > $@

$(BUILD_DIR)/files_latex_panic2021.txt : $(shell find panic2021)
	mkdir -p $(BUILD_DIR)
	find poster -type f | sort > $@

$(PY_FILES) : $(shell find code -maxdepth 2 -name "*.py")
	mkdir -p $(BUILD_DIR)
	find code -maxdepth 2 -name "*.py" | sort > $@

$(IMG_FILES) : $(shell $(IMG_PATTERN))
	mkdir -p $(BUILD_DIR)
	$(IMG_PATTERN) -type f | sort > $@

# -----------------------------------------------------------------------------
#
# This block is only needed when building new data tables.
#
TABLES_INPUT:=tables_e1e1 tables_e2e2
TABLES_INPUT+=presel_e2e2
# presel_e1e1 not used in presentation.
TABLES_INPUT+=$(HIGGSTABLES_CONFIG)

tables : $(TABLES_INPUT)

tables_% : $(PER_EVENT_DATA)/z_to_% $(HIGGSTABLES_CONFIG)
	echo "$?"
	python3 -c "import higgstables; print(higgstables._version_info)"  # Ensure that we use a good python interpreter before building anything.
	mkdir -p $(TIMESTAMPED_TABLES)
	higgstables $< --config $(HIGGSTABLES_CONFIG) \
		--data_dir $(TIMESTAMPED_TABLES)/tables_$(shell (basename $<) | rev | cut -d'_' -f 1 | rev)
	cp -r $(TIMESTAMPED_TABLES)/* code/data/
	# We need to change the folder's timestamp (the content changed, but the file names stayed).
	touch code/data/$@

presel_% : $(PER_EVENT_DATA)/z_to_% $(HIGGSTABLES_CONFIG) code/preselection_make_tables.py
	mkdir -p $(TIMESTAMPED_TABLES)
	python3 code/preselection_make_tables.py $< $(HIGGSTABLES_CONFIG) $(TIMESTAMPED_TABLES)
	cp -r $(TIMESTAMPED_TABLES)/* code/data/
	touch code/data/$@

ERROR_MSG="ERROR:  Making new event count tables needs access to per-event rootfiles at $(PER_EVENT_DATA). Look into the init.sh options."
$(PER_EVENT_DATA)/z_to_% :
	# The rootfiles are large and thus only available on the LLR servers.
	@if ! [ -e $(PER_EVENT_DATA) ]; then echo $(ERROR_MSG) & exit 1; fi
