LATEX=latexmk -pdf -pdflatex="pdflatex --file-line-error --shell-escape --synctex=1" -use-make
PLOT_FORMAT=pdf

# Some paths that will be used.
BUILD_DIR=build
PLOT_DATA=code/data
PLOT_DIR=img/code
RAW_DATA_DIR=data
PER_EVENT_DATA_DIR=code/tmp/per_event_data
HIGGSTABLES_CONFIG=$(RAW_DATA_DIR)/higgstables-config.yaml
SETTINGS=code/settings

STANDARD_SETTINGS:=$(SETTINGS)/HIGGS_BR $(SETTINGS)/POLARIZATION
STANDARD_SETTINGS+=$(SETTINGS)/SIGNAL_SCALER $(SETTINGS)/LUMINOSITY
# helper_data does the data_set reading which is always needed.
STANDARD_SETTINGS+=code/helper_data.py

PRESEL_STEP_CSVS=$(patsubst $(RAW_DATA_DIR)/presel_e2e2/eLpR/step_%.csv,$(PLOT_DATA)/presel_e2e2_step%.csv,$(wildcard $(RAW_DATA_DIR)/presel_e2e2/eLpR/step_*.csv))
PLOTS_PRESENTATION:=$(PLOT_DIR)/intro_sample_counts.$(PLOT_FORMAT) $(PLOT_DIR)/intro_signal_composition_per_category.$(PLOT_FORMAT)
PLOTS_PRESENTATION+=$(PLOT_DIR)/intro_signal_composition_per_category_w_bkg.$(PLOT_FORMAT)
PLOTS_PRESENTATION+=$(PLOT_DIR)/intro_category_counts.$(PLOT_FORMAT) $(PLOT_DIR)/intro_category_counts_w_bkg.$(PLOT_FORMAT)
PLOTS_PRESENTATION+=$(PLOT_DIR)/expected_counts_matrix_bkg_e2e2.$(PLOT_FORMAT) $(PLOT_DIR)/probability_matrix.$(PLOT_FORMAT)
PLOTS_PRESENTATION+=$(PLOT_DIR)/presel_e2e2_eff_0.$(PLOT_FORMAT)
PLOTS_PRESENTATION+=$(patsubst $(PLOT_DATA)/presel_e2e2_step%.csv,$(PLOT_DIR)/presel_e2e2_eff_%.$(PLOT_FORMAT),$(PRESEL_STEP_CSVS))
PLOTS_PRESENTATION+=$(patsubst $(PLOT_DATA)/presel_e2e2_step%.csv,$(PLOT_DIR)/presel_e2e2_%.$(PLOT_FORMAT),$(PRESEL_STEP_CSVS))
PLOTS_PRESENTATION+=$(PLOT_DIR)/correlations_default.$(PLOT_FORMAT)
PLOTS_PRESENTATION+=$(PLOT_DIR)/br_estimates_default.$(PLOT_FORMAT) $(PLOT_DIR)/br_estimates_changed_bbww.$(PLOT_FORMAT)
PLOTS_PRESENTATION+=$(PLOT_DIR)/bias_table_default.tex
PLOTS_PRESENTATION+=$(PLOT_DIR)/toys_default_bb.$(PLOT_FORMAT) $(PLOT_DIR)/toys_default_az.$(PLOT_FORMAT)
PLOTS_PRESENTATION+=$(PLOT_DIR)/toys_changed_bbww_bb.$(PLOT_FORMAT) $(PLOT_DIR)/toys_changed_bbww_ww.$(PLOT_FORMAT)
PLOTS_PRESENTATION+=$(PLOT_DIR)/comparison_with_others.$(PLOT_FORMAT)
PLOTS_PRESENTATION+=$(PLOT_DIR)/comparison_br_scenarios_1.$(PLOT_FORMAT) $(PLOT_DIR)/comparison_br_scenarios_2.$(PLOT_FORMAT)
PLOTS_PRESENTATION+=$(PLOT_DIR)/comparison_polarizations.$(PLOT_FORMAT)
PLOTS_PRESENTATION+=$(PLOT_DIR)/comparison_signal_scaler_partial.$(PLOT_FORMAT) $(PLOT_DIR)/comparison_signal_scaler.$(PLOT_FORMAT)
PLOTS_PRESENTATION+=$(PLOT_DIR)/presel_e2e2_for_proceedings.$(PLOT_FORMAT)

all : presentation.pdf
tables-all : tables all

.PHONY: clean
clean :
	rm -f *.pdf
	rm -rf $(BUILD_DIR)
	rm -rf img/code
	rm -rf code/.venv
	rm -rf code/tmp/timestamped_tables
	@echo INFO: $(PER_EVENT_DATA_DIR) can be expensive to obtain and is thus not removed.

# -----------------------------------------------------------------------------
#
# Latex presentation and proceedings.
#
# $(call make_pdf, tex_folder)
define make_pdf
  cd $1; $(LATEX) --output-directory=../$(BUILD_DIR) $1.tex
  ln -sf $(BUILD_DIR)/$1.pdf
  @# Update the pdf even if latex decided it does not have to be rebuilt.
  @touch $1.pdf
endef

presentation.pdf : presentation/presentation.tex presentation/beamerx.sty presentation/jonas.sty \
        $(wildcard presentation/backup/*.tex) $(wildcard presentation/*.tex) $(PLOTS_PRESENTATION)
	@echo "DEBUG: All prerequisits for" $@ "---" $?
	$(call make_pdf,presentation)

# -----------------------------------------------------------------------------
#
# Prepare the data for plots.
#
$(PLOT_DATA)/counts_%.csv: code/data_intro.py $(RAW_DATA_DIR) $(STANDARD_SETTINGS) $(SETTINGS)/PROCESS_GROUPS
	@echo "Reproduce plotting data:" $@
	@mkdir -p $(PLOT_DATA)
	python3 $(word 1,$^) $@ $(filter-out $<,$^)

$(PLOT_DATA)/presel_e2e2_step%.csv : code/data_presel.py $(RAW_DATA_DIR)/presel_e2e2/eLpL/step_%.csv $(HIGGSTABLES_CONFIG) $(STANDARD_SETTINGS) $(SETTINGS)/PROCESS_GROUPS
	@echo "Reproduce plotting data:" $@
	@mkdir -p $(PLOT_DATA)
	@if [ -f $@ ]; then rm $@; fi
	python3 $(word 1,$^) $@ $(filter-out $<,$^)

$(PLOT_DATA)/presel_e2e2_eff_pur.csv : code/data_presel.py $(RAW_DATA_DIR)/presel_e2e2/eLpL/step_1.csv $(HIGGSTABLES_CONFIG) $(STANDARD_SETTINGS) $(SETTINGS)/PROCESS_GROUPS
	@echo "Reproduce plotting data:" $@
	@mkdir -p $(PLOT_DATA)
	@if [ -f $@ ]; then rm $@; fi
	python3 $(word 1,$^) $@ $(filter-out $<,$^)

$(PLOT_DATA)/fit_default.csv : code/data_fit.py $(RAW_DATA_DIR) $(STANDARD_SETTINGS)
	@echo "Reproduce plotting data:" $@
	@mkdir -p $(PLOT_DATA)
	python3 $(word 1,$^) $@ $(filter-out $<,$^)

$(PLOT_DATA)/fit_changed_%.csv : code/data_fit.py $(RAW_DATA_DIR) $(STANDARD_SETTINGS) $(SETTINGS)/HIGGS_BR_CHANGED_%
	@echo "Reproduce plotting data:" $@
	@mkdir -p $(PLOT_DATA)
	python3 $(word 1,$^) $@ $(filter-out $<,$^)

$(PLOT_DATA)/toys_default.csv : code/data_toys.py $(RAW_DATA_DIR) $(STANDARD_SETTINGS)
	@echo "Reproduce plotting data:" $@
	@mkdir -p $(PLOT_DATA)
	python3 $(word 1,$^) $@ $(filter-out $<,$^)

# DO NOT DELETE THE TOY FILES!
# https://unix.stackexchange.com/questions/517190/what-causes-make-to-delete-intermediate-files
.PRECIOUS: $(PLOT_DATA)/toys_changed_%.csv
$(PLOT_DATA)/toys_changed_%.csv : code/data_toys.py $(RAW_DATA_DIR) $(STANDARD_SETTINGS) $(SETTINGS)/HIGGS_BR_CHANGED_%
	@echo "Reproduce plotting data:" $@
	@mkdir -p $(PLOT_DATA)
	python3 $(word 1,$^) $@ $(filter-out $<,$^)

.PRECIOUS: $(PLOT_DATA)/comparison_%.csv
$(PLOT_DATA)/comparison_%.csv : code/data_comparison_%.py $(RAW_DATA_DIR) $(STANDARD_SETTINGS)
	@echo "Reproduce plotting data:" $@
	@mkdir -p $(PLOT_DATA)
	python3 $(word 1,$^) $@ $(filter-out $<,$^)

# -----------------------------------------------------------------------------
#
# Create the plots.
#
$(PLOT_DIR)/intro_sample_counts.$(PLOT_FORMAT) : code/plot_scripts/intro_sample_counts.py $(SETTINGS)/FANCY_NAMES $(PLOT_DATA)/counts_e1e1.csv $(PLOT_DATA)/counts_e2e2.csv
	@mkdir -p $(PLOT_DIR)
	python3 $(word 1,$^) $@ $(filter-out $<,$^)

$(PLOT_DIR)/intro_category_count%.$(PLOT_FORMAT) : code/plot_scripts/intro_category_counts.py $(SETTINGS)/FANCY_NAMES $(PLOT_DATA)/counts_e2e2.csv
	@mkdir -p $(PLOT_DIR)
	python3 $(word 1,$^) $@ $(filter-out $<,$^)

$(PLOT_DIR)/intro_signal_composition_per_cat%.$(PLOT_FORMAT) : code/plot_scripts/intro_category_counts.py $(SETTINGS)/FANCY_NAMES $(PLOT_DATA)/counts_e2e2.csv
	@mkdir -p $(PLOT_DIR)
	python3 $(word 1,$^) $@ $(filter-out $<,$^)

$(PLOT_DIR)/expected_counts_matrix_%.$(PLOT_FORMAT) : code/plot_scripts/expected_counts_matrix.py $(SETTINGS)/FANCY_NAMES $(PLOT_DATA)/counts_e2e2.csv
	@mkdir -p $(PLOT_DIR)
	python3 $(word 1,$^) $@ $(filter-out $<,$^)

$(PLOT_DIR)/probability_matrix.$(PLOT_FORMAT) : code/plot_scripts/expected_counts_matrix.py $(SETTINGS)/FANCY_NAMES $(PLOT_DATA)/counts_e2e2.csv
	@mkdir -p $(PLOT_DIR)
	python3 $(word 1,$^) $@ $(filter-out $<,$^)

$(PLOT_DIR)/presel_e2e2_%.$(PLOT_FORMAT) : code/plot_scripts/presel_e2e2.py $(RAW_DATA_DIR)/presel_e2e2/step_info.csv $(PRESEL_STEP_CSVS)
	@mkdir -p $(PLOT_DIR)
	python3 $(word 1,$^) $@ $(filter-out $<,$^)

$(PLOT_DIR)/presel_e2e2_eff_%.$(PLOT_FORMAT) : code/plot_scripts/presel_e2e2_eff.py $(PLOT_DATA)/presel_e2e2_eff_pur.csv
	@mkdir -p $(PLOT_DIR)
	python3 $(word 1,$^) $@ $(filter-out $<,$^)

$(PLOT_DIR)/presel_e2e2_for_proceedings.$(PLOT_FORMAT) : code/plot_scripts/presel_e2e2_for_proceedings.py $(PLOT_DATA)/presel_e2e2_eff_pur.csv $(RAW_DATA_DIR)/presel_e2e2/step_info.csv $(PRESEL_STEP_CSVS)
	@mkdir -p $(PLOT_DIR)
	python3 $(word 1,$^) $@ $(filter-out $<,$^)

$(PLOT_DIR)/correlations_%.$(PLOT_FORMAT) : code/plot_scripts/correlations.py $(SETTINGS)/FANCY_NAMES $(PLOT_DATA)/fit_%.csv
	@mkdir -p $(PLOT_DIR)
	python3 $(word 1,$^) $@ $(filter-out $<,$^)

$(PLOT_DIR)/br_estimates_%.$(PLOT_FORMAT) : code/plot_scripts/br_estimates.py $(SETTINGS)/FANCY_NAMES $(PLOT_DATA)/fit_%.csv
	@mkdir -p $(PLOT_DIR)
	python3 $(word 1,$^) $@ $(filter-out $<,$^)

$(PLOT_DIR)/bias_table_%.tex : code/plot_scripts/bias_table.py $(SETTINGS)/FANCY_NAMES $(PLOT_DATA)/fit_%.csv
	@mkdir -p $(PLOT_DIR)
	python3 $(word 1,$^) $@ $(filter-out $<,$^)

$(PLOT_DIR)/toys_default_%.$(PLOT_FORMAT) : code/plot_scripts/toys.py $(SETTINGS)/FANCY_NAMES $(PLOT_DATA)/toys_default.csv  $(PLOT_DATA)/fit_default.csv
	@mkdir -p $(PLOT_DIR)
	python3 $(word 1,$^) $@ $(filter-out $<,$^)

$(PLOT_DIR)/toys_changed_%_bb.$(PLOT_FORMAT) : code/plot_scripts/toys.py $(SETTINGS)/FANCY_NAMES $(PLOT_DATA)/toys_changed_%.csv  $(PLOT_DATA)/fit_changed_%.csv
	@mkdir -p $(PLOT_DIR)
	python3 $(word 1,$^) $@ $(filter-out $<,$^)

$(PLOT_DIR)/toys_changed_%_ww.$(PLOT_FORMAT) $(PLOT_DIR)/toys_changed_%_gg.$(PLOT_FORMAT) $(PLOT_DIR)/toys_changed_%_e3e3.$(PLOT_FORMAT) \
    $(PLOT_DIR)/toys_changed_%_cc.$(PLOT_FORMAT) $(PLOT_DIR)/toys_changed_%_zz.$(PLOT_FORMAT) $(PLOT_DIR)/toys_changed_%_aa.$(PLOT_FORMAT) \
	$(PLOT_DIR)/toys_changed_%_az.$(PLOT_FORMAT) $(PLOT_DIR)/toys_changed_%_e2e2.$(PLOT_FORMAT) : $(PLOT_DIR)/toys_changed_%_bb.$(PLOT_FORMAT)

$(PLOT_DIR)/comparison_with_others.$(PLOT_FORMAT) : code/plot_scripts/comparison_with_others.py $(SETTINGS)/FANCY_NAMES $(SETTINGS)/LUMINOSITY $(PLOT_DATA)/fit_default.csv
	@mkdir -p $(PLOT_DIR)
	python3 $(word 1,$^) $@ $(filter-out $<,$^)

$(PLOT_DIR)/comparison_br_scenarios_1.$(PLOT_FORMAT) : code/plot_scripts/comparison_br_scenarios.py $(SETTINGS)/FANCY_NAMES \
        $(PLOT_DATA)/fit_default.csv $(PLOT_DATA)/fit_changed_bbww.csv
	@mkdir -p $(PLOT_DIR)
	python3 $(word 1,$^) $@ $(filter-out $<,$^)

$(PLOT_DIR)/comparison_br_scenarios_2.$(PLOT_FORMAT) : code/plot_scripts/comparison_br_scenarios.py $(SETTINGS)/FANCY_NAMES \
        $(PLOT_DATA)/fit_default.csv $(PLOT_DATA)/fit_changed_bbww.csv $(PLOT_DATA)/fit_changed_e2e2.csv
	@mkdir -p $(PLOT_DIR)
	python3 $(word 1,$^) $@ $(filter-out $<,$^)

$(PLOT_DIR)/comparison_signal_scaler_partial.$(PLOT_FORMAT) : $(PLOT_DIR)/comparison_signal_scaler.$(PLOT_FORMAT)

$(PLOT_DIR)/comparison_%.$(PLOT_FORMAT) : code/plot_scripts/comparison_%.py $(SETTINGS)/FANCY_NAMES $(PLOT_DATA)/comparison_%.csv
	@mkdir -p $(PLOT_DIR)
	python3 $(word 1,$^) $@ $(filter-out $<,$^)

# -----------------------------------------------------------------------------
#
# This block is only needed when building new data tables.
#
TABLES_INPUT:=tables_e1e1 tables_e2e2
TABLES_INPUT+=presel_e2e2
# presel_e1e1 not used in presentation.
TABLES_INPUT+=$(HIGGSTABLES_CONFIG)
TIMESTAMPED_TABLES=code/tmp/timestamped_tables/$(shell date +%Y-%m-%d-%H%M%S)


tables : $(TABLES_INPUT)

tables_% : $(PER_EVENT_DATA_DIR)/z_to_% $(HIGGSTABLES_CONFIG)
	echo "$?"
	python3 -c "import higgstables; print(higgstables._version_info)"  # Ensure that we use a good python interpreter before building anything.
	mkdir -p $(TIMESTAMPED_TABLES)
	higgstables $< --config $(HIGGSTABLES_CONFIG) \
		--data_dir $(TIMESTAMPED_TABLES)/tables_$(shell (basename $<) | rev | cut -d'_' -f 1 | rev)
	cp -r $(TIMESTAMPED_TABLES)/* $(RAW_DATA_DIR)/
	# We need to change the folder's timestamp (the content changed, but the file names stayed).
	touch $(RAW_DATA_DIR)/$@ & touch $(RAW_DATA_DIR)

presel_% : $(PER_EVENT_DATA_DIR)/z_to_% $(HIGGSTABLES_CONFIG) code/data_presel_make_tables.py
	mkdir -p $(TIMESTAMPED_TABLES)
	python3 code/data_presel_make_tables.py $< $(HIGGSTABLES_CONFIG) $(TIMESTAMPED_TABLES)
	cp -r $(TIMESTAMPED_TABLES)/* $(RAW_DATA_DIR)/
	touch $(RAW_DATA_DIR)/$@

ERROR_MSG="ERROR:  Making new event count tables needs access to per-event rootfiles at $(PER_EVENT_DATA_DIR). Look into the init.sh options."
$(PER_EVENT_DATA_DIR)/z_to_% :
	# The rootfiles are large and thus only available on the LLR servers.
	@if ! [ -e $(PER_EVENT_DATA_DIR) ]; then echo $(ERROR_MSG) & exit 1; fi
