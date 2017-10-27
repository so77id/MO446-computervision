#FUNCTION
define cecho
    @echo "\033[92m$(1)\033[0m"
endef

# COMMANDS
WGET_COMMAND=wget
PYTHON_COMMAND=python3.6
PIP_COMMAND=pip3
MKDIR_COMMAND=mkdir
RM_COMMAND=rm
UNZIP_COMMAND=unzip
LS_COMMAND=ls

# FOLDERS
INPUT_FOLDER=./input
OUTPUT_FOLDER=./output
SRC_FOLDER=./src

ZIP_NAME=p4-images
INPUT_URL=http://www.recod.ic.unicamp.br/~mrodriguez/computer_vision/p4/$(ZIP_NAME).zip
INPUT_CACHE=$(INPUT_FOLDER)/$(ZIP_NAME).zip
INPUT_LIST=input_list.txt
IMG_PATH_EXPRESSION=$(INPUT_FOLDER)/$(ZIP_NAME)/%s.jpg

# MY_OUTPUT (OUR RESULTS)
ZIP_MY_OUTPUT=my-output
OUR_RESULTS_URL=http://www.ic.unicamp.br/~ra107396/MO446/P4/$(ZIP_MY_OUTPUT).zip

# SCRIPTS
CREATE_INDEX=index.py
REGIONS=segment.py
QUERIES=query.py

# SEGMENTATION PARAMS
NUMBER_OF_CLUSTERS=2
#DESCRIPTOR_NAME=classic
DESCRIPTOR_NAME=lbp
REGIONS_FOLDER=$(OUTPUT_FOLDER)/regions
REGION_FILE=$(OUTPUT_FOLDER)/regions.dic

# VOCABULARY PARAMS
NUMBER_OF_FEATURES=50
REGION_DESCRIPTORS_FILE=$(OUTPUT_FOLDER)/region_descriptors.dic
DESCRIPTORS_FILE=$(OUTPUT_FOLDER)/descriptors.dic
VISUAL_WORDS_FILE=$(OUTPUT_FOLDER)/visual_words.dic

#QUERY PARAMS
QUERY_IDS='beach_2+boat_5+cherry_3+pond_2+stHelens_2+sunset1_2+sunset2_2'
RESULT_SIZE=3
DISTANCE_FUNCTION=cosine
#distance choices can be: euclidean, cosine, correlation, and a few others
PLOT_RESULTS_FILE='output/query_results.png'

all: init_pipeline segment create_index query_results download_results

init_pipeline:
	$(call cecho,"[Start up configs]")

	$(call cecho,"- Installing dependencies")
	@$(PIP_COMMAND) install scikit-image

	$(call cecho,"- Creating input/output folders")
	@$(MKDIR_COMMAND) -p $(INPUT_FOLDER)
	@$(MKDIR_COMMAND) -p $(OUTPUT_FOLDER)

	$(call cecho,"- Downloading images")
	@$(WGET_COMMAND) $(INPUT_URL) -O $(INPUT_CACHE)

	$(call cecho,"- Unzip images")
	@$(UNZIP_COMMAND) $(INPUT_CACHE) -d $(INPUT_FOLDER)

	@$(LS_COMMAND) $(INPUT_FOLDER)/p4-images > $(INPUT_FOLDER)/$(INPUT_LIST)

	@$(RM_COMMAND) $(INPUT_CACHE)

download_results:
	$(call cecho,"[Download my-output]")
	@$(MKDIR_COMMAND) -p $(ZIP_MY_OUTPUT)
	@$(WGET_COMMAND) $(OUR_RESULTS_URL)
	@$(UNZIP_COMMAND) $(ZIP_MY_OUTPUT) -d $(ZIP_MY_OUTPUT)
	@$(RM_COMMAND) $(ZIP_MY_OUTPUT).zip

segment:
	$(call cecho,"[Segmenting images]")
	@$(PYTHON_COMMAND) $(SRC_FOLDER)/$(REGIONS) -i $(INPUT_FOLDER)/p4-images -k $(NUMBER_OF_CLUSTERS) -b 3 -sf $(REGION_FILE)

create_index:
	$(call cecho,"[Creating index]")
	@$(PYTHON_COMMAND) $(SRC_FOLDER)/$(CREATE_INDEX) -il $(INPUT_FOLDER)/$(INPUT_LIST) -if $(INPUT_FOLDER)/p4-images -ro $(REGION_DESCRIPTORS_FILE) -do $(DESCRIPTORS_FILE) -vwo $(VISUAL_WORDS_FILE) -rf $(REGION_FILE) -nf $(NUMBER_OF_FEATURES) -de $(DESCRIPTOR_NAME)

query_results:
	$(call cecho,"[Retrieving images]")
	@$(PYTHON_COMMAND) $(SRC_FOLDER)/$(QUERIES) -d $(DESCRIPTORS_FILE) -q $(QUERY_IDS) -s $(RESULT_SIZE) -f $(DISTANCE_FUNCTION) -i $(IMG_PATH_EXPRESSION) -o $(PLOT_RESULTS_FILE)
