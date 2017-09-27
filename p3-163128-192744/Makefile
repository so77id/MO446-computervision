#FUNCTION
define cecho
    @echo "\033[92m$(1)\033[0m"
endef

# COMMANDS
WGET_COMMAND=wget
PYTHON_COMMAND=python3
MKDIR_COMMAND=mkdir
RM_COMMAND=rm
BASH_COMMAND=sh
UNZIP_COMMAND=unzip

# FOLDERS
INPUT_FOLDER=./input
ZIP_FOLDER=$(INPUT_FOLDER)/zip
OUTPUT_FOLDER=./output
SRC_FOLDER=./src




# URLs
TEMPLE_URL=http://vision.middlebury.edu/mview/data/data/temple.zip
TEMPLE_RING_URL=http://vision.middlebury.edu/mview/data/data/templeRing.zip
TEMPLE_SPARCE_RING_URL=http://vision.middlebury.edu/mview/data/data/templeSparseRing.zip

DINO_URL=http://vision.middlebury.edu/mview/data/data/dino.zip
DINO_RING_URL=http://vision.middlebury.edu/mview/data/data/dinoRing.zip
DINO_SPARCE_RING_URL=http://vision.middlebury.edu/mview/data/data/dinoSparseRing.zip

URL_DOWNLOAD=$(TEMPLE_SPARCE_RING_URL)

# ZIP names
ZIP_FILE=input.zip

#LISTS
INPUT_LIST=templeSR_ang.txt


# INPUTS
VIDEO_INPUT_1=p2-1.mp4

# OUTPUTS
VIDEO_OUTPUT=p2-2b.mp4

# SRCs
KP_REVIEW=key_points_review.py

# ARGS

all: kp_review

	@echo "Finish"

init_pipeline:
	$(call cecho,"[Start up configs]")
	$(call cecho,"- Creating input/output folders")
	@$(MKDIR_COMMAND) -p $(INPUT_FOLDER)
	@$(RM_COMMAND) -f -R $(INPUT_FOLDER)/*
	@$(MKDIR_COMMAND) -p $(ZIP_FOLDER)
	@$(MKDIR_COMMAND) -p $(OUTPUT_FOLDER)

	$(call cecho,"- Downloading video")
	@# Download temple
	@$(WGET_COMMAND) $(URL_DOWNLOAD) -O $(ZIP_FOLDER)/$(ZIP_FILE)
	@$(UNZIP_COMMAND) -j $(ZIP_FOLDER)/$(ZIP_FILE) -d $(INPUT_FOLDER)
	@$(RM_COMMAND) -R $(ZIP_FOLDER)

kp_review:
	$(call cecho,"[Key points review]")
	@$(PYTHON_COMMAND) $(SRC_FOLDER)/$(KP_REVIEW) -i $(INPUT_FOLDER)/$(INPUT_LIST) -if $(INPUT_FOLDER) -d



clean:
	$(call cecho,"[Clean files]")
	@$(RM_COMMAND) -f -R $(INPUT_FOLDER)/*

	@$(RM_COMMAND) -f -R $(OUTPUT_FOLDER)/*