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
#ZIP_FOLDER=$(INPUT_FOLDER)/zip
OUTPUT_FOLDER=./output
SRC_FOLDER=./src




# URLs
TEMPLE_URL=http://vision.middlebury.edu/mview/data/data/temple.zip
TEMPLE_RING_URL=http://vision.middlebury.edu/mview/data/data/templeRing.zip
TEMPLE_SPARCE_RING_URL=http://vision.middlebury.edu/mview/data/data/templeSparseRing.zip

DINO_URL=http://vision.middlebury.edu/mview/data/data/dino.zip
DINO_RING_URL=http://vision.middlebury.edu/mview/data/data/dinoRing.zip
DINO_SPARCE_RING_URL=http://vision.middlebury.edu/mview/data/data/dinoSparseRing.zip

# VIDEOS URLs
P3_1_URL=http://www.recod.ic.unicamp.br/~mrodriguez/computer_vision/p3/p3-1.mp4
P3_2_URL=http://www.recod.ic.unicamp.br/~mrodriguez/computer_vision/p3/p3-2.mp4
P3_3_URL=http://www.recod.ic.unicamp.br/~mrodriguez/computer_vision/p3/p3-3.mp4

#URL_DOWNLOAD=$(TEMPLE_SPARCE_RING_URL)

# ZIP names
#ZIP_FILE=input.zip

#LISTS
#INPUT_LIST=templeSR_ang.txt

#FileNames
P3_1_FILE=p3-1.mp4
P3_2_FILE=p3-2.mp4
P3_3_FILE=p3-3.mp4


# INPUTS
VIDEO_INPUT=$(P3_3_FILE)

# OUTPUTS
VIDEO_OUTPUT=$(P3_3_FILE)

# SRCs
KP_REVIEW=key_points_review.py
OF_REVIEW=optical_flow_review.py

# ARGS

all: of_review

	@echo "Finish"

init_pipeline:
	$(call cecho,"[Start up configs]")
	$(call cecho,"- Creating input/output folders")
	@$(MKDIR_COMMAND) -p $(INPUT_FOLDER)
	@$(MKDIR_COMMAND) -p $(OUTPUT_FOLDER)

	$(call cecho,"- Downloading videos")
	@# Download videos
	@$(WGET_COMMAND) $(P3_1_URL) -O $(INPUT_FOLDER)/$(P3_1_FILE)
	@$(WGET_COMMAND) $(P3_2_URL) -O $(INPUT_FOLDER)/$(P3_2_FILE)
	@$(WGET_COMMAND) $(P3_3_URL) -O $(INPUT_FOLDER)/$(P3_3_FILE)

kp_review:
	$(call cecho,"[Key points review]")
	@$(PYTHON_COMMAND) $(SRC_FOLDER)/$(KP_REVIEW) -i $(INPUT_FOLDER)/$(VIDEO_INPUT) -d

of_review:
	$(call cecho,"[Optical flow review]")
	$(PYTHON_COMMAND) $(SRC_FOLDER)/$(OF_REVIEW) -i $(INPUT_FOLDER)/$(VIDEO_INPUT) -o $(OUTPUT_FOLDER)/$(VIDEO_OUTPUT) -d



clean:
	$(call cecho,"[Clean files]")
	@$(RM_COMMAND) -f -R $(INPUT_FOLDER)/*

	@$(RM_COMMAND) -f -R $(OUTPUT_FOLDER)/*