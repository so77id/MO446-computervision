# You must configure this Makefile to suit your solution

# COMMANDS
WGET_COMMAND=wget
PYTHON_COMMAND=python3
MOGRIFY_COMAND=convert
MKDIR_COMMAND=mkdir

# FOLDERS
INPUT_FOLDER=./input
OUTPUT_FOLDER=./output
SRC_FOLDER=./src

# INPUT FILES

IMAGE_INPUT_FILE=p0-1-0.png
IMAGE_INPUT_URL=https://cdn.pixabay.com/photo/2013/10/24/20/16/lago-bleu-200429_640.jpg

# OUTPUT FILES
# Q2
COLOR_PLANES_A_OUTPUT=p0-2-a-0.png
COLOR_PLANES_B_OUTPUT=p0-2-b-0.png
COLOR_PLANES_C_OUTPUT=p0-2-c-0.png
# Q3
REPLACEMENT_PIXELS_A_OUTPUT=p0-3-0.png
REPLACEMENT_PIXELS_B_OUTPUT=p0-3-1.png
# Q4
AG_OPERATIONS_B_OUTPUT=p0-4-b-0.png
AG_OPERATIONS_C_0_OUTPUT=p0-4-c-0.png
AG_OPERATIONS_C_1_OUTPUT=p0-4-c-1.png
# Q5
NOISE_A_OUTPUT=p0-5-a-0.png
NOISE_B_OUTPUT=p0-5-b-0.png


# SRC FILES
RESIZE_FILE=resize.py
COLOR_PLANES_FILE=color_planes.py
REPLACEMENT_PIXELS_FILE=replacement_pixels.py
ARITHMETIC_AND_GEOMETRIC_OPERATIONS_FILE=arithmetic_and_geometric_operations.py
NOISE_FILE=noise.py


all: download_input color_planes replacement_pixels arithmetic_and_geometric_operations noise
	@echo "Finish"

download_input:
	@echo "[Downloading inputs]"
	@$(MKDIR_COMMAND) -p $(INPUT_FOLDER)
	@$(MKDIR_COMMAND) -p $(OUTPUT_FOLDER)
	@$(WGET_COMMAND) $(IMAGE_INPUT_URL) -O $(INPUT_FOLDER)/$(IMAGE_INPUT_FILE)
	@$(PYTHON_COMMAND) $(SRC_FOLDER)/$(RESIZE_FILE) -i $(INPUT_FOLDER)/$(IMAGE_INPUT_FILE) -o $(INPUT_FOLDER)/$(IMAGE_INPUT_FILE)

color_planes:
	@echo "[Running color planes (Q-2)]"
	@$(PYTHON_COMMAND) $(SRC_FOLDER)/$(COLOR_PLANES_FILE) -i $(INPUT_FOLDER)/$(IMAGE_INPUT_FILE) -oa $(OUTPUT_FOLDER)/$(COLOR_PLANES_A_OUTPUT) -ob $(OUTPUT_FOLDER)/$(COLOR_PLANES_B_OUTPUT) -oc $(OUTPUT_FOLDER)/$(COLOR_PLANES_C_OUTPUT)

replacement_pixels:
	@echo "[Running replacement pixels (Q-3)]"
	@$(PYTHON_COMMAND) $(SRC_FOLDER)/$(REPLACEMENT_PIXELS_FILE) -i $(INPUT_FOLDER)/$(IMAGE_INPUT_FILE) -oa $(OUTPUT_FOLDER)/$(REPLACEMENT_PIXELS_A_OUTPUT) -ob $(OUTPUT_FOLDER)/$(REPLACEMENT_PIXELS_B_OUTPUT)

arithmetic_and_geometric_operations:
	@echo "[Running arithmetic and geometric operations (Q-4)]"
	@$(PYTHON_COMMAND) $(SRC_FOLDER)/$(ARITHMETIC_AND_GEOMETRIC_OPERATIONS_FILE) -i $(INPUT_FOLDER)/$(IMAGE_INPUT_FILE) -ob $(OUTPUT_FOLDER)/$(AG_OPERATIONS_B_OUTPUT) -oc0 $(OUTPUT_FOLDER)/$(AG_OPERATIONS_C_0_OUTPUT) -oc1  $(OUTPUT_FOLDER)/$(AG_OPERATIONS_C_1_OUTPUT)

noise:
	@echo "[Running noise (Q-5)]"
	@$(PYTHON_COMMAND) $(SRC_FOLDER)/$(NOISE_FILE) -i $(INPUT_FOLDER)/$(IMAGE_INPUT_FILE) -oa $(OUTPUT_FOLDER)/$(NOISE_A_OUTPUT) -ob $(OUTPUT_FOLDER)/$(NOISE_B_OUTPUT)

.PHONY: all download_input color_planes replacement_pixels arithmetic_and_geometric_operations noise
