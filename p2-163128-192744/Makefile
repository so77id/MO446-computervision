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

# FOLDERS
INPUT_FOLDER=./input
OUTPUT_FOLDER=./output
MY_OUTPUT_FOLDER=./my-output
SRC_FOLDER=./src
BENCHMARK_FOLDER=$(INPUT_FOLDER)/benchmark
SRC_LIST_FOLDER=lists


OUTPUT_CORRECTED_FOLDER=$(OUTPUT_FOLDER)/corrected
OUTPUT_ORIGINAL_FOLDER=$(OUTPUT_FOLDER)/original
OUTPUT_COMPOSED_FOLDER=$(OUTPUT_FOLDER)/composed



# URLs
VIDEO_INPUT_URL_1=http://www.recod.ic.unicamp.br/~mrodriguez/computer_vision/p2/inputs/p2-1.mp4
VIDEO_INPUT_URL_2=http://www.recod.ic.unicamp.br/~mrodriguez/computer_vision/p2/inputs/p2-2.mp4
VIDEO_INPUT_URL_2B=http://www.recod.ic.unicamp.br/~mrodriguez/computer_vision/p2/inputs/p2-2b.mp4
VIDEO_INPUT_URL_3=http://www.recod.ic.unicamp.br/~mrodriguez/computer_vision/p2/inputs/p2-3.mp4
VIDEO_INPUT_URL_4=http://www.recod.ic.unicamp.br/~mrodriguez/computer_vision/p2/inputs/p2-4.mp4


# URLs my-outputs
VIDEO_OUTPUT_URL_1=http://www.recod.ic.unicamp.br/~mrodriguez/computer_vision/p2/my-output/p2-com-1.mp4
VIDEO_OUTPUT_URL_2=http://www.recod.ic.unicamp.br/~mrodriguez/computer_vision/p2/my-output/p2-com-2.mp4
VIDEO_OUTPUT_URL_2SIFT=http://www.recod.ic.unicamp.br/~mrodriguez/computer_vision/p2/my-output/p2-com-2-sift.mp4
VIDEO_OUTPUT_URL_3=http://www.recod.ic.unicamp.br/~mrodriguez/computer_vision/p2/my-output/p2-com-3.mp4
VIDEO_OUTPUT_URL_4=http://www.recod.ic.unicamp.br/~mrodriguez/computer_vision/p2/my-output/p2-com-4.mp4


BENCHMARK_IMAGE_URL=http://www.recod.ic.unicamp.br/~mrodriguez/computer_vision/p2/inputs/p2-0.jpg
BENCHMARK_SIZE_LIST_URL=https://gist.githubusercontent.com/so77id/1f912253f3075b812017bd0a3583efc1/raw/605585442a53950daee3286bc704a92c1a7b405c/MO446p1-size-list.txt

#LISTS
BENCHMARK_SIZE_LIST=benchmark_size_list.txt
BENCHMARK_IMAGE_LIST=becnhmark_image_list.txt
BENCHMARK_PATTERN_NAME=p2_bench
BENCHMARK_PATTERN_EXTENSION=png

# INPUTS
VIDEO_INPUT_1=p2-1.mp4
VIDEO_INPUT_2=p2-2.mp4
VIDEO_INPUT_2B=p2-2b.mp4
VIDEO_INPUT_3=p2-3.mp4
VIDEO_INPUT_4=p2-4.mp4

BENCHMARK_IMAGE_FILE=p2-0.jpg

# OUTPUTS
VIDEO_OUTPUT=p2-2b.mp4
VIDEO_COMPARER_OUTPUT=p2-com-2b.mp4


VIDEO_OUTPUT_1=p2-com-1-affine.mp4
VIDEO_OUTPUT_2=p2-com-2-affine.mp4
VIDEO_OUTPUT_2_SIFT=p2-com-2-our-sift.mp4
VIDEO_OUTPUT_3=p2-com-3-affine.mp4
VIDEO_OUTPUT_4=p2-com-4-projective.mp4

BENCHMARK_GRAPH_FILE1=time_benchmark.png
BENCHMARK_GRAPH_FILE2=kps_benchmark.png

# SRCs
VIDEO_STABILIZATOR=video_stabilizator.py
SIFT_BENCHMARK=sift_benchmark.py
GET_BENCHMARK_IMAGES_SRC=$(SRC_LIST_FOLDER)/get_benchmark_list.sh

# ARGS
KP_TRESHOLD=100

#all: init_pipeline video_stabilizator
all: init_pipeline video_stabilizator

	@echo "Finish"

init_pipeline:
	$(call cecho,"[Start up configs]")
	$(call cecho,"- Creating input/output folders")
	@$(MKDIR_COMMAND) -p $(INPUT_FOLDER)
	@$(MKDIR_COMMAND) -p $(OUTPUT_FOLDER)
	@$(MKDIR_COMMAND) -p $(MY_OUTPUT_FOLDER)

	$(call cecho,"- Downloading video")
	@$(WGET_COMMAND) $(VIDEO_INPUT_URL_1) -O $(INPUT_FOLDER)/$(VIDEO_INPUT_1)
	@$(WGET_COMMAND) $(VIDEO_INPUT_URL_2) -O $(INPUT_FOLDER)/$(VIDEO_INPUT_2)
	@$(WGET_COMMAND) $(VIDEO_INPUT_URL_2B) -O $(INPUT_FOLDER)/$(VIDEO_INPUT_2B)
	@$(WGET_COMMAND) $(VIDEO_INPUT_URL_3) -O $(INPUT_FOLDER)/$(VIDEO_INPUT_3)
	@$(WGET_COMMAND) $(VIDEO_INPUT_URL_4) -O $(INPUT_FOLDER)/$(VIDEO_INPUT_4)

	@$(WGET_COMMAND) $(VIDEO_OUTPUT_URL_1) -O $(MY_OUTPUT_FOLDER)/$(VIDEO_OUTPUT_1)
	@$(WGET_COMMAND) $(VIDEO_OUTPUT_URL_2) -O $(MY_OUTPUT_FOLDER)/$(VIDEO_OUTPUT_2)
	@$(WGET_COMMAND) $(VIDEO_OUTPUT_URL_2SIFT) -O $(MY_OUTPUT_FOLDER)/$(VIDEO_OUTPUT_2_SIFT)
	@$(WGET_COMMAND) $(VIDEO_OUTPUT_URL_3) -O $(MY_OUTPUT_FOLDER)/$(VIDEO_OUTPUT_3)
	@$(WGET_COMMAND) $(VIDEO_OUTPUT_URL_4) -O $(MY_OUTPUT_FOLDER)/$(VIDEO_OUTPUT_4)

	@# Comment for final version
	@# @$(MKDIR_COMMAND) -p $(BENCHMARK_FOLDER)
	@#$(MKDIR_COMMAND) -p $(OUTPUT_CORRECTED_FOLDER)
	@#$(MKDIR_COMMAND) -p $(OUTPUT_ORIGINAL_FOLDER)
	@#$(MKDIR_COMMAND) -p $(OUTPUT_COMPOSED_FOLDER)

	@# $(call cecho,"- Downloading original image for benchmark")
	@# @$(WGET_COMMAND) $(BENCHMARK_IMAGE_URL) -O $(BENCHMARK_FOLDER)/$(BENCHMARK_IMAGE_FILE)

	@# $(call cecho,"- Downloading list for benchmark")
	@# @$(WGET_COMMAND) $(BENCHMARK_SIZE_LIST_URL) -O $(BENCHMARK_FOLDER)/$(BENCHMARK_SIZE_LIST)
	@# $(call cecho,"- Making images for benchmark")
	@# @$(BASH_COMMAND) $(SRC_FOLDER)/$(GET_BENCHMARK_IMAGES_SRC) $(BENCHMARK_FOLDER)/$(BENCHMARK_IMAGE_FILE)  $(BENCHMARK_FOLDER)/$(BENCHMARK_SIZE_LIST) $(BENCHMARK_PATTERN_NAME) $(BENCHMARK_PATTERN_EXTENSION) $(BENCHMARK_FOLDER) $(BENCHMARK_FOLDER)/$(BENCHMARK_IMAGE_LIST)
	@#  End comment


video_stabilizator:
	$(call cecho,"[Video stabilizator]")
	@$(PYTHON_COMMAND) $(SRC_FOLDER)/$(VIDEO_STABILIZATOR) -i $(INPUT_FOLDER)/$(VIDEO_INPUT_2B) -o $(OUTPUT_FOLDER)/$(VIDEO_OUTPUT) -oc $(OUTPUT_FOLDER)/$(VIDEO_COMPARER_OUTPUT) -kp_t $(KP_TRESHOLD)

sift_benchmark:
	$(call cecho,"[Sift benchmark]")
	$(PYTHON_COMMAND) $(SRC_FOLDER)/$(SIFT_BENCHMARK) -i $(BENCHMARK_FOLDER)/$(BENCHMARK_IMAGE_LIST) -g1 $(OUTPUT_FOLDER)/$(BENCHMARK_GRAPH_FILE1) -g2 $(OUTPUT_FOLDER)/$(BENCHMARK_GRAPH_FILE2)

clean:
	$(call cecho,"[Clean files]")
	@$(RM_COMMAND) -R $(OUTPUT_FOLDER)
	@$(MKDIR_COMMAND) -p $(OUTPUT_FOLDER)
	@$(MKDIR_COMMAND) -p $(OUTPUT_CORRECTED_FOLDER)
	@$(MKDIR_COMMAND) -p $(OUTPUT_ORIGINAL_FOLDER)
	@$(MKDIR_COMMAND) -p $(OUTPUT_COMPOSED_FOLDER)
