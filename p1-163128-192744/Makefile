#FUNCTION
define cecho
	@echo "\033[92m$(1)\033[0m"
endef


# COMMANDS
WGET_COMMAND=wget
PYTHON_COMMAND=python3
BASH_COMMAND=sh
MOGRIFY_COMAND=convert
MKDIR_COMMAND=mkdir
REMOVE_COMMAND=rm

# FOLDERS
INPUT_FOLDER=./input
BENCHMARK_FOLDER=$(INPUT_FOLDER)/benchmark
OUTPUT_FOLDER=./output
SRC_FOLDER=./src


BENCHMARK_SIZE_LIST=benchmark_size_list.txt
BENCHMARK_IMAGE_LIST=becnhmark_image_list.txt

BLENDING_INPUT_LIST=blending_list.txt

#URLS
BENCHMARK_IMAGE_URL=https://i.pinimg.com/736x/1c/a1/f4/1ca1f4aacf8ad80bf1fffc1f48095378--beautiful-nature-wallpaper-beautiful-landscapes.jpg
BENCHMARK_SIZE_LIST_URL=https://gist.githubusercontent.com/so77id/1f912253f3075b812017bd0a3583efc1/raw/605585442a53950daee3286bc704a92c1a7b405c/MO446p1-size-list.txt
BLENDING_INPUT_LIST_URL=https://gist.githubusercontent.com/so77id/af1ef6763e6b9b8dff01af3a4c16633b/raw/b4b165e07efe4d20fad6116414659b2f9263d56d/MO446p1-inputs.txt

#PYRAMID PARAMETERS
GAUSSIAN_PYRAMID_SIZE=6


# INPUT FILES
# Benchmark files
BENCHMARK_IMAGE_FILE=$(P1_1_A)
BENCHMARK_PATTERN_NAME=p1_bench
BENCHMARK_PATTERN_EXTENSION=png
# First experiment
P1_1_A=p1_1_b.png
P1_1_B=p1_1_a.png
P1_1_M=p1_1_m.png

# Second experiment
P1_2_A=p1_2_a.png
P1_2_B=p1_2_b.png
P1_2_M=p1_2_m.png

# Third experiment
P1_3_A=p1_3_a.png
P1_3_B=p1_3_b.png
P1_3_M=p1_3_m.png

# OUTPUT FILES
# BENCHMARK FILES
BENCHMARK_GRAPH_FILE1=p1_2_0.png
BENCHMARK_GRAPH_FILE2=p1_2_1.png

# Pyramid compose
P1_2_2_G=p1_2_2_g.png
P1_2_2_L=p1_2_2_l.png

# spatial blending experiments
P1_2_3_1_O=p1_2_3_1_o.png
P1_2_3_2_O=p1_2_3_2_o.png
P1_2_3_3_O=p1_2_3_3_o.png

# Frequency experiments
P1_3_1_mg_min=p1_3_1_mg_min.png
P1_3_1_mg_min_25=p1_3_1_mg_min_25.png
P1_3_1_mg_min_50=p1_3_1_mg_min_50.png
P1_3_1_mg_min_75=p1_3_1_mg_min_75.png
P1_3_1_mg_min_09999=p1_3_1_mg_min_09999.png
P1_3_1_mg_min_099999=p1_3_1_mg_min_099999.png
P1_3_1_mg_min_100=p1_3_1_mg_min_100.png
P1_3_1_mg_max=p1_3_1_mg_max.png
P1_3_1_mg_max_01=p1_3_1_mg_max_01.png
P1_3_1_mg_max_1=p1_3_1_mg_max_1.png
P1_3_1_mg_max_25=p1_3_1_mg_max_25.png
P1_3_1_mg_max_50=p1_3_1_mg_max_50.png
P1_3_1_mg_max_75=p1_3_1_mg_max_75.png
P1_3_1_mg_max_100=p1_3_1_mg_max_100.png
P1_3_1_ph_min=p1_3_1_ph_min.png
P1_3_1_ph_min_25=p1_3_1_ph_min_25.png
P1_3_1_ph_min_50=p1_3_1_ph_min_50.png
P1_3_1_ph_min_75=p1_3_1_ph_min_75.png
P1_3_1_ph_min_100=p1_3_1_ph_min_100.png
P1_3_1_ph_max=p1_3_1_ph_max.png
P1_3_1_ph_max_25=p1_3_1_ph_max_25.png
P1_3_1_ph_max_50=p1_3_1_ph_max_50.png
P1_3_1_ph_max_75=p1_3_1_ph_max_75.png
P1_3_1_ph_max_100=p1_3_1_ph_max_100.png

# Frequency blending
P1_3_2_1_0=p1_3_2_1_0.png
P1_3_2_1_1=p1_3_2_1_1.png
P1_3_2_1_2=p1_3_2_1_2.png
P1_3_2_2_0=p1_3_2_2_0.png
P1_3_2_2_1=p1_3_2_2_1.png
P1_3_2_2_2=p1_3_2_2_2.png

# SRC FILES
SRC_LIST_FOLDER=lists
GET_BENCHMARK_IMAGES_SRC=$(SRC_LIST_FOLDER)/get_benchmark_list.sh
GET_BLENDING_IMAGES_SRC=$(SRC_LIST_FOLDER)/get_blending_list.sh

CONVOLUTION_BENCHMARK_FILE=convolution_benchmark.py
PYRAMID_TEST_FILE=pyramid_test.py
SPATIAL_BLENDING=spatial_blending.py
FREQUENCY_BLENDING=frequency_blending.py
FOURIER_SPACE_FILE=fourier_space.py




all: init_pipeline convolution_benchmark pyramid_test spatial_blending fourier_space_exploration frequency_blending

	@echo "Finish"

init_pipeline:
	$(call cecho,"[Start up configs]")
	$(call cecho,"- Creating input/output folders")
	@$(MKDIR_COMMAND) -p $(INPUT_FOLDER)
	@$(MKDIR_COMMAND) -p $(BENCHMARK_FOLDER)
	@$(MKDIR_COMMAND) -p $(OUTPUT_FOLDER)


	$(call cecho,"- Downloading list for blending")
	@$(WGET_COMMAND) $(BLENDING_INPUT_LIST_URL) -O $(INPUT_FOLDER)/$(BLENDING_INPUT_LIST)
	$(call cecho,"- Downloading input files")
	@$(BASH_COMMAND) $(SRC_FOLDER)/$(GET_BLENDING_IMAGES_SRC) $(INPUT_FOLDER)/$(BLENDING_INPUT_LIST) $(INPUT_FOLDER)


	#$(call cecho,"- Downloading original image for benchmark")
	#@$(WGET_COMMAND) $(BENCHMARK_IMAGE_URL) -O $(BENCHMARK_FOLDER)/$(BENCHMARK_IMAGE_FILE)
	$(call cecho,"- Downloading list for benchmark")
	@$(WGET_COMMAND) $(BENCHMARK_SIZE_LIST_URL) -O $(BENCHMARK_FOLDER)/$(BENCHMARK_SIZE_LIST)
	$(call cecho,"- Making images for benchmark")
	@$(BASH_COMMAND) $(SRC_FOLDER)/$(GET_BENCHMARK_IMAGES_SRC) $(INPUT_FOLDER)/$(BENCHMARK_IMAGE_FILE)  $(BENCHMARK_FOLDER)/$(BENCHMARK_SIZE_LIST) $(BENCHMARK_PATTERN_NAME) $(BENCHMARK_PATTERN_EXTENSION) $(BENCHMARK_FOLDER) $(BENCHMARK_FOLDER)/$(BENCHMARK_IMAGE_LIST)

	$(call cecho,"- Deleting cache files")
	@$(REMOVE_COMMAND) $(BENCHMARK_FOLDER)/$(BENCHMARK_SIZE_LIST) $(INPUT_FOLDER)/$(BLENDING_INPUT_LIST)

convolution_benchmark:
	$(call cecho,"[Making benchmark of convolution function]")
	$(PYTHON_COMMAND) $(SRC_FOLDER)/$(CONVOLUTION_BENCHMARK_FILE) -i $(BENCHMARK_FOLDER)/$(BENCHMARK_IMAGE_LIST) -nk 10 -g1 $(OUTPUT_FOLDER)/$(BENCHMARK_GRAPH_FILE1) -g2 $(OUTPUT_FOLDER)/$(BENCHMARK_GRAPH_FILE2)

pyramid_test:
	$(call cecho,"[Making a test of pyramids implementation]")
	@$(PYTHON_COMMAND) $(SRC_FOLDER)/$(PYRAMID_TEST_FILE) -i $(INPUT_FOLDER)/$(BENCHMARK_IMAGE_FILE) -ps $(GAUSSIAN_PYRAMID_SIZE) -opg $(OUTPUT_FOLDER)/$(P1_2_2_G) -opl $(OUTPUT_FOLDER)/$(P1_2_2_L)

spatial_blending:
	$(call cecho,"[Making a spatial blending]")
	$(call cecho,"- First experiment")
	@$(PYTHON_COMMAND) $(SRC_FOLDER)/$(SPATIAL_BLENDING) -i1 $(INPUT_FOLDER)/$(P1_1_A) -i2 $(INPUT_FOLDER)/$(P1_1_B) -m $(INPUT_FOLDER)/$(P1_1_M) -ps $(GAUSSIAN_PYRAMID_SIZE) -o $(OUTPUT_FOLDER)/$(P1_2_3_1_O)
	$(call cecho,"- Second experiment")
	@$(PYTHON_COMMAND) $(SRC_FOLDER)/$(SPATIAL_BLENDING) -i1 $(INPUT_FOLDER)/$(P1_2_A) -i2 $(INPUT_FOLDER)/$(P1_2_B) -m $(INPUT_FOLDER)/$(P1_2_M) -ps $(GAUSSIAN_PYRAMID_SIZE) -o $(OUTPUT_FOLDER)/$(P1_2_3_2_O)
	$(call cecho,"- Third experiment")
	@$(PYTHON_COMMAND) $(SRC_FOLDER)/$(SPATIAL_BLENDING) -i1 $(INPUT_FOLDER)/$(P1_3_A) -i2 $(INPUT_FOLDER)/$(P1_3_B) -m $(INPUT_FOLDER)/$(P1_3_M) -ps $(GAUSSIAN_PYRAMID_SIZE) -o $(OUTPUT_FOLDER)/$(P1_2_3_3_O)

fourier_space_exploration:
	$(call cecho,"[Fourier space exploration]")
	@$(PYTHON_COMMAND) $(SRC_FOLDER)/$(FOURIER_SPACE_FILE) -i $(INPUT_FOLDER)/$(P1_3_A) -o $(OUTPUT_FOLDER)/$(P1_3_1_mg_min)
	@$(PYTHON_COMMAND) $(SRC_FOLDER)/$(FOURIER_SPACE_FILE) -i $(INPUT_FOLDER)/$(P1_3_A) -o $(OUTPUT_FOLDER)/$(P1_3_1_mg_min_09999) -t 0.9999
	@$(PYTHON_COMMAND) $(SRC_FOLDER)/$(FOURIER_SPACE_FILE) -i $(INPUT_FOLDER)/$(P1_3_A) -o $(OUTPUT_FOLDER)/$(P1_3_1_mg_min_099999) -t 0.99999
	@$(PYTHON_COMMAND) $(SRC_FOLDER)/$(FOURIER_SPACE_FILE) -i $(INPUT_FOLDER)/$(P1_3_A) -o $(OUTPUT_FOLDER)/$(P1_3_1_mg_min_25) -t 0.25
	@$(PYTHON_COMMAND) $(SRC_FOLDER)/$(FOURIER_SPACE_FILE) -i $(INPUT_FOLDER)/$(P1_3_A) -o $(OUTPUT_FOLDER)/$(P1_3_1_mg_min_50) -t 0.5
	@$(PYTHON_COMMAND) $(SRC_FOLDER)/$(FOURIER_SPACE_FILE) -i $(INPUT_FOLDER)/$(P1_3_A) -o $(OUTPUT_FOLDER)/$(P1_3_1_mg_min_75) -t 0.75
	@$(PYTHON_COMMAND) $(SRC_FOLDER)/$(FOURIER_SPACE_FILE) -i $(INPUT_FOLDER)/$(P1_3_A) -o $(OUTPUT_FOLDER)/$(P1_3_1_mg_min_100) -t 1.0
	@$(PYTHON_COMMAND) $(SRC_FOLDER)/$(FOURIER_SPACE_FILE) -i $(INPUT_FOLDER)/$(P1_3_A) -o $(OUTPUT_FOLDER)/$(P1_3_1_mg_max) -r
	@$(PYTHON_COMMAND) $(SRC_FOLDER)/$(FOURIER_SPACE_FILE) -i $(INPUT_FOLDER)/$(P1_3_A) -o $(OUTPUT_FOLDER)/$(P1_3_1_mg_max_01) -t 0.001 -r
	@$(PYTHON_COMMAND) $(SRC_FOLDER)/$(FOURIER_SPACE_FILE) -i $(INPUT_FOLDER)/$(P1_3_A) -o $(OUTPUT_FOLDER)/$(P1_3_1_mg_max_1) -t 0.01 -r
	@$(PYTHON_COMMAND) $(SRC_FOLDER)/$(FOURIER_SPACE_FILE) -i $(INPUT_FOLDER)/$(P1_3_A) -o $(OUTPUT_FOLDER)/$(P1_3_1_mg_max_25) -t 0.25 -r
	@$(PYTHON_COMMAND) $(SRC_FOLDER)/$(FOURIER_SPACE_FILE) -i $(INPUT_FOLDER)/$(P1_3_A) -o $(OUTPUT_FOLDER)/$(P1_3_1_mg_max_50) -t 0.5 -r
	@$(PYTHON_COMMAND) $(SRC_FOLDER)/$(FOURIER_SPACE_FILE) -i $(INPUT_FOLDER)/$(P1_3_A) -o $(OUTPUT_FOLDER)/$(P1_3_1_mg_max_75) -t 0.75 -r
	@$(PYTHON_COMMAND) $(SRC_FOLDER)/$(FOURIER_SPACE_FILE) -i $(INPUT_FOLDER)/$(P1_3_A) -o $(OUTPUT_FOLDER)/$(P1_3_1_mg_max_100) -t 1.0
	@$(PYTHON_COMMAND) $(SRC_FOLDER)/$(FOURIER_SPACE_FILE) -i $(INPUT_FOLDER)/$(P1_3_A) -o $(OUTPUT_FOLDER)/$(P1_3_1_ph_min) -p
	@$(PYTHON_COMMAND) $(SRC_FOLDER)/$(FOURIER_SPACE_FILE) -i $(INPUT_FOLDER)/$(P1_3_A) -o $(OUTPUT_FOLDER)/$(P1_3_1_ph_min_25) -t 0.25 -p
	@$(PYTHON_COMMAND) $(SRC_FOLDER)/$(FOURIER_SPACE_FILE) -i $(INPUT_FOLDER)/$(P1_3_A) -o $(OUTPUT_FOLDER)/$(P1_3_1_ph_min_50) -t 0.5 -p
	@$(PYTHON_COMMAND) $(SRC_FOLDER)/$(FOURIER_SPACE_FILE) -i $(INPUT_FOLDER)/$(P1_3_A) -o $(OUTPUT_FOLDER)/$(P1_3_1_ph_min_75) -t 0.75 -p
	@$(PYTHON_COMMAND) $(SRC_FOLDER)/$(FOURIER_SPACE_FILE) -i $(INPUT_FOLDER)/$(P1_3_A) -o $(OUTPUT_FOLDER)/$(P1_3_1_ph_min_100) -t 1.0 -p
	@$(PYTHON_COMMAND) $(SRC_FOLDER)/$(FOURIER_SPACE_FILE) -i $(INPUT_FOLDER)/$(P1_3_A) -o $(OUTPUT_FOLDER)/$(P1_3_1_ph_max) -p  -r
	@$(PYTHON_COMMAND) $(SRC_FOLDER)/$(FOURIER_SPACE_FILE) -i $(INPUT_FOLDER)/$(P1_3_A) -o $(OUTPUT_FOLDER)/$(P1_3_1_ph_max_25) -t 0.25 -p -r
	@$(PYTHON_COMMAND) $(SRC_FOLDER)/$(FOURIER_SPACE_FILE) -i $(INPUT_FOLDER)/$(P1_3_A) -o $(OUTPUT_FOLDER)/$(P1_3_1_ph_max_50) -t 0.5 -p -r
	@$(PYTHON_COMMAND) $(SRC_FOLDER)/$(FOURIER_SPACE_FILE) -i $(INPUT_FOLDER)/$(P1_3_A) -o $(OUTPUT_FOLDER)/$(P1_3_1_ph_max_75) -t 0.75 -p -r
	@$(PYTHON_COMMAND) $(SRC_FOLDER)/$(FOURIER_SPACE_FILE) -i $(INPUT_FOLDER)/$(P1_3_A) -o $(OUTPUT_FOLDER)/$(P1_3_1_ph_max_100) -t 1.0 -p -r

frequency_blending:
	$(call cecho,"[Making a frequency blending]")
	@$(PYTHON_COMMAND) $(SRC_FOLDER)/$(FREQUENCY_BLENDING) -i1 $(INPUT_FOLDER)/$(P1_1_B) -i2 $(INPUT_FOLDER)/$(P1_1_A) -m $(INPUT_FOLDER)/$(P1_1_M) -ps $(GAUSSIAN_PYRAMID_SIZE) -o $(OUTPUT_FOLDER)/$(P1_3_2_1_0) -a 1
	@$(PYTHON_COMMAND) $(SRC_FOLDER)/$(FREQUENCY_BLENDING) -i1 $(INPUT_FOLDER)/$(P1_2_B) -i2 $(INPUT_FOLDER)/$(P1_2_A) -m $(INPUT_FOLDER)/$(P1_2_M) -ps $(GAUSSIAN_PYRAMID_SIZE) -o $(OUTPUT_FOLDER)/$(P1_3_2_1_1) -a 1
	@$(PYTHON_COMMAND) $(SRC_FOLDER)/$(FREQUENCY_BLENDING) -i1 $(INPUT_FOLDER)/$(P1_3_B) -i2 $(INPUT_FOLDER)/$(P1_3_A) -m $(INPUT_FOLDER)/$(P1_3_M) -ps $(GAUSSIAN_PYRAMID_SIZE) -o $(OUTPUT_FOLDER)/$(P1_3_2_1_2) -a 1
	@$(PYTHON_COMMAND) $(SRC_FOLDER)/$(FREQUENCY_BLENDING) -i1 $(INPUT_FOLDER)/$(P1_1_B) -i2 $(INPUT_FOLDER)/$(P1_1_A) -m $(INPUT_FOLDER)/$(P1_1_M) -o $(OUTPUT_FOLDER)/$(P1_3_2_2_0) -a 2
	@$(PYTHON_COMMAND) $(SRC_FOLDER)/$(FREQUENCY_BLENDING) -i1 $(INPUT_FOLDER)/$(P1_2_B) -i2 $(INPUT_FOLDER)/$(P1_2_A) -m $(INPUT_FOLDER)/$(P1_2_M) -o $(OUTPUT_FOLDER)/$(P1_3_2_2_1) -a 2
	@$(PYTHON_COMMAND) $(SRC_FOLDER)/$(FREQUENCY_BLENDING) -i1 $(INPUT_FOLDER)/$(P1_3_B) -i2 $(INPUT_FOLDER)/$(P1_3_A) -m $(INPUT_FOLDER)/$(P1_3_M) -o $(OUTPUT_FOLDER)/$(P1_3_2_2_2) -a 2

