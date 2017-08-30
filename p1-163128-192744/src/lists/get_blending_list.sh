#bin/bash

IMAGE_LIST=$1
INPUT_DIR=$2

while read -r file; do
    FILE_NAME=$(echo $file | cut -d "/" -f 5)

    wget $file -O $INPUT_DIR/$FILE_NAME

done < $IMAGE_LIST

