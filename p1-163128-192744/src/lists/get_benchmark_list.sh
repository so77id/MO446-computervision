#bin/bash

ORIGINAL_IMAGE=$1
SIZE_LIST=$2
NAME_PATTERN=$3
EXTENSION=$4
OUTPUT_FOLDER=$5
OUTPUT_LIST=$6

> $OUTPUT_LIST

while read -r SIZE; do
    echo "Creating image with size: $SIZE"
    FILE_NAME=$OUTPUT_FOLDER/$NAME_PATTERN\_$SIZE.$EXTENSION
    echo "convert $ORIGINAL_IMAGE -resize $SIZE! $FILE_NAME"
    echo "convert $ORIGINAL_IMAGE -resize $SIZE! $FILE_NAME"
    convert $ORIGINAL_IMAGE -resize $SIZE! $FILE_NAME

    echo "$FILE_NAME $SIZE" >> $OUTPUT_LIST
done < $SIZE_LIST

