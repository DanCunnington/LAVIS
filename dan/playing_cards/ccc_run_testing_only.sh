#!/bin/bash

# Handle command line arguments
while getopts e:c:d: flag
do
    case "${flag}" in
        e) NUM_EX=${OPTARG};;
        c) PCT_CORRECT=${OPTARG};;
        d) DECK=${OPTARG};;
    esac
done
if [ -z $NUM_EX ]; then NUM_EX=3328; fi
if [ -z $PCT_CORRECT ]; then PCT_CORRECT=1.0; fi
if [ -z $DECK ]; then DECK=""; fi
if [[ $PCT_CORRECT == "1" ]]; then
    PCT_CORRECT="1.0"
fi
FILENAME="${NUM_EX}_ex_${PCT_CORRECT}_correct"
MODEL_SAVE_PARENT_DIR=/dccstor/llama-7b/output/BLIP/${FILENAME}

# Copy Lavis code to new instance
echo "Copying code to new LAVIS directory: LAVIS_$FILENAME"
FULL_PATH=/u/$USER/lavis_runs/LAVIS_$FILENAME
if [ -d "$FULL_PATH" ] ; then
    echo "Removing existing directory"
    rm -fr "$FULL_PATH"
fi
cp -r /u/$USER/lavis_runs/LAVIS $FULL_PATH
cd $FULL_PATH


# Get the path of the results directory and set in the model config
dirs=($MODEL_SAVE_PARENT_DIR/*/)
CHECKPOINT_DIR="${dirs[0]}"
MODEL_CONFIG="blip_vqa_v2_playing_cards.yaml"
cp lavis/configs/models/blip_vqa_v2_playing_cards_template.yaml lavis/configs/models/$MODEL_CONFIG
sed -i -e "s#<<CHECKPOINT_DIR>>#${CHECKPOINT_DIR}#g" lavis/configs/models/$MODEL_CONFIG


# Run Test script and save result to output directory
echo "Running testing...."
source activate ilasp_python
export PYTHONPATH=$FULL_PATH
cd dan/playing_cards
IMAGE_DIR="${DECK}images"
python -u run_testing.py --image_dir $IMAGE_DIR > $CHECKPOINT_DIR/test_set_accuracy_score.txt

