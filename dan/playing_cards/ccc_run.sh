#!/bin/bash

# Handle command line arguments
while getopts e:c: flag
do
    case "${flag}" in
        e) NUM_EX=${OPTARG};;
        c) PCT_CORRECT=${OPTARG};;
    esac
done
if [ -z $NUM_EX ]; then NUM_EX=3328; fi
if [ -z $PCT_CORRECT ]; then PCT_CORRECT=1.0; fi
if [[ $PCT_CORRECT == "1" ]]; then
	PCT_CORRECT="1.0"
fi
FILENAME="${NUM_EX}_ex_${PCT_CORRECT}_correct"

# Copy Lavis code to new instance
cp -r /u/$USER/lavis_runs/LAVIS /u/$USER/lavis_runs/LAVIS_$FILENAME
cd /u/$USER/lavis_runs/LAVIS_$FILENAME

# Create new config file from template
FILENAME="${NUM_EX}_ex_${PCT_CORRECT}_correct"
DATASET_CONFIG="playing_cards_vqa_${FILENAME}.yaml"
cp lavis/configs/datasets/playing_cards/playing_cards_vqa_template.yaml lavis/configs/datasets/playing_cards/${DATASET_CONFIG}
sed -i -e 's/<<FILENAME>>/train_${FILENAME}.json/g' lavis/configs/datasets/playing_cards/${DATASET_CONFIG}


# Create new projects/blip/train config with custom output dir
BLIP_CONFIG="vqav2_playing_cards_${FILENAME}.yaml"
cp lavis/projects/blip/train/vqav2_playing_cards_template.yaml lavis/projects/blip/train/$BLIP_CONFIG
sed -i -e 's/<<FILENAME>>/${FILENAME}/g' lavis/projects/blip/train/$BLIP_CONFIG 


source activate ilasp_python
# python train.py --cfg-path lavis/projects/blip/train/$BLIP_CONFIG 

# TODO: Develop test scoring script