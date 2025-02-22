#!/bin/bash
set -e
git clone https://github.com/Aivis-Project/AivisSpeech-Engine ./scripts/tmp --filter=blob:none -n
cd ./scripts/tmp
git checkout 168b2a1144afe300b0490d9a6dd773ec6e927667 -- resources/dictionaries/*.csv
cd ../..
rm -rf ./crates/sbv2_core/src/dic
cp -r ./scripts/tmp/resources/dictionaries ./crates/sbv2_core/src/dic
rm -rf ./scripts/tmp
for file in ./crates/sbv2_core/src/dic/0*.csv; do
    /usr/bin/cat "$file"
    echo
done > ./crates/sbv2_core/src/all.csv
lindera build ./crates/sbv2_core/src/all.csv ./crates/sbv2_core/src/dic/all.dic -u -k ipadic