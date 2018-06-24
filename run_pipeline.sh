#!/bin/sh
# single camera: run on all cameras
declare -a arr=("cam11exp5b" "cam13exp5b" "cam2exp5b" "cam5exp5b" "cam9exp5b")
for i in "${arr[@]}"
do
python pipeline.py --exp "$i" --vis False
done

# multi camera: run on cam 5,9,13 for demo
python mycore/reidentification.py --source_view cam9exp5a --exps cam11exp5a,cam13exp5a,cam9exp5a
python mycore/association.py --exps cam11exp5a,cam13exp5a,cam9exp5a --vis False

