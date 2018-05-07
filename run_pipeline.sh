declare -a arr=("cam11exp5b" "cam13exp5b" "cam2exp5b" "cam5exp5b" "cam9exp5b")
# declare -a arr=("cam2exp5a" "cam5exp5a")
for i in "${arr[@]}"
do
python pipeline.py --exp "$i" --vis False
done

#python mycore/reidentification.py --source_view cam9exp5a --exps cam11exp5a,cam13exp5a,cam9exp5a
#python mycore/association.py --exps cam11exp5a,cam13exp5a,cam9exp5a --vis False

