declare -a arr=("cam9exp5a" "cam11exp5a" "cam13exp5a")
for i in "${arr[@]}"
do
python pipeline.py --exp "$i" --vis True
done

