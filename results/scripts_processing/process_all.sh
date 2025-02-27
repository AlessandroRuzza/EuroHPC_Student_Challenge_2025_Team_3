for fold in ../2h_*/output; do
    echo $fold
    python script.py $fold
done