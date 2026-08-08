head_dim=${1:-128}

cd ./tests/ncu_analyse/

./profile.sh $head_dim
