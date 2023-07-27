category_list=(arts business computers games home health news recreation reference science shopping society sports)

CATEGORY=$1
ORIGIN_PATH="./dataset/"
TARGET_PATH="./dataset/truncated/"

if [[ ! " ${category_list[*]} " =~ " ${CATEGORY} " ]]; then
  echo "Invalid category! Category ${CATEGORY} is not in (${category_list[*]})."
  exit
fi

python3 ./dataset/truncate_data.py $CATEGORY $ORIGIN_PATH $TARGET_PATH
