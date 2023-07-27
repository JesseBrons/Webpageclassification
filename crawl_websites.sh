category_list=(arts business computers games home health news recreation reference science shopping society sports)

CATEGORY=$1
DATA_FOLDER="./dataset/"

if [[ ! " ${category_list[*]} " =~ " ${CATEGORY} " ]]; then
  echo "Invalid category! Category ${CATEGORY} is not in (${category_list[*]})."
  exit
fi

python3 ./dataset/crawl_websites.sh $CATEGORY $DATA_FOLDER
