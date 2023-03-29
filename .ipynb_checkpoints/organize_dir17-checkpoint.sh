TGT_DIR=$1
CURRENT_DIR=$PWD

echo $PWD
cd $TGT_DIR
mkdir -p ensembeled
cd ensembeled
echo $PWD

cp $TGT_DIR/anon-proc-hits-seg-zh/analysis/ad-enzh-good-stnd.csv.gz .
gzip -d ad-enzh-good-stnd.csv.gz
cp $TGT_DIR/anon-proc-hits-seg-zh/analysis/ad-seg-scores.csv.gz .
gzip -d ad-seg-scores.csv.gz
mv ad-seg-scores.csv ad-enzh-seg-scores.csv

cp $TGT_DIR/anon-proc-hits-seg-ru/analysis/ad-enru-good-stnd.csv .
cp $TGT_DIR/anon-proc-hits-seg-ru/analysis/ad-seg-scores.csv .
mv ad-seg-scores.csv ad-enru-seg-scores.csv

cp $TGT_DIR/anon-proc-hits-seg-en/analysis/ad-csen-good-stnd.csv .
cp $TGT_DIR/anon-proc-hits-seg-en/analysis/ad-deen-good-stnd.csv .
cp $TGT_DIR/anon-proc-hits-seg-en/analysis/ad-fien-good-stnd.csv .
cp $TGT_DIR/anon-proc-hits-seg-en/analysis/ad-lven-good-stnd.csv .
cp $TGT_DIR/anon-proc-hits-seg-en/analysis/ad-ruen-good-stnd.csv .
cp $TGT_DIR/anon-proc-hits-seg-en/analysis/ad-tren-good-stnd.csv .
cp $TGT_DIR/anon-proc-hits-seg-en/analysis/ad-zhen-good-stnd.csv .
cp $TGT_DIR/anon-proc-hits-seg-en/analysis/ad-seg-scores.csv .

cd $CURRENT_DIR
echo $PWD




