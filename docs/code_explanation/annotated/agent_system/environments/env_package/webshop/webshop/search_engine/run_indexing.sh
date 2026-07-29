# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input resources_100 \
  --index indexes_100 \
  --generator DefaultLuceneDocumentGenerator \
  --threads 1 \
  --storePositions --storeDocvectors --storeRaw

# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input resources \
  --index indexes \
  --generator DefaultLuceneDocumentGenerator \
  --threads 1 \
  --storePositions --storeDocvectors --storeRaw

# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input resources_1k \
  --index indexes_1k \
  --generator DefaultLuceneDocumentGenerator \
  --threads 1 \
  --storePositions --storeDocvectors --storeRaw

# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input resources_100k \
  --index indexes_100k \
  --generator DefaultLuceneDocumentGenerator \
  --threads 1 \
  --storePositions --storeDocvectors --storeRaw
