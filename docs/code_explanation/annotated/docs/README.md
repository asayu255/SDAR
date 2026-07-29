<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
# verl documents

<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
## Build the docs

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
```bash
# Install dependencies.
pip install -r requirements-docs.txt

# Build the docs.
make clean
make html
```

<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
## Open the docs with your browser

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
```bash
python -m http.server -d _build/html/
```
Launch your browser and navigate to http://localhost:8000 to view the documentation.