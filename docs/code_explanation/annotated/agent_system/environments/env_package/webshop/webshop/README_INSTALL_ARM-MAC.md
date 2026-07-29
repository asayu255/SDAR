<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
# Getting started on arm-based Mac

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
This README support the installation on the arm based mac. 


<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
## 🚀 Setup on arm based mac.
Our code is implemented in Python. To setup, do the following:
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
1. Install [Python 3.8.13](https://www.python.org/downloads/release/python-3813/)
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
2. Install [Java](https://www.java.com/en/download/)
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
3. Download the source code:
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
```sh
> git clone https://github.com/princeton-nlp/webshop.git webshop
```
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
4. Create a virtual environment using [Anaconda](https://anaconda.org/anaconda/python) and activate it
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
```sh
> conda create -n webshop python=3.8.13
> conda activate webshop
```
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
5. Install requirements into the `webshop` virtual environment via the `setup.sh` script
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
```sh
> ./setup_arm.sh [-d small|all]
```




<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
## Default Installation (Failures):
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
1. `pip3 install -r requirements.txt`

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
Fails at:
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- tokenizers
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- nmslib
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- lightgbm
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- transformers==4.19.2
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- PyYAML==6.0.0

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
Fails, because wrong versions installed
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- Werkzeug==2.2.2 (needs to be installed for Flask instead of 3.0.0 to work [https://stackoverflow.com/questions/77213053/why-did-flask-start-failing-with-importerror-cannot-import-name-url-quote-fr])
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- numpy-1.24.4 (needs to be installed instead of numpy 1.22 [https://stackoverflow.com/questions/33859531/runtimeerror-module-compiled-against-api-version-a-but-this-version-of-numpy-is])

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
**tokenizers fix**:
`pip3 install tokenizers`

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
**nmslib fix**:
`pip3 install Cython`
`pip3 install CFLAGS="-mavx -DWARN(a)=(a)" pip install nmslib`
[https://github.com/nmslib/nmslib/issues/476]

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
**lightgbm fix**:
[https://github.com/microsoft/LightGBM/issues/5328]
`brew install libomp`
`pip3 install lightgbm`


<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
**transformers fix**:
`pip3 install transformers-4.23.1` works

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
**PyYAML fix**:
`pip3 install PyYAML==6.0.1` works


<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
## Running setup.sh
Fails at:
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- `python -m spacy download en_core_web_lg
`

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
**Spacy Fix**:
Interestingly:

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- `pip install -U 'spacy[apple]'`

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
Does **NOT** work.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
So, first installing spacy with conda works.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
Remove:
`spacy==3.3.0` from `requirements.txt`




<!-- env_name:env_conda_webshop -->