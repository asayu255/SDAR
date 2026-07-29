# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
# [EXPLAIN] この行は現在の処理ブロックで使用する宣言、状態更新または制御を表す。
SPHINXOPTS    =
# [EXPLAIN] この行は現在の処理ブロックで使用する宣言、状態更新または制御を表す。
SPHINXBUILD   = sphinx-build
# [EXPLAIN] この行は現在の処理ブロックで使用する宣言、状態更新または制御を表す。
SPHINXPROJ    = verl
# [EXPLAIN] この行は現在の処理ブロックで使用する宣言、状態更新または制御を表す。
SOURCEDIR     = .
# [EXPLAIN] この行は現在の処理ブロックで使用する宣言、状態更新または制御を表す。
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
# [EXPLAIN] この行は現在の処理ブロックで使用する宣言、状態更新または制御を表す。
help:
	# [EXPLAIN] この行は現在の処理ブロックで使用する宣言、状態更新または制御を表す。
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# [EXPLAIN] この行は現在の処理ブロックで使用する宣言、状態更新または制御を表す。
.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
# [EXPLAIN] この行は現在の処理ブロックで使用する宣言、状態更新または制御を表す。
%: Makefile
	# [EXPLAIN] この行は現在の処理ブロックで使用する宣言、状態更新または制御を表す。
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
