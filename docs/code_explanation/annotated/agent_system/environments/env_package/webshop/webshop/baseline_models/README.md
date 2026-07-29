<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
# 🤖 WebShop Baseline Models

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
This repository contains the source code for the baseline models discussed in the original paper, along with instructions for training the models and running them on WebShop.
<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
## 🚀 Set Up
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
* Install additional dependencies via `pip install -r requirements.txt`
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
* Download the training data for choice IL and place it into the `data` folder
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
```bash
cd data
unzip il_trajs_finalized_images.zip
cd ..
```
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
* Download the trained model checkpoints for search and choice IL from [here](https://drive.google.com/drive/folders/1liZmB1J38yY_zsokJAxRfN8xVO1B_YmD?usp=sharing).

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
When running the scripts discussed below, by default, the code will seek out the model parameters specified in the files/folders of the trained model checkpoints as:
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
* `./ckpts/web_click/epoch_9/model.pth` for `choice_il_epoch9.pth`
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
* `./ckpts/web_search/checkpoint-800` for `checkpoints-800/` (from `search_il_checkpoints_800.zip`)

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
We recommend creating these directories and putting the renamed files in the aforementioned, corresponding locations. If you are currently in this directory (`baseline_models`) and have the model checkpoints `.zip` file in your `Downloads` folder, these commands should do the trick.
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
```bash
mkdir -p ckpts/web_click/epoch_9/
mkdir -p ckpts/web_search/
mv ~/Downloads/choice_il_epoch9.pth ~/Downloads/model.pth
mv ~/Downloads/model.pth ckpts/web_click/epoch_9/
mv ~/Downloads/search_il_checkpoints_800.zip ckpts/web_search/
unzip ckpts/web_search_il_checkpoints_800.zip
```

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
Your final layout should look like this:
<p float="left">
    <img src="../assets/model_ckpts.png">
</p>


<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
On the other hand, if you'd like to put the files in a custom location, you can specify the custom file paths as arguments for the `test.py` as described below.

<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
## 🛠️ Usage
➤ Train the **search IL model** (BART Transformer):
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
> Note: Trained values will be output to `./ckpts/web_search` based on this [line](https://github.com/princeton-nlp/WebShop/blob/master/baseline_models/train_search_il.py#L119)
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
```bash
python train_search.py
```

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
➤ Train the **choice IL model** (BERT Transformer):
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
> Notes: Trained values will be output to `./ckpts/web_choice` based on this [line](https://github.com/princeton-nlp/WebShop/blob/master/baseline_models/train_choice_il.py#L299); List of Arguments [here](https://github.com/princeton-nlp/WebShop/blob/master/baseline_models/train_choice_il.py#L213) 
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
```bash
python train_choice.py
```

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
➤ Train the **choice RL** models
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
> Note: List of Arguments [here](https://github.com/princeton-nlp/WebShop/blob/master/baseline_models/train_rl.py#L171)
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
```bash
python train_rl.py
```

<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
## 🧪 Testing
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- Test the model on WebShop:
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
```bash
python test.py
```
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
- List of Arguments [here](https://github.com/princeton-nlp/WebShop/blob/master/baseline_models/test.py#L86)
    <!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
    - `--model_path` should point to the `choice_il_epoch9.pth` file
    <!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
    - `--bart_path` should point to the `checkpoints-800/` folder

<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
### 📙 Notes about Testing
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
1. You can specify the choice model path (`--model_path`) and the search model path (`--bart_path`) to load different models. 
    
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
2. While the rule baseline result is deterministic, model results could have variance due to the softmax sampling of the choice policy. `--softmax 0` will use a greedy policy and yield deterministic (but worse) results.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
3. `--bart 0` will use the user instruction as the only search query.

<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
## 🔀 Miscellaneous
Generate the search IL model's top-10 queries on all WebShop instructions:
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
```bash
# Will generate ./data/goal_query_predict.json
python generate_search.py
```

