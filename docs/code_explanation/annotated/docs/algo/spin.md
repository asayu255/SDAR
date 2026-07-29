<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
# Recipe: Self-Play Fine-Tuning (SPIN)

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
`verl` provides a recipe inspired by the paper **"Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models"** (SPIN). SPIN is a language model finetuning algorithm that enables iterative self-improvement through a self-play mechanism inspired by game theory.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
**Core Idea:** Models learn by playing against themselves, reducing reliance on external preference datasets or stronger teacher models:

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
1.  **Synthetic Data Generation:** The current model generates responses, creating its own training data from previous iterations.
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
2.  **Two-Player Game Setup:** A game involving two players acted by a single LLM.
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
3.  **Iterative Training:** The model progressively improves by refining its policy, with each iteration's model becoming the opponent for the next iteration.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
Paper Authors: [Zixiang Chen](https://github.com/uclaml/SPIN)\*, [Yihe Deng](https://github.com/uclaml/SPIN)\*, [Huizhuo Yuan](https://scholar.google.com/citations?user=8foZzX4AAAAJ)\*, [Kaixuan Ji](https://scholar.google.com/citations?user=FOoKDukAAAAJ), [Quanquan Gu](https://web.cs.ucla.edu/~qgu/)

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
[[Webpage](https://uclaml.github.io/SPIN/)] [[Huggingface](https://huggingface.co/papers/2401.01335)] [[Paper](https://arxiv.org/abs/2401.01335)] [[Original Implementation](https://github.com/uclaml/SPIN)]

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
verl Implementation Authors: [Chendong Wang](https://cdwang96.github.io/), [Chenyang Zhao](https://github.com/zhaochenyang20)

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
---

<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
## Key Function (compute_online_dpo_loss) and Related works
SPIN (Chen et al., 2024) proposes an iterative self-play mechanism to fine-tune language models. In each iteration, SPIN's training objective, when using a logistic loss function, is equivalent to Direct Preference Optimization (DPO) loss (Rafailov et al., 2023). 

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
This `verl` recipe realizes SPIN's core concept by using DPO loss iteratively (Xu et al., 2023; Xiong et al., 2023; Snorkel AI, 2024). This means that in each iteration, we fine-tune the LLM using DPO loss for preference optimization. Notably, Xu et al. (2023) explored iterative preference optimization with pairwise cringe loss, while Xiong et al. (2023) discussed how to bridge theory and practice for RLHF under KL constraints using iterative training. The concept of iterative preference learning was also explored in online DPO (Guo et al., 2024), which focuses on direct alignment from online AI feedback. In online DPO, preference data is dynamically updated during training, allowing the model to learn from its own generated data.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
Specifically, we developed the **`compute_online_dpo_loss`** function and built this SPIN recipe on top of it. By incorporating online preference generation, this approach enables continuously refining language models without relying on fixed external preference datasets.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
**Reference Papers:**
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
* [Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models](https://arxiv.org/abs/2401.01335) (Chen et al., 2024) 
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
* [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) (Rafailov et al., 2023) 
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
* [Somethings are more cringe than others: Preference optimization with the pairwise cringe loss](https://arxiv.org/abs/2312.16682) (Xu et al., 2023) 
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
* [Iterative preference learning from human feedback: Bridging theory and practice for rlhf under kl-constraint](https://arxiv.org/abs/2312.11456) (Xiong et al., 2023)
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
* [Snorkel-Mistral-PairRM-DPO](https://huggingface.co/snorkelai/Snorkel-Mistral-PairRM-DPO) (Snorkel AI, 2024)
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
* [Direct language model alignment from online ai feedback](https://arxiv.org/abs/2402.04792) (Guo et al., 2024)


<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
## Our Online DPO Implementation

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
Our `compute_online_dpo_loss` function adapts `verl`'s existing PPO infrastructure (based on `verl` v0.3.0.post1) for this iterative online DPO. Key aspects of our implementation include:

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
* **No Critic:** Unlike PPO, we omit the value function critic.
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
* **Dynamic Reference Model:** An explicit reference policy (`ref_policy_wg`) is used for DPO loss. This reference model's weights can be periodically updated from the actor (`ref_update_freq`), providing a dynamic baseline.
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
* **Online Preference Generation:** The `compute_onlineDPO_pref` function (in `core_algos.py`) dynamically creates chosen/rejected pairs based on a reward source (e.g., rule-based ranking for math problems).
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
* **DPO Loss Integration:** We replace PPO's policy loss with our `compute_online_dpo_loss` (in `core_algos.py`) within the actor update (`dp_actor.py`), directly optimizing the policy using the generated preferences.
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
* **Iterative Training Orchestration:** The `SpinTrainer` (in `spin_trainer.py`) manages the entire self-play loop: generation, preference labeling, optional reference model updates, and policy updates, enabling continuous self-improvement aligned with SPIN's principles.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
---
<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
## Algorithm

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
This recipe implements an Online algorithm adapted to the `verl` Reinforcement Learning framework, which provides an alternative to PPO for fine-tuning language models.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
**Online Loop:** Instead of maximizing a scalar reward signal in PPO, this approach directly optimizes the policy model to align with preference data generated *online* during training:

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
1.  **Generation:** The current model generates multiple responses for each prompt in a batch.
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
2.  **Preference Labeling:** A function evaluates these generated responses to determine which one is preferred (chosen) and which is dispreferred (rejected). This can be done using a reward function or implicit ranking based on specific rules. (In this recipe, we use rule-based ranking on the math problem).
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
3.  **Update:** This preference tuple (`prompt`, `chosen_response`, `rejected_response`) is used to update the actor model using `compute_online_dpo_loss`, comparing against a reference model.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
**Connection with SPIN:**
Instead of only using a fixed target data distribution, the online generation loop in step 2 will dynamically change the target data distribution by using a certain Preference Labeling method (rule-based ranking on the math problem by selecting the better one in this recipe). This explores the direction mentioned in SPIN's paper Section 7 about "dynamically changing target data distribution" to potentially elevate LLM performance beyond the fixed human-annotated data ceiling.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
---

<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
## Reproduce the Experiment (Example Setup)

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
The following steps outline how to set up the environment and run the SPIN recipe, based on the provided test log using GSM8K and Qwen2.5-3B-Instruct.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
1.  **Setup Environment (Example using Docker):**
    <!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
    ```bash
    # Start a container with GPU access and shared memory
    docker run -it --name spin_test --gpus all \
        --shm-size=32g \
        --ipc=host \
        -v /path/to/host/.cache:/root/.cache \
        -e HF_TOKEN=<YOUR_HUGGINGFACE_TOKEN> \
        lmsysorg/sglang:latest \
        /bin/bash

    # Inside the container or on your host machine:
    # Ensure /tmp is writable
    mkdir -p /tmp
    chmod 1777 /tmp

    # Install Python 3.10 (if not present) and venv
    sudo apt update
    sudo apt install -y python3.10 python3.10-venv tmux
    python3 -m ensurepip --upgrade

    # Create and activate a virtual environment
    python3 -m venv ~/.python/spin_env
    source ~/.python/spin_env/bin/activate

    # Install uv (fast package installer)
    python3 -m pip install uv
    ```

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
2.  **Install verl and Dependencies:**
    <!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
    ```bash
    # Clone the verl repository and checkout the spin branch
    cd ~
    git clone git@github.com:volcengine/verl.git && cd verl

    # Install flash-attn (handle potential build issues)
    python3 -m uv pip install wheel packaging
    python3 -m uv pip install flash-attn --no-build-isolation --no-deps

    # Install verl with sglang extras
    python3 -m uv pip install -e ".[sglang]"
    ```
    <!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
    *Note: If `flash-attn` installation fails, try the manual steps again or consult its documentation.*

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
3.  **Login & Download Data/Model:**
    <!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
    ```bash
    # Login to Weights & Biases (optional, for logging)
    export WANDB_API_KEY=<YOUR_WANDB_API_KEY>
    # wandb login

    # Download the GSM8K dataset
    python3 examples/data_preprocess/gsm8k.py --local_dir ~/data/gsm8k # Adjusted path

    # Download the base model (Example: Qwen2.5-3B-Instruct)
    huggingface-cli download Qwen/Qwen2.5-3B-Instruct --local-dir $HOME/models/Qwen2.5-3B-Instruct
    ```

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
4.  **Configure:**
    <!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
    * Modify the configuration file (e.g., `config/spin_trainer.yaml` or the one specified in the run script) with correct paths to your downloaded model, data, desired hyperparameters (`dpo_beta`, learning rate, etc.), and distributed training settings (nodes, GPUs per node).
    <!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
    * Pay attention to `actor_rollout_ref.model_path`, `data` paths, `reward_model` config (if using one), and `trainer.ref_update_freq`.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
5.  **Run Training:**
    <!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
    ```bash
    # Set CUDA visible devices (adjust based on your hardware and config)
    export CUDA_VISIBLE_DEVICES=0,1,2,3

    # Launch the training script (e.g., test.sh or a custom script)
    # Ensure test.sh points to the correct config and main script
    bash recipe/spin/run_spin.sh
    ```

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
---

<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
## Configuration

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
* The primary configuration is typically managed through a YAML file specified in the launch script (e.g., `config/spin_trainer.yaml`).
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
* Key configuration sections:
    <!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
    * `data`: Paths to training/validation prompt files, batch sizes, sequence lengths.
    <!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
    * `actor_rollout_ref`: Paths to the base model (used for actor and initial reference), FSDP settings, optimization parameters (learning rate, scheduler).
    <!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
    * `reward_model`: Configuration for the reward model used for online preference labeling (path, batch size, etc.). Can be omitted if using a simpler reward function.
    <!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
    * `algorithm`: DPO-specific hyperparameters like `dpo_beta`, `dpo_loss_type`.
    <!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
    * `trainer`: Distributed training settings (nodes, GPUs per node), logging (WandB), checkpointing frequency, and `ref_update_freq` (set > 0 to enable periodic reference model updates from the actor).

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
---

<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
## Key Files

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
* `main_spin.py`: Main entry point using Hydra to load the config and launch the `SpinTrainer`.
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
* `spin_trainer.py`: Defines the `SpinTrainer` class, orchestrating the Online DPO training loop.
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
* `fsdp_workers.py`: Implements Ray workers (Actor, Reference) potentially using FSDP.
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
* `dp_actor.py`: Contains the actor class, including the DPO policy update logic.
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
* `core_algos.py`: Includes helper functions for `compute_online_dpo_loss` and `compute_onlineDPO_pref`.
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
* `config/spin_trainer.yaml` (or similar): Main Hydra configuration file for the recipe.
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
* `run_spin.sh` (or similar): Example bash script for launching a training run.
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
* `README.md`: This file.

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
---

<!-- [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。 -->
## Acknowledgement

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
We sincerely thank the contribution and guidance from the `verl` community and advisors, including (adapted from SPPO):

<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
* [Zixiang Chen](https://sites.google.com/view/zxchen)
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
* [Yuhao Yang](https://github.com/yhyang201)
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
* [Yifan Zhang](https://github.com/yifanzhang-pro)
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
* [Yongan Xiang](https://github.com/BearBiscuit05)
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
* [Junrong Lin](https://github.com/ocss884)
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
* [Yuxuan Tong](https://github.com/tongyx361)
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
* [Guangming Shen](https://github.com/PeterSH6)
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
* [Biao He](https://www.linkedin.com/in/biao-he/)
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
* [Qingquan Song](https://qingquansong.github.io/)
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
* [Chenyang Zhao](https://zhaochenyang20.github.io/Chayenne/)
<!-- [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。 -->
* [Quanquan Gu](https://web.cs.ucla.edu/~qgu/)
