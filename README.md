# Mini LLM

A small character-level language model built with PyTorch. This project includes a training script (`train.py`) and an interactive chat/testing script (`agent.py`).

## Requirements

- Windows
- Git
- Python 3.10 or newer
- Internet connection for `pip install`

## Clone the Repository

```bash
git clone https://github.com/McEmil1993/mini-llm.git
cd mini-llm
```

## Setup Using Git Bash on Windows

Use this if you are using Git Bash.

```bash
py -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

When the virtual environment is active, you should see `(.venv)` in your terminal prompt.

To deactivate it:

```bash
deactivate
```

## Setup Using PowerShell on Windows

Use this if you are using PowerShell.

```powershell
py -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

When the virtual environment is active, you should see `(.venv)` in your terminal prompt.

To deactivate it:

```powershell
deactivate
```

## Training

The training data is located at:

```text
data/train.txt
```

To train the model:

```bash
python train.py
```

After training, the checkpoint will be saved here:

```text
checkpoints/mini_llm.pt
```

Note: If `checkpoints/mini_llm.pt` already exists, you can use `agent.py` right away for testing/chat.

## Hardware Notes

The training script automatically uses CUDA if a compatible NVIDIA GPU is available. Otherwise, it uses CPU mode.

To check your CPU name, cores, and logical processors/threads in PowerShell:

```powershell
Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors
```

For an AMD Ryzen 3 PRO 2200G with 32GB RAM, the project should run in CPU mode. Training will work, but it can take longer than training on a GPU.

Recommended CPU settings for AMD Ryzen 3 PRO 2200G with 32GB RAM:

```python
batch_size = 16
block_size = 128
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 10

n_embd = 128
n_head = 4
n_layer = 4
dropout = 0.1
```

Recommended CPU settings for an Intel Core i3:

```python
batch_size = 8
block_size = 128
max_iters = 3000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 10

n_embd = 128
n_head = 4
n_layer = 4
dropout = 0.1
```

Recommended CPU settings for an Intel Core i3 with 8GB RAM:

```python
batch_size = 8
block_size = 96
max_iters = 3000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 10

n_embd = 96
n_head = 4
n_layer = 3
dropout = 0.1
```

If the Intel Core i3 with 8GB RAM still feels too slow or runs out of memory, reduce `batch_size` to `4`.

Recommended CPU settings for an Intel Core 3 N355 with 8 cores / 8 logical processors:

```python
batch_size = 12
block_size = 128
max_iters = 4000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 10

n_embd = 128
n_head = 4
n_layer = 4
dropout = 0.1
```

If the Intel Core 3 N355 machine has 16GB RAM or more and training speed is acceptable, you can use:

```python
batch_size = 16
max_iters = 5000
```

If training becomes too slow or the laptop gets too hot, reduce `batch_size` to `8`.

These values are found in `train.py` inside the `if device == "cpu":` section.

For a safer first upgrade, change only:

```python
max_iters = 5000
```

Training tips:

- If training is too slow or crashes, reduce `batch_size` to `8`.
- Keep `n_embd` divisible by `n_head`. For example, `128 / 4` is valid.
- Better and cleaner text in `data/train.txt` usually improves the model more than simply making the model larger.
- Avoid using the larger GPU settings on CPU unless you are okay with very slow training.

## Testing Using agent.py

Make sure the `.venv` is active, then run:

```bash
python agent.py
```

Once the agent starts, you can type a prompt:

```text
You: hello
Agent: ...
```

Available commands inside `agent.py`:

```text
/exit   - quit the agent
/clear  - clear chat memory
/save   - save the chat to chat_log.txt
/fast   - make the typing effect faster
/slow   - make the typing effect slower
/temp X - change the temperature, example: /temp 0.8
```

## Optional: Simple Generation Script

There is also `generate.py` if you only want to enter one prompt and generate text:

```bash
python generate.py
```

## Notes

- Do not commit `.venv`; it is already included in `.gitignore`.
- If `checkpoints/mini_llm.pt` does not exist, train the model first using `python train.py`.
- Training on CPU is slower than training on GPU.
