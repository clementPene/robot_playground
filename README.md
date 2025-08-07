prerequies:

install [uv](https://docs.astral.sh/uv/getting-started/installation/)

```bash
sudo apt update && sudo apt install libeigen3-dev libboost-all-dev liboctomap-dev
sudo apt install python-is-python3


uv venv
uv pip install -r requirements.txt
```

launch Jupyter notebook :
```bash
source .venv/bin/activate
jupyter lab
```

or

```bash
uv run jupyter lab
```

