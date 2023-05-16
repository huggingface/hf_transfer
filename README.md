# HF Transfer

Speed up file transfers with the Hub.

# DISCLAIMER

This library is a power user tool, to go beyond `~500MB/s` on very high bandwidth
network, where Python cannot cap out the available bandwidth.

This is *not* meant to be a general usability tool.
It purposefully lacks progressbars and comes generally as-is.

Please file issues *only* if there's an issue on the underlying downloaded file.

## Contributing

```sh
python3 -m venv ~/.venv/hf_transfer
source ~/.venv/hf_transfer/bin/activate
pip install maturin
maturin develop
```

### `huggingface_hub`

If you are working on changes with `huggingface_hub`

```sh
git clone git@github.com:huggingface/huggingface_hub.git
# git clone https://github.com/huggingface/huggingface_hub.git

cd huggingface_hub
python3 -m pip install -e ".[quality]"
```

You can use the following test script:

```py
import os

# os.environ["HF_ENDPOINT"] = "http://localhost:5564"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from huggingface_hub import HfApi, logging

logging.set_verbosity_debug()
hf = HfApi()
hf.upload_file(path_or_fileobj="/path/to/my/repo/some_file", path_in_repo="some_file", repo_id="my/repo", repo_type="model")
```

