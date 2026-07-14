"""
Fetch a file off a running Colab VM WITHOUT the `colab` CLI's download command.

Why: the Colab runtime-proxy token expires ~1h into a session, after which
`colab download`/`ls` fail with "File or directory not found" even for files
that exist (the VM and the run are fine). `list_assignments()` re-mints a fresh
~3600s token on every call, so this hits the Jupyter contents API directly and
works indefinitely. Use it to monitor long (>1h) detached runs — no `colab exec`
(zero teardown risk).

Run with the CLI tool's own python (has colab_cli + auth state):
  ~/.local/share/uv/tools/google-colab-cli/bin/python colab_fetch.py \
      content/drive/MyDrive/<...>/run.log > /tmp/run.log
Path is relative to the jupyter root ("/"), so strip the leading slash:
  /content/drive/... -> content/drive/...
"""
import sys, json, urllib.request, urllib.parse
from colab_cli.common import state

a = state.client.list_assignments()[0]
url = a.runtime_proxy_info.url.rstrip("/")
token = a.runtime_proxy_info.token
path = sys.argv[1].lstrip("/")
full = f"{url}/api/contents/{urllib.parse.quote(path)}?colab-runtime-proxy-token={token}&type=file&format=text"
req = urllib.request.Request(full)
try:
    with urllib.request.urlopen(req, timeout=90) as r:
        data = json.load(r)
    content = data.get("content")
    if isinstance(content, str):
        sys.stdout.write(content)
    else:
        print("NON-TEXT keys:", list(data.keys()))
except Exception as e:
    print("FETCH ERROR:", type(e).__name__, e)
