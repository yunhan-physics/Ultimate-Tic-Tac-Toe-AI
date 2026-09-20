# Optional local tools

This directory is reserved for downloaded command-line tools. Executables and
tool state are intentionally excluded from Git.

For Cloudflare quick tunnels, download the appropriate `cloudflared` binary
from the [official releases page](https://github.com/cloudflare/cloudflared/releases/latest),
place it here as `cloudflared.exe` on Windows, and run:

```powershell
python launch_public.py --provider cloudflare
```

The default localtunnel provider does not require `cloudflared`.
