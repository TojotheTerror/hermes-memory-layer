import sys, os, subprocess, json
import psutil

# Grab the live uv-managed backend server process env
backend = None
for p in psutil.process_iter(['pid','name','cmdline','exe']):
    try:
        cl = ' '.join(p.info['cmdline'] or [])
        if 'hermes_cli.main serve' in cl and 'uv' in (p.info.get('exe') or ''):
            backend = p; break
    except Exception:
        pass

if not backend:
    print("NO_BACKEND"); sys.exit(0)

env = backend.environ()
env['PYTHONIOENCODING'] = 'utf-8'
env['PYTHONUTF8'] = '1'
interp = backend.info['exe']

r = subprocess.run([interp,'-m','hermes_cli.main','memory','status'],
                   env=env, capture_output=True, text=True,
                   encoding='utf-8', errors='replace', timeout=90,
                   cwd=r'C:\Users\rpgmo\AppData\Local\hermes\hermes-agent')
out = (r.stdout or '') + '\n---STDERR---\n' + (r.stderr or '')
with open(r'C:\Users\rpgmo\hermes-memory-layer\scripts\live_status_output.txt','w',encoding='utf-8') as f:
    f.write(f"backend_pid={backend.pid}\ninterp={interp}\nexit={r.returncode}\n\n")
    f.write(out)
print("WRITTEN", backend.pid, r.returncode)
