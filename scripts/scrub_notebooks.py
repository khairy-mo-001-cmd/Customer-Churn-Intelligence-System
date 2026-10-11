"""Remove the hard-coded DB password and personal Windows paths from the notebooks.

Usage (from the repo root):  python scripts/scrub_notebooks.py notebooks/*.ipynb
It also adds a 'legacy' banner to notebooks that describe the old (superseded) pipeline
and replaces the Monitoring notebook's random fallback model with an explicit error.
"""
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT_RE = re.compile(
    r'C:\\Users\\LOQ\\OneDrive\\Desktop\\PROJECTS\\PROGRAMMING\\'
    r'Customer Churn Intelligence & Retention System')
TEMP_RE = re.compile(r'C:\\Users\\LOQ\\AppData\\Local\\Temp')
RAW_PATH_RE = re.compile(r'Path\(r"C:\\Users\\LOQ[^"]*?online_retail_II\.xlsx"\)')
RAW_PATH_NEW = ("(Path.cwd().resolve().parent if Path.cwd().name.lower() == 'notebooks' "
                "else Path.cwd().resolve()) / 'data' / 'raw' / 'online_retail_II.xlsx'")

LEGACY = {'feature eng and target.ipynb', 'Modeling & Evaluation.ipynb',
          'Deployment & Integration.ipynb', 'Monitoring & Maintenance.ipynb'}
BANNER = ('> **Legacy notebook.** It documents the first version of the pipeline '
          '(non-temporal target, 6 features). The maintained implementation is the CLI: '
          '`python -m src.pipeline clean | train | score` (see README.md). '
          'Do not use its saved artifacts.')


def scrub_text(text: str) -> str:
    text = PROJECT_ROOT_RE.sub('<PROJECT_ROOT>', text)
    text = TEMP_RE.sub('<TEMP>', text)
    return text.replace('2612', '****')


def scrub_code_lines(lines):
    out = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('DB_PASS') and '2612' in line:
            line = "DB_PASS = os.environ['DB_PASS']  # never hard-code credentials\n"
        elif "os.environ['POSTGRES_DATABASE_URL']" in line and '2612' in line:
            line = '# Set POSTGRES_DATABASE_URL in your shell or .env (never commit credentials)\n'
        line = RAW_PATH_RE.sub(RAW_PATH_NEW, line)
        out.append(line)
    return out


def replace_fallback_model(lines):
    """Swap the 'recreate a random fallback model' block for an explicit error."""
    out, skipping = [], False
    for line in lines:
        if line.strip().startswith('if not MODEL_PATH.is_file():'):
            out.append(line)
            out.append("    raise FileNotFoundError(f'{MODEL_PATH} is missing: run the training step first')\n")
            skipping = True
            continue
        if skipping:
            if 'Fallback model successfully recreated' in line:
                skipping = False
            continue
        out.append(line)
    return out


def process(path: Path) -> None:
    nb = json.loads(path.read_text(encoding='utf-8'))
    for cell in nb['cells']:
        src = cell['source']
        lines = src if isinstance(src, list) else src.splitlines(keepends=True)
        if cell['cell_type'] == 'code':
            lines = scrub_code_lines(lines)
            if path.name == 'Monitoring & Maintenance.ipynb':
                lines = replace_fallback_model(lines)
        cell['source'] = [scrub_text(l) for l in lines]
        for out in cell.get('outputs', []):
            if 'text' in out:
                out['text'] = [scrub_text(t) for t in out['text']]
            for key in ('data',):
                for mime, val in out.get(key, {}).items():
                    if mime == 'text/plain' and isinstance(val, list):
                        out[key][mime] = [scrub_text(t) for t in val]
            if 'traceback' in out:
                out['traceback'] = [scrub_text(t) for t in out['traceback']]
    if path.name in LEGACY and not (nb['cells'] and nb['cells'][0]['source'][:1] == [BANNER]):
        nb['cells'].insert(0, {'cell_type': 'markdown', 'metadata': {}, 'source': [BANNER]})
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    print('scrubbed', path)


if __name__ == '__main__':
    for arg in sys.argv[1:]:
        process(Path(arg))
