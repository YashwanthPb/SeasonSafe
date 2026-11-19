"""Build report/SeasonSafe_IEEE_Report.docx from report/seasonsafe.tex with pandoc.

Run from the repository root:  python report/make_docx.py
Needs pandoc on PATH. Citations are converted to [n] numbers, figure/table references are
numbered, and the IEEE author block is flattened. Styling comes from report/word_reference.docx.
"""
import re, subprocess, tempfile, shutil
from pathlib import Path

R = Path(__file__).resolve().parent
s = (R / 'seasonsafe.tex').read_text(encoding='utf-8')
head, bib = s.split('\\begin{thebibliography}{00}')
items = re.findall(r'\\bibitem\{(b\d+)\}\s*(.*)', bib.split('\\end{thebibliography}')[0])
num = {k: i + 1 for i, (k, _) in enumerate(items)}
head = re.sub(r'\\cite\{([^}]*)\}', lambda m: ', '.join('{[%d]}' % num[k.strip()] for k in m.group(1).split(',')), head)
head = re.sub(r'\\author\{.*?\n\n\\maketitle',
              lambda m: '\\author{Garima Rohilla (B2026126), Harleen Kaur (B2026128), Yashwanth M (B2026161)\\\\ '
                        'Goa Institute of Management, Big Data Analytics\\\\ '
                        'Code and data: \\url{https://github.com/YashwanthPb/SeasonSafe}}\n\\maketitle', head, flags=re.S)
head = head.replace('\\begin{IEEEkeywords}', '\\noindent\\textit{Index Terms}: ').replace('\\end{IEEEkeywords}', '')
head = head.replace('figure*', 'figure').replace('width=\\textwidth', 'width=\\linewidth')
for i, f in enumerate(['fig:rates', 'fig:cap', 'fig:models', 'fig:drift', 'fig:learned', 'fig:probs'], 1):
    head, n = re.subn(r'\\caption\{([^\n]*)\}\n\\label\{%s\}' % f, lambda m, i=i: '\\caption{Fig. %d. %s}' % (i, m.group(1)), head)
    assert n == 1, f
    head = head.replace('\\ref{%s}' % f, str(i))
for t, r in {'tab:rw': 'I', 'tab:policy': 'II', 'tab:abl': 'III'}.items():
    head, n = re.subn(r'\\caption\{([^\n}]*)\}\\label\{%s\}' % t, lambda m, r=r: '\\caption{TABLE %s. %s}' % (r, m.group(1)), head)
    assert n == 1, t
    head = head.replace('\\ref{%s}' % t, r)
head = head.replace('Fig.~', 'Fig. ').replace('Table~', 'Table ')
assert '\\ref{' not in head
refs = '\\section*{References}\n' + '\n\n'.join('[%d] %s' % (i + 1, t.strip()) for i, (_, t) in enumerate(items))
with tempfile.TemporaryDirectory() as d:
    tex = Path(d) / 'paper.tex'
    tex.write_text(head + refs + '\n\\end{document}\n', encoding='utf-8')
    out = R / 'SeasonSafe_IEEE_Report.docx'
    subprocess.run(['pandoc', str(tex), '-f', 'latex', '-t', 'docx', '--reference-doc', str(R / 'word_reference.docx'),
                    '--resource-path', f'{R}:{R / "figures"}', '-o', str(out)], check=True)
print('wrote', R / 'SeasonSafe_IEEE_Report.docx')
