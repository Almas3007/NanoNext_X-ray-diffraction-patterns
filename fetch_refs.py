"""Fetch full Crossref metadata (and abstracts, for the similarity check) for the approved references."""
import json, re, time, html, urllib.request, urllib.parse
from pathlib import Path
UA = {"User-Agent": "paper-ref-check/1.0 (mailto:adstechlearning@gmail.com)"}
APPROVED = {
 "park2017": "10.1107/S205225251700714X", "oviedo2019": "10.1038/s41524-019-0196-x",
 "lee2020": "10.1038/s41467-019-13749-3", "suzuki2020": "10.1038/s41598-020-77474-4",
 "schuetzke2021": "10.1107/S2052252521002402", "szymanski2021": "10.1021/acs.chemmater.1c01071",
 "dong2021": "10.1038/s41524-021-00542-4", "maffettone2021": "10.1038/s43588-021-00059-2",
 "salgado2023": "10.1038/s41524-023-01164-8", "szymanski2023": "10.1038/s41524-023-00984-y",
 "greasley2023": "10.1007/s10853-023-08343-4", "zhang2024": "10.1107/S2052252524005323",
 "szymanski2024": "10.1038/s41524-024-01230-9", "souesme2026": "10.1088/2632-2153/ae55f9",
 "yadav2026": "10.1063/5.0345499", "patterson1939": "10.1103/PhysRev.56.978",
 "holzwarth2011": "10.1038/nnano.2011.145", "mustapha2021": "10.1007/s40089-021-00338-w",
 "samanta2025": "10.1016/j.nxmate.2025.100841", "khorsandzak2025": "10.1038/s41598-025-31317-2",
 "zhuo2018": "10.1021/acs.jpclett.8b00124", "ward2016": "10.1038/npjcompumats.2016.28",
 "jain2013": "10.1063/1.4812323", "venkatraman2021": "10.1016/j.commatsci.2021.110637",
 "jung2024": "10.1021/acs.jcim.3c01897",
}
def get(url):
    return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30))
refs, src = [], Path("sources"); src.mkdir(exist_ok=True)
for key, doi in APPROVED.items():
    m = get("https://api.crossref.org/works/" + urllib.parse.quote(doi))["message"]
    authors = [f"{a.get('given','')} {a.get('family','')}".strip() if a.get("family") else a.get("name", "") for a in m.get("author", [])]
    yr = (m.get("published-print") or m.get("published-online") or m.get("issued"))["date-parts"][0][0]
    r = dict(key=key, type="article", authors=authors, title=html.unescape(re.sub("<[^>]+>", "", m["title"][0])),
             journal=(m.get("short-container-title") or m.get("container-title") or [""])[0] or m["container-title"][0],
             journal_full=m.get("container-title", [""])[0], volume=m.get("volume", ""), issue=m.get("issue", ""),
             pages=m.get("page", m.get("article-number", "")), year=yr, doi=m["DOI"])
    refs.append(r)
    ab = m.get("abstract", "")
    if not ab:
        try:
            ab = get(f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}?fields=abstract").get("abstract") or ""
        except Exception as e:
            ab = ""
    ab = html.unescape(re.sub("<[^>]+>", " ", ab))
    (src / f"{key}.txt").write_text(r["title"] + "\n\n" + ab)
    print(key, yr, len(authors), r["journal"], r["volume"], r["pages"], "abstract chars", len(ab), flush=True)
    time.sleep(1.2)
json.dump(refs, open("references.json", "w"), indent=1, ensure_ascii=False)
