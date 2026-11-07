"""Build the manuscript with the skill's builder: single-column layout, references in the
IEEE style NanoNEXT asks for (all authors up to six, otherwise the first three + et al.)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.home() / ".claude/skills/sci-journal-paper/scripts"))
import build_paper as bp

_orig = bp.format_reference


def nanonext_reference(r, style):
    r = dict(r)
    if len(r.get("authors", [])) > 6:
        r["authors"] = r["authors"][:3]
        txt = _orig(r, "ieee")
        first3 = ", ".join(bp.initials(a) for a in r["authors"])
        return txt.replace(bp.initials(r["authors"][0]) + ", " + bp.initials(r["authors"][1]) + ", and " + bp.initials(r["authors"][2]), first3 + " et al.", 1)
    return _orig(r, "ieee")


bp.format_reference = nanonext_reference

_orig_docx = bp.build_docx


def build_docx_keep_tables(meta, blocks, reg, refs_sorted, math, template, project, out, *args, **kw):
    """Build the DOCX, then keep each table (and its caption) on one page in Word."""
    _orig_docx(meta, blocks, reg, refs_sorted, math, template, project, out, *args, **kw)
    import docx
    d = docx.Document(out)
    for tbl in d.tables:
        for row in tbl.rows[:-1]:
            for cell in row.cells:
                for par in cell.paragraphs:
                    par.paragraph_format.keep_with_next = True
        prev = tbl._element.getprevious()
        for _ in range(2):   # "Table N" label and caption paragraph
            if prev is None or not prev.tag.endswith("}p"):
                break
            docx.text.paragraph.Paragraph(prev, d).paragraph_format.keep_with_next = True
            prev = prev.getprevious()
    d.save(out)


bp.build_docx = build_docx_keep_tables
if __name__ == "__main__":
    bp.main()
