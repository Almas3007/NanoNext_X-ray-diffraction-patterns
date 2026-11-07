"""Title page, Other Information and cover-letter drafts for the NanoNEXT submission."""
import json
from pathlib import Path
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output_submission"
meta = json.load(open(ROOT / "meta.json"))
TITLE = meta["title"]


def new_doc():
    d = Document()
    st = d.styles["Normal"]; st.font.name = "Times New Roman"; st.font.size = Pt(12)
    st.paragraph_format.space_after = Pt(6); st.paragraph_format.line_spacing = 1.15
    return d


def head(d, text):
    p = d.add_paragraph(); r = p.add_run(text); r.bold = True
    return p


def para(d, text):
    return d.add_paragraph(text)


# Title page
d = new_doc()
p = d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run(TITLE); r.bold = True; r.font.size = Pt(14)
head(d, "Article type"); para(d, "Original Article")
head(d, "Authors"); para(d, "[TO CONFIRM: Full name(s) of all authors, in order, with affiliation superscripts]")
head(d, "Affiliations"); para(d, "[TO CONFIRM: a Department, Institution, City, Postal code, Country]")
head(d, "Corresponding author")
for line in ["Name: [TO CONFIRM]", "Email: [TO CONFIRM]", "Postal address: [TO CONFIRM]", "Country: [TO CONFIRM]", "ORCID: [TO CONFIRM]"]:
    para(d, line)
head(d, "Email addresses and ORCID iDs of all authors"); para(d, "[TO CONFIRM]")
head(d, "Keywords"); para(d, "; ".join(meta["keywords"]))
head(d, "Manuscript statistics")
para(d, "Abstract: 301 words; figures: 13; tables: 12; references: 24.")
d.save(OUT / "title_page.docx")

# Other Information
dec = json.load(open(ROOT / "meta.json"))["declarations"]
d = new_doc()
p = d.add_paragraph(); r = p.add_run("Other Information"); r.bold = True; r.font.size = Pt(14)
para(d, TITLE)
head(d, "Acknowledgements"); para(d, "[TO CONFIRM: acknowledgements, or 'Not applicable']")
head(d, "Funding details"); para(d, dec["funding"])
head(d, "Author contribution statement"); para(d, dec["credit"])
head(d, "Data availability statement")
para(d, "The crystal structures, simulated powder X-ray diffraction patterns and train/test split used in this study are from a public data set derived from Zhuo et al. (J. Phys. Chem. Lett. 9 (2018) 1668–1673, doi:10.1021/acs.jpclett.8b00124) and the Materials Project. The broadening code, training scripts, per-run results and figures are provided as supplementary material and are available from the corresponding author on reasonable request [TO CONFIRM: public repository link, e.g. Zenodo DOI].")
head(d, "Conflict of interest statement"); para(d, dec["competing_interest"])
head(d, "Ethics approval and consent"); para(d, "Not applicable. This study involves no human participants, animals or personal data.")
head(d, "Declaration of generative AI use"); para(d, dec["ai_declaration"])
d.save(OUT / "other_information.docx")

# Cover letter draft
d = new_doc()
for line in ["[TO CONFIRM: Date]", "", "The Editor-in-Chief", "NanoNEXT", "", f"Subject: Submission of an Original Article, “{TITLE}”", "", "Dear Editor,"]:
    d.add_paragraph(line)
for text in [
    "Please consider the enclosed manuscript for publication in NanoNEXT as an Original Article.",
    "Powder X-ray diffraction is usually the first characterisation step for a new nanomaterial, and machine-learning classifiers are increasingly used to interpret the patterns. These classifiers are almost always trained on simulated patterns of large crystals, whereas nanocrystals of a few to a few tens of nanometres produce strongly broadened peaks. The manuscript measures, size by size, how this Scherrer broadening affects crystal-system identification across 3,571 inorganic structures and crystallite sizes from 100 nm to 3 nm.",
    "The main findings are that a convolutional network trained on bulk patterns loses almost all of its accuracy even at 100–50 nm, assigning nearly every pattern to the triclinic class; that training with random crystallite sizes recovers about three quarters of the bulk performance at 3 nm; and that the augmentation must extend below the smallest expected size, since a model trained on 10–100 nm fails completely at 5 nm. The study also reports an audit of the public data set used, which affects other users of these structures.",
    "The work fits the journal's scope in characterisation, theoretical modelling and simulation of nanomaterials and nanostructures. The manuscript is original, has not been published previously and is not under consideration elsewhere. All authors have approved the submitted version. Code, results and figures are available as supplementary material.",
    "Thank you for considering this submission.",
]:
    para(d, text)
for line in ["", "Sincerely,", "[TO CONFIRM: Corresponding author name, affiliation, email]"]:
    d.add_paragraph(line)
d.save(OUT / "cover_letter.docx")
print("ok")
