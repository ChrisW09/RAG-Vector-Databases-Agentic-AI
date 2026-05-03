"""Generate a small sample PDF for testing PoC 1.

Creates `sample_data/acme_handbook.pdf`, a fake 3-page employee handbook
with a few well-defined facts you can quiz the RAG app about.

Usage:
    pip install fpdf2
    python sample_data/make_sample_pdf.py
"""

from __future__ import annotations

from pathlib import Path

from fpdf import FPDF

OUT = Path(__file__).parent / "acme_handbook.pdf"

# Each top-level entry is one PDF page. Keep the facts deliberately
# distinctive so retrieval quality is easy to eyeball.
PAGES: list[tuple[str, list[str]]] = [
    (
        "1. Welcome to ACME Corp",
        [
            "ACME Corp was founded in 1997 by Wilma Coyote in Albuquerque, "
            "New Mexico. The company designs and manufactures specialised "
            "outdoor equipment for desert environments.",
            "",
            "Our headquarters moved to Phoenix, Arizona in 2014. As of "
            "January 2026 we employ 312 people across three offices: "
            "Phoenix (HQ), Berlin (EMEA), and Singapore (APAC).",
            "",
            "Our flagship product is the RoadRunner Trail Pack, a 38-litre "
            "hiking backpack with an integrated 2-litre hydration bladder.",
        ],
    ),
    (
        "2. Returns and refunds policy",
        [
            "Customers may return any item within 30 days of delivery for "
            "a full refund, provided the product is unused and in its "
            "original packaging. Shipping costs for the return are paid "
            "by the customer unless the item arrived damaged.",
            "",
            "Refunds are processed within 7 business days of the returned "
            "item arriving at our warehouse. Refunds are issued to the "
            "original payment method.",
            "",
            "Items marked as 'final sale' on the product page are not "
            "eligible for return. Gift cards are also non-refundable.",
            "",
            "For warranty claims (manufacturing defects discovered after "
            "the 30-day return window), please email support@acme.example "
            "with photos of the defect and your order number.",
        ],
    ),
    (
        "3. Employee remote-work policy",
        [
            "Full-time employees may work remotely up to three days per "
            "week. The remaining two days must be spent in the assigned "
            "office to support team collaboration and onboarding.",
            "",
            "Employees who relocate more than 50 kilometres from their "
            "assigned office must request approval from People Operations "
            "before changing their primary work location.",
            "",
            "Each employee receives a one-time home-office stipend of "
            "USD 1,200 to set up their remote workstation. The stipend "
            "may be used for a desk, chair, monitor, or ergonomic "
            "accessories. Receipts must be submitted within 60 days of "
            "the employee's start date.",
        ],
    ),
]


def build_pdf(out: Path = OUT) -> Path:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    for title, paragraphs in PAGES:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)
        pdf.set_font("Helvetica", "", 12)
        for p in paragraphs:
            if p == "":
                pdf.ln(3)
            else:
                pdf.multi_cell(0, 6, p)

    out.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(out))
    return out


if __name__ == "__main__":
    path = build_pdf()
    print(f"Wrote {path} ({path.stat().st_size:,} bytes, 3 pages)")
