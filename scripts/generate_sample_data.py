"""
Generate labeled contract / procurement document samples across 6 categories.
Run once: python scripts/generate_sample_data.py
"""

import csv
import random
from pathlib import Path

random.seed(18)
OUTPUT = Path(__file__).parent.parent / "data" / "sample" / "labeled_documents.csv"

CATEGORIES = {
    "RFP": [
        "The Government is soliciting proposals for professional services in support of",
        "Offerors are invited to submit technical and price proposals for",
        "This Request for Proposal covers information technology services including",
        "Proposals shall address all requirements outlined in the Performance Work Statement",
        "The contracting officer anticipates award of a firm-fixed-price contract for",
        "Section L of this solicitation provides instructions for proposal preparation",
        "Offerors must demonstrate technical capability and relevant past performance",
        "The Government intends to award a single Indefinite Delivery Indefinite Quantity contract",
        "All proposals must be submitted electronically via the SAM.gov portal",
        "The proposal evaluation criteria include technical approach, management, and price",
        "Offerors shall provide a detailed staffing plan and key personnel resumes",
        "The base period of performance is 12 months with four option years",
    ],
    "SOW": [
        "The contractor shall provide program management support services including",
        "The contractor is responsible for delivering the following tasks and deliverables",
        "Task 1: Program Planning and Coordination — the contractor shall develop",
        "The contractor shall maintain an integrated master schedule and report status weekly",
        "All deliverables shall be submitted in accordance with the Contract Deliverable Requirements List",
        "The contractor shall provide subject matter expertise in the following domains",
        "Work shall be performed on-site at the Government facility located in",
        "The contractor shall conduct quarterly performance reviews and submit written reports",
        "The contractor shall provide technical writing and documentation support",
        "The contractor shall support data collection analysis and reporting activities",
        "The Period of Performance for this task order is from date of award through",
        "The contractor shall not disclose any sensitive information obtained during performance",
    ],
    "Contract Modification": [
        "This modification is issued to incorporate the following changes to the contract",
        "The purpose of this modification is to extend the period of performance by",
        "Modification P00001 adds the following scope to the existing task order",
        "The total obligated amount is hereby increased from the original contract value",
        "This bilateral modification is agreed to by both parties and incorporated herein",
        "The contract is hereby modified to add the following contract line items",
        "Modification P00003 deobligates funds and reduces the total contract ceiling",
        "This unilateral modification is issued pursuant to FAR clause 52.243-1",
        "The delivery schedule is modified to reflect the revised completion date",
        "This change order modifies the statement of work to include additional tasks",
        "The contractor acknowledges receipt of this modification and agrees to the changes",
        "All other terms and conditions of the contract remain unchanged and in full force",
    ],
    "Invoice": [
        "Invoice Number INV-2024-0451 for professional services rendered during the period",
        "The contractor hereby submits this invoice for payment of labor hours incurred",
        "Direct labor costs billed at the negotiated fixed hourly rates per the contract",
        "Indirect costs including overhead and general and administrative expenses are billed",
        "This invoice covers the period of performance from the first through the last day",
        "Travel expenses incurred in direct support of contract performance are included",
        "Subcontractor costs as authorized under the contract are included on this invoice",
        "Payment is requested within 30 days of receipt per the prompt payment terms",
        "This is a cost-plus-fixed-fee invoice submitted for reimbursement of allowable costs",
        "The invoice total reflects labor materials and other direct costs for the billing period",
        "Invoices must be submitted via the Wide Area Workflow electronic invoicing system",
        "Backup documentation including timesheets and receipts is attached to this invoice",
    ],
    "Deliverable Report": [
        "This Monthly Status Report covers activities completed during the reporting period",
        "The Program Management Review slide deck summarizes progress against the integrated master schedule",
        "This After-Action Report documents findings and lessons learned from the exercise",
        "The Technical Design Document provides detailed specifications for the proposed system",
        "This Quality Assurance Surveillance Plan establishes the surveillance methodology",
        "The Transition Plan outlines steps for knowledge transfer to the incoming contractor",
        "This Risk Register identifies mitigation strategies for the top program risks",
        "The Final Report presents conclusions and recommendations based on the analysis conducted",
        "This Weekly Activity Report summarizes tasks completed issues encountered and planned work",
        "The Configuration Management Plan governs version control and change control procedures",
        "This Data Management Plan describes how data will be collected stored and protected",
        "The Corrective Action Plan addresses deficiencies identified during the performance review",
    ],
    "Compliance Document": [
        "This System Security Plan describes the security controls implemented for the information system",
        "The Contractor must comply with all applicable Federal Acquisition Regulation clauses",
        "This FedRAMP authorization package demonstrates compliance with NIST SP 800-53 controls",
        "The Privacy Impact Assessment evaluates risks to personally identifiable information",
        "Section 508 compliance testing confirms the system meets accessibility requirements",
        "The Continuity of Operations Plan ensures mission-critical functions during disruptions",
        "This Cybersecurity Maturity Model Certification assessment covers all required practices",
        "The contractor certifies compliance with the Buy American Act and Trade Agreements Act",
        "This Authority to Operate is granted based on an acceptable level of residual risk",
        "The contractor must submit annual compliance certifications including ethics and conflict of interest",
        "Data handling procedures comply with the Federal Information Security Modernization Act",
        "The Supply Chain Risk Management Plan addresses third-party hardware and software risks",
    ],
}


def make_doc(category: str, phrases: list[str]) -> str:
    n = random.randint(3, 5)
    selected = random.sample(phrases, n)
    connectors = [
        "{p}.",
        "{p}, as required by the contract.",
        "{p} in accordance with applicable regulations.",
        "{p} per the approved project plan.",
        "{p}, including all associated documentation.",
    ]
    return " ".join(random.choice(connectors).format(p=p) for p in selected)


rows = []
doc_id = 1
for category, phrases in CATEGORIES.items():
    n_docs = random.randint(55, 70)
    for _ in range(n_docs):
        rows.append({
            "doc_id":   f"DOC-{doc_id:04d}",
            "category": category,
            "text":     make_doc(category, phrases),
        })
        doc_id += 1

random.shuffle(rows)

with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["doc_id", "category", "text"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Generated {len(rows)} rows -> {OUTPUT}")
