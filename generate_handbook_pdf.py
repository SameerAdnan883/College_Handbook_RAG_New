"""
generate_handbook_pdf.py
Generates the fictional "Greenfield University College Handbook" PDF.
Run once: python generate_handbook_pdf.py
Requires: reportlab
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, HRFlowable,
    PageBreak, Table, TableStyle
)
import os

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "sample_college_handbook.pdf")

HANDBOOK_CONTENT = [
    # ─── Cover ────────────────────────────────────────────────────────────────
    {
        "type": "cover",
        "title": "GREENFIELD UNIVERSITY",
        "subtitle": "Student Handbook 2025–2026",
        "note": "⚠ FICTIONAL EDUCATIONAL DOCUMENT — For demonstration purposes only. "
                "Not affiliated with any real institution.",
    },
    # ─── Chapter 1 — Introduction ─────────────────────────────────────────────
    {
        "type": "chapter",
        "heading": "Chapter 1 — Welcome to Greenfield University",
        "body": [
            ("Introduction",
             "Welcome to Greenfield University (GU), a fictional institution created "
             "for this RAG demonstration. GU offers undergraduate and postgraduate "
             "programs across Engineering, Sciences, Arts, and Management."),
            ("Vision & Mission",
             "Vision: To be a world-class fictional university committed to academic "
             "excellence, research innovation, and holistic student development.\n"
             "Mission: Provide quality education, foster critical thinking, promote "
             "diversity, and prepare students for rewarding careers."),
            ("Academic Calendar 2025–2026",
             "Odd Semester: August 1 – December 20, 2025\n"
             "Exam Period (Odd): November 15 – December 15, 2025\n"
             "Winter Break: December 21, 2025 – January 5, 2026\n"
             "Even Semester: January 6 – May 31, 2026\n"
             "Exam Period (Even): April 20 – May 20, 2026\n"
             "Summer Vacation: June 1 – July 31, 2026"),
        ],
    },
    # ─── Chapter 2 — Attendance Policy ───────────────────────────────────────
    {
        "type": "chapter",
        "heading": "Chapter 2 — Attendance Policy",
        "body": [
            ("Minimum Attendance Requirement",
             "Students must maintain a minimum of 75% attendance in each course "
             "to be eligible to appear in end-semester examinations. Attendance is "
             "calculated separately for lectures, tutorials, and laboratory sessions."),
            ("Consequences of Low Attendance",
             "Below 75%: Debarred from appearing in end-semester examinations for "
             "that course.\n"
             "Below 60%: Automatic 'F' grade in the course; must repeat the course "
             "in the next academic year.\n"
             "Students debarred from examinations will have 'AB' (Absent/Debarred) "
             "marked on their grade sheet."),
            ("Medical & Exceptional Leave",
             "Students may apply for attendance condonation of up to 10% for "
             "genuine medical reasons with a certified medical certificate from a "
             "registered physician. The application must be submitted within 5 "
             "working days of rejoining.\n"
             "Sports, cultural, and university-deputed activities are granted "
             "special attendance by the Academic Dean upon prior approval."),
            ("Attendance Monitoring",
             "Attendance is recorded digitally via the GU Student Portal (gusportal.edu). "
             "Students can view their attendance dashboard in real time. "
             "SMS alerts are sent to students and parents when attendance falls below 80%."),
            ("Leave of Absence",
             "Students requiring extended leave (>2 weeks) must apply to the Registrar's "
             "office. Approved leave does not waive the minimum attendance requirement "
             "but qualifies for pro-rated condonation consideration by the Academic Committee."),
        ],
    },
    # ─── Chapter 3 — Examination Policy ──────────────────────────────────────
    {
        "type": "chapter",
        "heading": "Chapter 3 — Examination Policy",
        "body": [
            ("Examination Structure",
             "Each course is evaluated through:\n"
             "• Mid-Semester Examination (MSE) — 30 marks\n"
             "• End-Semester Examination (ESE) — 50 marks\n"
             "• Continuous Internal Assessment (CIA) — 20 marks\n"
             "Total: 100 marks per course"),
            ("Grading Scale",
             "O  (Outstanding) — 90–100 → 10 grade points\n"
             "A+ (Excellent)   — 80–89  → 9 grade points\n"
             "A  (Very Good)   — 70–79  → 8 grade points\n"
             "B+ (Good)        — 60–69  → 7 grade points\n"
             "B  (Average)     — 50–59  → 6 grade points\n"
             "C  (Satisfactory)— 45–49  → 5 grade points\n"
             "P  (Pass)        — 40–44  → 4 grade points\n"
             "F  (Fail)        — Below 40 → 0 grade points\n"
             "AB (Debarred)    — Ineligible → 0 grade points"),
            ("Re-Examination Policy",
             "Students who obtain an 'F' grade may apply for a supplementary "
             "examination held within 30 days of result declaration. "
             "A fee of ₹500 per paper applies. The maximum grade achievable in "
             "supplementary examinations is 'P' (Pass)."),
            ("Malpractice & Unfair Means",
             "Any student found using unfair means during examinations shall face:\n"
             "First offence: Grade 'F' in that subject and academic warning.\n"
             "Second offence: Suspension for one semester.\n"
             "Third offence: Expulsion from the university.\n"
             "All malpractice cases are reviewed by the Examination Disciplinary Committee."),
            ("Result Declaration",
             "Results are typically declared within 30 working days after the last "
             "examination. Official transcripts are available on the GU Student Portal. "
             "Provisional certificates are issued within 7 days of result declaration."),
            ("Revaluation",
             "Students may apply for answer-sheet revaluation within 10 days of result "
             "declaration. Revaluation fee: ₹300 per paper. If the revaluation results "
             "in a higher grade, the fee is refunded. The revaluated grade is final."),
        ],
    },
    # ─── Chapter 4 — Fee Structure ────────────────────────────────────────────
    {
        "type": "chapter",
        "heading": "Chapter 4 — Fee Structure & Payment Policy",
        "body": [
            ("Annual Fee Components (2025–2026)",
             "Tuition Fee:          ₹1,20,000 per year\n"
             "Development Fee:      ₹15,000 per year\n"
             "Examination Fee:      ₹5,000 per year\n"
             "Library Fee:          ₹3,000 per year\n"
             "Laboratory Fee:       ₹8,000 per year (applicable for science/engineering)\n"
             "Sports & Activity Fee:₹2,000 per year\n"
             "Student Welfare Fund: ₹1,000 per year\n"
             "Total (General):      ₹1,46,000 per year\n"
             "Total (Sci/Engg):     ₹1,54,000 per year"),
            ("Payment Schedule",
             "Semester 1 (Odd): Due by July 31 each year.\n"
             "Semester 2 (Even): Due by December 31 each year.\n"
             "Late payment attracts a fine of ₹100 per day up to 30 days, after which "
             "the student's portal access may be suspended."),
            ("Scholarships & Fee Waivers",
             "Merit Scholarship: Top 5% of each batch receive 50% tuition fee waiver.\n"
             "Need-Based Aid: Students from families with annual income below ₹3 lakh "
             "may apply for 100% tuition fee waiver via the Financial Aid Office.\n"
             "Government Scholarships (SC/ST/OBC/EBC): Processed through the National "
             "Scholarship Portal; GU coordinates directly with the scholarship authority.\n"
             "Sports Quota: Outstanding athletes receive 25% fee waiver with maintenance allowance."),
            ("Refund Policy",
             "Withdrawal before classes begin: 100% refund (excluding ₹5,000 registration fee).\n"
             "Withdrawal within 30 days of commencement: 80% refund.\n"
             "Withdrawal between 31–60 days: 50% refund.\n"
             "After 60 days: No refund."),
            ("Hostel Fee (Optional)",
             "Single Occupancy Room: ₹48,000 per year\n"
             "Double Occupancy Room: ₹36,000 per year\n"
             "Triple Occupancy Room: ₹28,000 per year\n"
             "Mess Charges: ₹24,000 per year (non-optional for hostel residents)\n"
             "Security Deposit (refundable): ₹10,000"),
        ],
    },
    # ─── Chapter 5 — Library Policy ──────────────────────────────────────────
    {
        "type": "chapter",
        "heading": "Chapter 5 — Library Policy & Services",
        "body": [
            ("Library Overview",
             "The Greenfield University Central Library holds over 80,000 physical volumes, "
             "access to 50,000+ e-books, and 120+ academic journal subscriptions. "
             "Library hours: Monday–Saturday 8:00 AM – 10:00 PM; "
             "Sunday and public holidays 10:00 AM – 6:00 PM."),
            ("Borrowing Privileges",
             "Undergraduate students: Up to 4 books for 14 days.\n"
             "Postgraduate students: Up to 6 books for 21 days.\n"
             "Research Scholars & Faculty: Up to 10 books for 30 days.\n"
             "Journals, reference books, and rare books are for in-library use only."),
            ("Fine & Overdue Policy",
             "Overdue books: ₹5 per book per day after the due date.\n"
             "Lost or damaged books: Replacement cost plus ₹200 processing fee.\n"
             "Uncleared library dues prevent examination registration and certificate issuance."),
            ("E-Library & Digital Resources",
             "Students have 24/7 access to the digital library through gusportal.edu/library. "
             "Available resources include: JSTOR, IEEE Xplore, Springer, Elsevier ScienceDirect, "
             "NPTEL video lectures, and GU's internal research repository.\n"
             "VPN access is required from off-campus."),
            ("Inter-Library Loan",
             "Students may request books not available in GU's collection via the "
             "inter-library loan (ILL) service. Requests take 3–7 working days. "
             "A processing fee of ₹50 per request applies."),
            ("Silence & Conduct",
             "The main reading hall is a strict silence zone. Mobile phones must be "
             "on silent mode. Food and beverages are prohibited inside the library. "
             "Violation may result in library access suspension for 30 days."),
        ],
    },
    # ─── Chapter 6 — Computer Laboratory Policy ──────────────────────────────
    {
        "type": "chapter",
        "heading": "Chapter 6 — Computer Laboratory Policy",
        "body": [
            ("Available Facilities",
             "GU has 6 computer laboratories with a combined capacity of 360 workstations. "
             "All systems run Windows 11 and Ubuntu 22.04 LTS (dual boot). "
             "Available software: Python 3.11, MATLAB R2024, Visual Studio Code, "
             "Eclipse IDE, Android Studio, MySQL 8.0, and licensed Microsoft Office."),
            ("Access Hours & Booking",
             "General lab hours: 8:00 AM – 8:00 PM on all working days.\n"
             "Extended hours (9:00 PM – 12:00 AM): Available for final-year project students "
             "with prior approval from the lab coordinator.\n"
             "Lab slots must be booked through gusportal.edu/labs at least 2 hours in advance."),
            ("Usage Rules",
             "Students must log in using their university credentials.\n"
             "Unauthorized software installation is strictly prohibited and may lead to "
             "account suspension.\n"
             "Internet use is restricted to academic purposes during scheduled classes.\n"
             "USB drives must be scanned at the lab entrance before use.\n"
             "Each session is logged; activity monitoring is in place."),
            ("Personal Device Policy",
             "Students may use personal laptops in designated areas of the lab.\n"
             "The university provides free Wi-Fi (GU-Campus-5G) up to 10 GB per day.\n"
             "The university is not responsible for damage or theft of personal devices."),
            ("Printing & Scanning",
             "Each student receives a free printing quota of 200 pages per semester.\n"
             "Additional prints: ₹2 per page (black & white) and ₹10 per page (colour).\n"
             "Scanning is free up to 50 pages per day."),
            ("Violation Consequences",
             "Minor violations (unauthorised sites, food in lab): Written warning.\n"
             "Major violations (hacking, data theft, damaging equipment): Suspension "
             "of lab access for the remainder of the semester and disciplinary action."),
        ],
    },
    # ─── Chapter 7 — Student Support Services ────────────────────────────────
    {
        "type": "chapter",
        "heading": "Chapter 7 — Student Support & Welfare Services",
        "body": [
            ("Academic Advising",
             "Every student is assigned a Faculty Advisor for academic guidance. "
             "Students are expected to meet their advisor at least once per month. "
             "Advisors assist with course selection, career planning, and academic difficulties."),
            ("Mental Health & Counselling",
             "The Student Wellness Centre (SWC) provides free, confidential counselling. "
             "Services include individual therapy, group sessions, stress management workshops, "
             "and crisis intervention.\n"
             "Hours: Monday–Friday 9:00 AM – 5:00 PM.\n"
             "Crisis helpline (24/7): +91-800-GREENFIELD\n"
             "All sessions are strictly confidential and do not appear on academic records."),
            ("Placement & Career Services",
             "The Career Development Centre (CDC) assists students with:\n"
             "• Resume building and review\n"
             "• Mock interviews and group discussions\n"
             "• On-campus and off-campus placement drives\n"
             "• Internship facilitation\n"
             "• Entrepreneurship support through GU StartupHub\n"
             "Average placement rate (2024): 87% within 6 months of graduation."),
            ("Student Clubs & Extracurricular",
             "GU has 45+ registered student clubs including Technical, Cultural, "
             "Social, Literary, and Sports clubs. Students are encouraged to join at "
             "least one club. Club activities are recognized in the Co-Curricular "
             "Achievement Certificate issued at graduation."),
            ("Anti-Ragging Policy",
             "GU has a zero-tolerance policy toward ragging. Any form of ragging — "
             "physical, verbal, or online — is a criminal offence under the UGC "
             "Anti-Ragging Regulations 2009 (fictional reference for this demo).\n"
             "Complaint mechanism: Anti-Ragging Helpline +91-800-NОРАGGING (24/7).\n"
             "Proven perpetrators face expulsion and criminal prosecution."),
            ("Grievance Redressal",
             "Students may submit grievances to:\n"
             "1. Department Grievance Committee (DGC) — first point of contact.\n"
             "2. University Grievance Redressal Committee (UGRC) — escalation.\n"
             "3. Ombudsperson — independent review for unresolved cases.\n"
             "Grievances must be resolved within 30 days of filing. "
             "Anonymous complaints are also accepted via gusportal.edu/grievance."),
            ("Disability & Accessibility Support",
             "GU is committed to inclusive education. Students with disabilities receive:\n"
             "• Assistive technology in labs and library.\n"
             "• Extended exam time (up to 50% additional time with documentation).\n"
             "• Priority classroom seating and accessible campus infrastructure.\n"
             "Contact the Disability Support Office (DSO) for personalized accommodation plans."),
        ],
    },
    # ─── Chapter 8 — Hostel & Campus Life ────────────────────────────────────
    {
        "type": "chapter",
        "heading": "Chapter 8 — Hostel & Campus Life",
        "body": [
            ("Hostel Allotment",
             "Hostel allotment is based on distance from hometown, academic merit, and "
             "availability. Applications open in June each year via gusportal.edu/hostel. "
             "Priority is given to first-year students and outstation students."),
            ("Hostel Rules",
             "In-time (weekdays): 10:00 PM. In-time (weekends): 11:00 PM.\n"
             "Visitors are allowed only in the common room between 9:00 AM and 7:00 PM.\n"
             "Cooking in rooms is prohibited. Kitchen facilities are available on every floor.\n"
             "Alcohol, tobacco, and non-prescription drugs are strictly banned on campus.\n"
             "Noise restrictions apply between 11:00 PM and 6:00 AM."),
            ("Campus Facilities",
             "• 24/7 Medical Centre with a resident doctor and nurse.\n"
             "• Sports complex: gymnasium, swimming pool, cricket ground, football field, "
             "basketball courts, and tennis courts.\n"
             "• Cafeteria serving breakfast, lunch, evening snacks, and dinner.\n"
             "• ATM, bank branch, post office, and stationery store on campus.\n"
             "• 24/7 campus security with CCTV surveillance."),
        ],
    },
    # ─── Chapter 9 — Code of Conduct ─────────────────────────────────────────
    {
        "type": "chapter",
        "heading": "Chapter 9 — Code of Conduct & Disciplinary Policy",
        "body": [
            ("Expected Behaviour",
             "Students are expected to maintain dignity, respect peers and staff, "
             "uphold academic integrity, and represent GU positively on and off campus."),
            ("Disciplinary Actions",
             "Level 1 — Counselling & Warning: Minor infractions (dress code, late submission).\n"
             "Level 2 — Fine & Probation: Repeated minor or first moderate infraction.\n"
             "Level 3 — Suspension: Serious misconduct, academic dishonesty.\n"
             "Level 4 — Expulsion: Extreme violations, criminal acts, repeat Level 3."),
            ("Academic Integrity",
             "Plagiarism in assignments, dissertations, or projects is a serious "
             "offence. GU uses plagiarism detection software for all final submissions. "
             "Similarity index above 20% triggers a review. Above 40% results in automatic "
             "failure of the submission."),
        ],
    },
    # ─── Chapter 10 — Contact Directory ──────────────────────────────────────
    {
        "type": "chapter",
        "heading": "Chapter 10 — Important Contacts",
        "body": [
            ("Administrative Offices",
             "Registrar's Office: registrar@gu.edu | +91-11-2345-6789\n"
             "Academic Dean's Office: dean.academic@gu.edu | +91-11-2345-6790\n"
             "Finance Office: finance@gu.edu | +91-11-2345-6791\n"
             "Admissions Office: admissions@gu.edu | +91-11-2345-6792"),
            ("Student Services",
             "Student Wellness Centre: swc@gu.edu | +91-11-2345-6800\n"
             "Career Development Centre: cdc@gu.edu | +91-11-2345-6801\n"
             "Library: library@gu.edu | +91-11-2345-6802\n"
             "Hostel Office: hostel@gu.edu | +91-11-2345-6803\n"
             "IT Help Desk: ithelpdesk@gu.edu | +91-11-2345-6804"),
            ("Emergency Contacts",
             "Campus Security (24/7): +91-11-2345-6900\n"
             "Medical Centre (24/7): +91-11-2345-6901\n"
             "Anti-Ragging Helpline (24/7): +91-800-NОРАGGING\n"
             "Student Crisis Line (24/7): +91-800-GREENFIELD"),
        ],
    },
]


def build_pdf(output_path: str = OUTPUT_PATH):
    """Build the fictional handbook PDF using reportlab."""
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "GUTitle", parent=styles["Title"],
        fontSize=28, textColor=colors.HexColor("#1a3a5c"),
        spaceAfter=10, leading=34,
    )
    subtitle_style = ParagraphStyle(
        "GUSubtitle", parent=styles["Normal"],
        fontSize=16, textColor=colors.HexColor("#2e6da4"),
        spaceAfter=8, leading=20,
    )
    note_style = ParagraphStyle(
        "GUNote", parent=styles["Normal"],
        fontSize=10, textColor=colors.HexColor("#cc0000"),
        backColor=colors.HexColor("#fff3f3"),
        borderColor=colors.HexColor("#cc0000"),
        borderWidth=1, borderPadding=6,
        spaceAfter=12,
    )
    chapter_style = ParagraphStyle(
        "GUChapter", parent=styles["Heading1"],
        fontSize=16, textColor=colors.HexColor("#1a3a5c"),
        spaceAfter=8, spaceBefore=12, leading=20,
    )
    section_style = ParagraphStyle(
        "GUSection", parent=styles["Heading2"],
        fontSize=12, textColor=colors.HexColor("#2e6da4"),
        spaceAfter=4, spaceBefore=6,
    )
    body_style = ParagraphStyle(
        "GUBody", parent=styles["Normal"],
        fontSize=10, leading=14, spaceAfter=6,
        textColor=colors.HexColor("#222222"),
    )

    story = []

    for block in HANDBOOK_CONTENT:
        if block["type"] == "cover":
            story.append(Spacer(1, 3 * cm))
            story.append(Paragraph(block["title"], title_style))
            story.append(Paragraph(block["subtitle"], subtitle_style))
            story.append(Spacer(1, 1 * cm))
            story.append(HRFlowable(width="100%", thickness=2,
                                    color=colors.HexColor("#2e6da4")))
            story.append(Spacer(1, 0.5 * cm))
            story.append(Paragraph(block["note"], note_style))
            story.append(PageBreak())

        elif block["type"] == "chapter":
            story.append(Paragraph(block["heading"], chapter_style))
            story.append(HRFlowable(width="100%", thickness=1,
                                    color=colors.HexColor("#aaccee")))
            story.append(Spacer(1, 0.3 * cm))
            for section_title, section_body in block["body"]:
                story.append(Paragraph(section_title, section_style))
                # Replace \n with <br/> for reportlab
                formatted = section_body.replace("\n", "<br/>")
                story.append(Paragraph(formatted, body_style))
                story.append(Spacer(1, 0.2 * cm))
            story.append(PageBreak())

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title="Greenfield University Student Handbook 2025-2026",
        author="Greenfield University (FICTIONAL)",
    )
    doc.build(story)
    print(f"✅ PDF generated: {output_path}")
    return output_path


if __name__ == "__main__":
    build_pdf()
