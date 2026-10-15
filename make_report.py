"""Build the 2-page PDF write-up from results/*.json (numbers are never typed by hand).

    python make_report.py          ->  Fruit_Fly_Video_Analysis_Writeup.pdf
"""
import json
import os

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether, NextPageTemplate,
                                PageTemplate, Paragraph, Spacer, Table, TableStyle)

BASE = os.path.dirname(os.path.abspath(__file__))
R = lambda *p: os.path.join(BASE, "results", *p)

m = json.load(open(R("metrics.json")))
d = json.load(open(R("demo_summary.json")))
acc, sp = d["accuracy_vs_ground_truth"], d["speed"]
s1, s2 = m["stage1_fly_counting"], m["stage2_sex_identification"]
s3m, s3f, s4 = m["stage3_orientation_male"], m["stage3_orientation_female"], m["stage4_wing_angle"]
pca = m["pca_size_study"]
ds = m["dataset"]
wing_1 = pca["wing_angle_test_error_degrees_spread_std"][pca["number_of_components"].index(1)]
wing_40 = pca["wing_angle_test_error_degrees_spread_std"][pca["number_of_components"].index(40)]

NAVY = colors.HexColor("#12202E")
BLUE = colors.HexColor("#1F5FBF")
ORANGE = colors.HexColor("#E8872D")
GREEN = colors.HexColor("#2E8B57")
INK = colors.HexColor("#1B2733")
MUTED = colors.HexColor("#5B6B7B")
PALE = colors.HexColor("#F2F4F8")
LINE = colors.HexColor("#D9DEE6")

W, H = A4
LM = 0.75 * inch
CW = W - 2 * LM                                  # content width

body = ParagraphStyle("body", fontName="Helvetica", fontSize=9.4, leading=13.6, textColor=INK,
                      alignment=TA_LEFT, spaceAfter=4)
small = ParagraphStyle("small", parent=body, fontSize=8, leading=10.6, textColor=MUTED, spaceAfter=0)
cap = ParagraphStyle("cap", parent=small, spaceBefore=3, spaceAfter=6)
cell = ParagraphStyle("cell", parent=body, fontSize=8.6, leading=11.2, spaceAfter=0)
cell_b = ParagraphStyle("cell_b", parent=cell, fontName="Helvetica-Bold")
cell_h = ParagraphStyle("cell_h", parent=cell, fontName="Helvetica-Bold", textColor=colors.white)
bullet = ParagraphStyle("bullet", parent=body, leftIndent=12, bulletIndent=0, spaceAfter=3.5)
head = ParagraphStyle("head", fontName="Helvetica-Bold", fontSize=12.5, leading=15, textColor=NAVY)


def section(num, title):
    """Section heading with a coloured accent bar and breathing space above."""
    t = Table([["", Paragraph(f'<font color="#1F5FBF">{num}</font>&nbsp;&nbsp;{title}', head)]],
              colWidths=[4, CW - 4], rowHeights=[20])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, 0), ORANGE), ("LEFTPADDING", (1, 0), (1, 0), 9),
                           ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                           ("LEFTPADDING", (0, 0), (0, 0), 0), ("RIGHTPADDING", (0, 0), (0, 0), 0),
                           ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    return [Spacer(1, 8), t, Spacer(1, 5)]


def stat_cards(items):
    """Row of headline-number cards."""
    n = len(items)
    gap = 8
    cw = (CW - gap * (n - 1)) / n
    row, widths = [], []
    for i, (value, label, col) in enumerate(items):
        inner = Table([[Paragraph(f'<font size="19" color="{col}"><b>{value}</b></font>',
                                  ParagraphStyle("v", parent=body, leading=23, spaceAfter=0))],
                       [Paragraph(label, ParagraphStyle("l", parent=small, leading=10.5))]],
                      colWidths=[cw])
        inner.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), PALE), ("LEFTPADDING", (0, 0), (-1, -1), 10),
                                   ("RIGHTPADDING", (0, 0), (-1, -1), 6), ("TOPPADDING", (0, 0), (0, 0), 7),
                                   ("BOTTOMPADDING", (0, 1), (0, 1), 8), ("TOPPADDING", (0, 1), (0, 1), 0),
                                   ("BOTTOMPADDING", (0, 0), (0, 0), 1), ("LINEABOVE", (0, 0), (-1, 0), 2.2, ORANGE)]))
        row.append(inner)
        widths.append(cw)
        if i < n - 1:
            row.append("")
            widths.append(gap)
    t = Table([row], colWidths=widths)
    t.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                           ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))
    return t


def grid(rows, widths, header=True, zebra=True):
    t = Table(rows, colWidths=widths)
    st = [("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 3.6),
          ("BOTTOMPADDING", (0, 0), (-1, -1), 3.6), ("LEFTPADDING", (0, 0), (-1, -1), 7),
          ("RIGHTPADDING", (0, 0), (-1, -1), 7), ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE)]
    if header:
        st += [("BACKGROUND", (0, 0), (-1, 0), NAVY)]
    if zebra:
        for r in range(1 if header else 0, len(rows)):
            if (r % 2) == 0:
                st.append(("BACKGROUND", (0, r), (-1, r), PALE))
    t.setStyle(TableStyle(st))
    return t


def fig(path, width, caption):
    from PIL import Image as PILImage
    w, h = PILImage.open(path).size
    return KeepTogether([Image(path, width=width, height=width * h / w), Paragraph(caption, cap)])


# ------------------------------------------------------------------ page decoration
TITLE = "Real-time Detailed Video Analysis of Fruit Flies"


def first_page(c, doc):
    c.saveState()
    c.setFillColor(NAVY)
    c.rect(0, H - 104, W, 104, stroke=0, fill=1)
    c.setFillColor(ORANGE)
    c.rect(0, H - 108, W, 4, stroke=0, fill=1)
    size = 20
    while stringWidth(TITLE, "Helvetica-Bold", size) > CW:
        size -= 0.5
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", size)
    c.drawString(LM, H - 46, TITLE)
    c.setFillColor(colors.HexColor("#B8C4D3"))
    c.setFont("Helvetica", 10)
    c.drawString(LM, H - 64, "UE24CS352A  Machine Learning  |  Mini-Project Write-up")
    authors = [
        ("Sriranga Bhardwaj", "PES1UG24AM286"),
        ("Sumuk D Hegde", "PES1UG24AM293"),
        ("Shraddha", "PES1UG25AM812"),
    ]
    x = LM
    for name, srn in authors:
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 9)
        c.drawString(x, H - 86, name)
        x += stringWidth(name, "Helvetica-Bold", 9) + 4
        c.setFont("Helvetica", 9)
        c.setFillColor(colors.HexColor("#B8C4D3"))
        c.drawString(x, H - 86, srn)
        x += stringWidth(srn, "Helvetica", 9) + 16
    footer(c, doc)
    c.restoreState()


def later_page(c, doc):
    c.saveState()
    c.setFillColor(ORANGE)
    c.rect(0, H - 6, W, 6, stroke=0, fill=1)
    footer(c, doc)
    c.restoreState()


def footer(c, doc):
    c.setStrokeColor(LINE)
    c.setLineWidth(0.5)
    c.line(LM, 38, W - LM, 38)
    c.setFont("Helvetica", 8)
    c.setFillColor(MUTED)
    c.drawString(LM, 26, TITLE)
    c.drawRightString(W - LM, 26, f"Page {doc.page} of 2")


doc = BaseDocTemplate(os.path.join(BASE, "Fruit_Fly_Video_Analysis_Writeup.pdf"), pagesize=A4,
                      title=TITLE, author="Sriranga Bhardwaj, Sumuk D Hegde, Shraddha",
                      subject="UE24CS352A Machine Learning mini-project write-up")
doc.addPageTemplates([
    PageTemplate("first", [Frame(LM, 46, CW, H - 46 - 124, leftPadding=0, rightPadding=0, topPadding=0,
                                 bottomPadding=0)], onPage=first_page),
    PageTemplate("later", [Frame(LM, 46, CW, H - 46 - 28, leftPadding=0, rightPadding=0, topPadding=0,
                                 bottomPadding=0)], onPage=later_page)])

S = [NextPageTemplate("later")]

# headline numbers
fps_times = sp["times_faster_than_source_video"]
S += [stat_cards([(f"{acc['sex_correct_percent']:.1f}%", "male / female identified correctly", GREEN),
                  (f"{acc['heading_error_degrees_average']:.2f}°", "average heading error (0-360°)", BLUE),
                  (f"{acc['wing_angle_error_degrees_average']:.1f}°", "average wing-angle error (0-90°)", BLUE),
                  (f"{sp['frames_per_second']:.0f} FPS", f"analysis speed, {fps_times:.1f}x the 30 FPS video", ORANGE)])]

# 1 ---------------------------------------------------------------------------------
S += section("1", "Problem statement")
S += [Paragraph(
    "Fruit-fly behaviour, such as courtship, is studied by filming flies and analysing the video. Existing tools "
    "(for example FlyTracker) run roughly ten times slower than real time, which makes long recordings slow to process "
    "and rules out closed-loop experiments that must react while the fly is behaving.", body),
    Paragraph(
    "<b>Goal.</b> Analyse a grayscale video of one male and one female fly and, for <b>every frame</b>, output each "
    "fly's <b>sex, position and heading (0-360°)</b> and the male's <b>right and left wing angles (0-90°)</b>, "
    "faster than the 30 FPS video, on an ordinary CPU, with classical machine learning (re-implementing S. Herbst, "
    "Stanford CS229, 2018).", body)]

# 2 ---------------------------------------------------------------------------------
S += section("2", "Dataset details")
S += [Paragraph(
    "The lab video of the reference project is not public, so we generated our own data with a simulator "
    "(<font face='Courier'>src/synth.py</font>) that draws a smaller male and a larger, striped female in a circular "
    "well, with wings at any angle, legs, blur, noise and debris. Because the images are drawn, every label is exact.", body), Spacer(1, 2)]
facts = [[Paragraph("Item", cell_h), Paragraph("Details", cell_h)],
         [Paragraph("Labelled frames", cell_b), Paragraph(f"{ds['total_frames']} grayscale images (512 × 512): 78% flies apart, 10% close, 12% touching", cell)],
         [Paragraph("Train / test split", cell_b),
          Paragraph(f"{ds['training_frames']} training / {ds['test_frames']} test frames (2/3 : 1/3), split by frame", cell)],
         [Paragraph("Labels", cell_b),
          Paragraph("per fly: sex, position, heading 0-360°; for the male also right and left wing angle", cell)],
         [Paragraph("Test video", cell_b),
          Paragraph("separate 600-frame video (20 s at 30 FPS), different random seed, never used for training", cell)]]
S += [grid(facts, [100, CW - 100]), Spacer(1, 8)]
S += [fig(R("dataset_samples.png"), CW * 0.64,
          "Fig. 1  Simulated examples. When the flies touch they form one merged blob (middle), which the pipeline "
          "detects and skips.")]

# 3 ---------------------------------------------------------------------------------
S += section("3", "Approach")
S += [Paragraph(
    "Dark pixels are thresholded to find the fly bodies; four small models then work on those regions only, which "
    "keeps the pipeline fast (Fig. 2). <b>HOG</b> (histogram of oriented gradients) describes a shape by its edge "
    "directions and <b>PCA</b> compresses it (3,780 / 3,564 numbers to 15 / 40).", body)]
stages = [[Paragraph(x, cell_h) for x in ("Stage", "Model", "What it does")],
          [Paragraph("1  Fly counting", cell_b), Paragraph("Decision tree", cell),
           Paragraph("blob area: debris, one fly, or two touching", cell)],
          [Paragraph("2  Sex identification", cell_b), Paragraph("Logistic regression", cell),
           Paragraph("area + shape of both flies: which is the male", cell)],
          [Paragraph("3  Orientation", cell_b), Paragraph("HOG + PCA + logistic regression", cell),
           Paragraph("moments give the axis, classifier the head end", cell)],
          [Paragraph("4  Wing angles", cell_b), Paragraph("HOG + PCA + linear regression", cell),
           Paragraph("upright male, legs masked: right/left wing angle", cell)]]
S += [grid(stages, [96, 150, CW - 246]), Spacer(1, 8)]
S += [fig(R("pipeline_overview.png"), CW * 0.94,
          "Fig. 2  The pipeline on one frame (blue = male, red = female): threshold, contours, upright patches, wing "
          "regions, final output.")]


# 4 ---------------------------------------------------------------------------------
S += section("4", "Implementation overview")
for txt in (
        "<b>Python:</b> OpenCV (thresholding, contours, moments), scikit-image (HOG), scikit-learn (all models, PCA) "
        "and Tkinter (GUI).",
        "<font face='Courier'>generate_dataset.py</font> > <font face='Courier'>train_evaluate.py</font> > "
        "<font face='Courier'>demo.py</font> / <font face='Courier'>gui.py</font>: create the data, train and test the "
        "four models, then analyse a video with a plain-English report or a live window. The whole pipeline is "
        "<font face='Courier'>FlyPipeline.process(frame)</font> in <font face='Courier'>src/pipeline.py</font>.",
        "Submitted as a private GitHub repository; the README explains setup and the run order."):
    S.append(Paragraph(txt, bullet, bulletText="\u2022"))

# 5 ---------------------------------------------------------------------------------
S += section("5", "Results")
S += [Paragraph(
    "Each model was first tested on the 300 frames it never saw in training; the full pipeline was then run on the "
    "held-out 600-frame video (Table 1, Fig. 3).", body), Spacer(1, 2)]
res = [[Paragraph(x, cell_h) for x in ("Stage", "Test error", "Train error")],
       [Paragraph("1  Fly counting", cell), Paragraph(f"{s1['test_error_percent']:.2f}%", cell), Paragraph(f"{s1['training_error_percent']:.2f}%", cell)],
       [Paragraph("2  Sex identification", cell), Paragraph(f"{s2['test_error_percent']:.2f}%", cell), Paragraph(f"{s2['training_error_percent']:.2f}%", cell)],
       [Paragraph("3  Orientation, male", cell), Paragraph(f"{s3m['test_error_percent']:.2f}%", cell), Paragraph(f"{s3m['training_error_percent']:.2f}%", cell)],
       [Paragraph("3  Orientation, female", cell), Paragraph(f"{s3f['test_error_percent']:.2f}%", cell), Paragraph(f"{s3f['training_error_percent']:.2f}%", cell)],
       [Paragraph("4  Wing angles*", cell), Paragraph(f"{s4['test_error_degrees_spread_std']:.2f}°", cell), Paragraph(f"{s4['training_error_degrees_spread_std']:.2f}°", cell)]]
left = [Paragraph("<b>Table 1.</b> Error on unseen test frames and on training frames (lower is better).", small),
        Spacer(1, 4), grid(res, [96, 56, 56]), Spacer(1, 3),
        Paragraph("* spread (standard deviation) of the wing-angle error; the true angles vary with a spread of "
                  f"{s4['true_wing_angle_spread_std_degrees']:.0f}° (R<super>2</super> = {s4['r_squared']:.2f}).", small)]
chart = Image(R("wing_timeseries_compact.png"), width=CW - 224, height=(CW - 224) * 3.1 / 4.6)
right = [chart, Paragraph("Fig. 3  Predicted vs true wing angle on the test video.", cap)]
side = Table([[left, right]], colWidths=[222, CW - 222])
side.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                          ("RIGHTPADDING", (0, 0), (0, 0), 8), ("RIGHTPADDING", (1, 0), (1, 0), 0),
                          ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))
S += [side, Spacer(1, 6)]
S += [Paragraph(
    f"<b>On the test video</b> ({d['frames_processed']} frames, {acc['flies_checked']} flies checked) no frame had to be "
    f"skipped and the heading was within 15\u00B0 for {acc['heading_within_15_degrees_percent']:.1f}% of flies. The pipeline ran at "
    f"<b>{sp['frames_per_second']:.0f} frames per second</b> ({sp['milliseconds_per_frame']:.1f} ms per frame), {fps_times:.1f}x faster "
    "than the video, on a CPU; orientation and wing angles take about 80% of the time. (Speed was measured in our "
    "test environment and differs between computers.)", body)]

# 6 ---------------------------------------------------------------------------------
S += section("6", "Conclusions")
for txt in (
        f"Four small classical models recover sex, orientation and wing angles at <b>{fps_times:.1f}x real time</b> on a CPU.",
        "Image moments reduce orientation to a yes/no question (which end is the head), solved with "
        f"{s3m['test_error_percent']:.0f}% test error here; wing-angle error falls as more PCA components are used "
        f"({wing_1:.1f}\u00B0 spread with 1, {wing_40:.1f}\u00B0 with 40).",
        "<b>Limitations:</b> the data are simulated, so real footage will be harder; touching flies form one blob and are "
        "skipped. <b>Future work:</b> split touching flies, train on real annotated video."):
    S.append(Paragraph(txt, bullet, bulletText="\u2022"))
S += [Spacer(1, 4), Paragraph(
    "<b>References</b>  [1] S. Herbst, <i>Real-time Detailed Video Analysis of Fruit Flies</i>, Stanford CS229 project, "
    "2018.  [2] N. Dalal, B. Triggs, Histograms of Oriented Gradients for Human Detection, CVPR 2005.  "
    "[3] F. Pedregosa et al., Scikit-learn, JMLR 12, 2011.  [4] G. Bradski, The OpenCV Library, 2000.", small)]

doc.build(S)
print("report built:", doc.filename)
