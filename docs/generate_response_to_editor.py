import os
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable

def create_docx(filename="docs/Response_to_Editor.docx"):
    doc = docx.Document()
    
    # Page setup
    sections = doc.sections
    for s in sections:
        s.top_margin = Inches(1.0)
        s.bottom_margin = Inches(1.0)
        s.left_margin = Inches(1.0)
        s.right_margin = Inches(1.0)
        
    # Styles
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Times New Roman'
    normal_style.font.size = Pt(11)
    normal_style.font.color.rgb = RGBColor(0x22, 0x22, 0x22)
    
    # Metadata Header
    p_meta = doc.add_paragraph()
    r = p_meta.add_run("Original Manuscript ID: ")
    r.bold = True
    p_meta.add_run("Access-2026-57395\n")
    r2 = p_meta.add_run("Original Article Title: ")
    r2.bold = True
    p_meta.add_run("“MobileViT Based Lightweight Road Extraction with Connectivity Aware Training for Rural Satellite Imagery”\n")
    r3 = p_meta.add_run("To: ")
    r3.bold = True
    p_meta.add_run("IEEE Access Editor\n")
    r4 = p_meta.add_run("Re: ")
    r4.bold = True
    p_meta.add_run("Response to Editor")
    
    doc.add_paragraph()
    
    # Salutation & Opening
    p_open = doc.add_paragraph()
    p_open.add_run("Dear Editor,\n\n")
    p_open.add_run("Thank you for allowing a resubmission of our manuscript, with an opportunity to address your comments and concerns.\n\n")
    p_open.add_run("We are uploading (a) our point-by-point response to your comments (below), (b) an updated manuscript with yellow highlighting indicating changes (under “Highlighted PDF”), and (c) a clean updated manuscript without highlights (“Main Manuscript”).\n\n")
    p_open.add_run("Best regards,\n")
    r_auth = p_open.add_run("Shanta Rangaswamy et al.")
    r_auth.bold = True
    
    doc.add_paragraph()
    
    # Concern 1
    p_c1 = doc.add_paragraph()
    r = p_c1.add_run("Concern # 1: ")
    r.bold = True
    r.font.color.rgb = RGBColor(0x00, 0x33, 0x66)
    p_c1.add_run("Table 2 gives accuracy scores from other papers. However, these methods used different training and test images.")
    
    p_r1 = doc.add_paragraph()
    r = p_r1.add_run("Author response: ")
    r.bold = True
    p_r1.add_run(
        "We agree completely with the Editor. Comparing accuracy metrics across disparate dataset splits (as was done in the previous Table 2) introduces confounding variables that compromise scientific rigor. "
        "Evaluating models on different training and test images does not provide an apples-to-apples basis for comparison. "
        "To eliminate this discrepancy, the previous Table 2 has been removed from the manuscript. Instead, architectural paradigms are discussed qualitatively in Section II, and all empirical accuracy comparisons have been consolidated into a unified benchmark under identical dataset conditions in Section V."
    )
    
    p_a1 = doc.add_paragraph()
    r = p_a1.add_run("Author action: ")
    r.bold = True
    p_a1.add_run(
        "We updated the manuscript by removing the previous Table 2 (the cross-study literature comparison). "
        "In Section II-E (“Reported Results”), we revised the text to explain that fair comparisons require evaluating baselines under identical training and testing partitions. "
        "All quantitative accuracy comparisons are now presented strictly in the new unified benchmark table (Table V) in Section V."
    )
    
    doc.add_paragraph()
    
    # Concern 2
    p_c2 = doc.add_paragraph()
    r = p_c2.add_run("Concern # 2: ")
    r.bold = True
    r.font.color.rgb = RGBColor(0x00, 0x33, 0x66)
    p_c2.add_run("In addition, Table 5 compares model size and speed but does not compare accuracy.")
    
    p_r2 = doc.add_paragraph()
    r = p_r2.add_run("Author response: ")
    r.bold = True
    p_r2.add_run(
        "We thank the Editor for this critical observation. In the original manuscript, Table 5 evaluated computational complexity (parameters, FLOPs, model size, and CPU latency) on laptop hardware, but omitted segmentation accuracy. "
        "Reporting computational efficiency in isolation fails to capture whether lightweight networks sacrifice vital topological connectivity and spatial precision. "
        "We have resolved this limitation by benchmarking the accuracy of the reference models alongside their computational cost."
    )
    
    p_a2 = doc.add_paragraph()
    r = p_a2.add_run("Author action: ")
    r.bold = True
    p_a2.add_run(
        "We updated the manuscript by expanding Table 5 into a comprehensive, multi-metric benchmark table (now Table V) that reports both segmentation accuracy (Road IoU, Relaxed F1, and Centerline Dice) and computational cost (Parameters, FLOPs, Model Size, and CPU Latency) for all evaluated architectures. "
        "Sections V-A and VI-A have been rewritten to analyze the multi-dimensional trade-offs between model efficiency and topological accuracy."
    )
    
    doc.add_paragraph()
    
    # Concern 3
    p_c3 = doc.add_paragraph()
    r = p_c3.add_run("Concern # 3: ")
    r.bold = True
    r.font.color.rgb = RGBColor(0x00, 0x33, 0x66)
    p_c3.add_run("Please compare the proposed model with existing methods using the same training and test images and report both accuracy and cost.")
    
    p_r3 = doc.add_paragraph()
    r = p_r3.add_run("Author response: ")
    r.bold = True
    p_r3.add_run(
        "We have fully addressed this requirement by benchmarking the proposed lightweight network alongside established segmentation architectures "
        "(U-Net, D-LinkNet34, DeepLabV3+ MobileNetV3, and LR-ASPP MobileNetV3) on the exact same 6,226-tile DeepGlobe dataset partition "
        "(5,323 training, 280 selection, and 623 held-out test tiles) evaluated at full native 1024×1024 resolution. "
        "The empirical results confirm that:\n"
        "1. Compared to the heavy U-Net baseline (31.04M parameters, 1541.4 GFLOPs), our proposed network achieves 5.6 to 6.9 points higher IoU (59.8% – 61.1% vs. 54.2%) and 22.7 points higher centerline Dice (84.5% vs. 61.8%), while reducing parameter count by 94.8% and executing 7.8× to 11.6× faster on CPU.\n"
        "2. Compared to the DeepGlobe challenge winner D-LinkNet34 (31.10M parameters, 212.6 GFLOPs), our network achieves higher IoU (61.1% vs. 58.1%) and substantially better topological connectivity (84.5% vs. 66.2% clDice) while requiring 19.4× fewer parameters and 3.5× fewer FLOPs.\n"
        "3. Compared to fast lightweight baselines (LR-ASPP MobileNetV3, 3.22M parameters), our model improves IoU by 9.7 points (61.1% vs. 51.4%) and clDice by 25.5 points (84.5% vs. 59.0%), demonstrating that lightweight CNNs lack the global receptive field needed to traverse canopy occlusions without our transformer and strip convolution priors."
    )
    
    p_a3 = doc.add_paragraph()
    r = p_a3.add_run("Author action: ")
    r.bold = True
    p_a3.add_run(
        "We updated the manuscript by:\n"
        "a) Incorporating the consolidated benchmark table (Table V) in Section V-A, reporting Parameters (M), GFLOPs, Weight Size (MB), CPU Latency (s), Road IoU (%), Relaxed F1 (%), and Centerline Dice (clDice, %).\n"
        "b) Revising Section I (Introduction), Section V (Results), and Section VI (Discussion) to detail the empirical findings of this identical-split benchmark.\n"
        "c) Removing the limitation item in Section VI-C that previously stated “No baselines on the same split”, and replacing it with an in-depth analysis of the observed accuracy-efficiency Pareto frontier.\n"
        "d) Highlighting all added and modified text and table entries in yellow in the resubmitted “Highlighted PDF”."
    )
    
    doc.save(filename)
    print(f"Saved: {filename}")

def create_pdf(filename="docs/Response_to_Editor.pdf"):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#0F172A'),
        spaceAfter=14
    )
    
    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#1E293B'),
        spaceAfter=8
    )
    
    concern_style = ParagraphStyle(
        'DocConcern',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=14.5,
        textColor=colors.HexColor('#003366'),
        spaceBefore=10,
        spaceAfter=6
    )
    
    label_style = ParagraphStyle(
        'DocLabel',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#0F172A')
    )
    
    story = []
    
    # Header metadata
    meta_text = (
        "<b>Original Manuscript ID:</b> Access-2026-57395<br/>"
        "<b>Original Article Title:</b> “MobileViT Based Lightweight Road Extraction with Connectivity Aware Training for Rural Satellite Imagery”<br/>"
        "<b>To:</b> IEEE Access Editor<br/>"
        "<b>Re:</b> Response to Editor"
    )
    story.append(Paragraph(meta_text, body_style))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#CBD5E1'), spaceAfter=14))
    
    # Letter Opening
    open_text = (
        "Dear Editor,<br/><br/>"
        "Thank you for allowing a resubmission of our manuscript, with an opportunity to address your comments and concerns.<br/><br/>"
        "We are uploading (a) our point-by-point response to your comments (below), (b) an updated manuscript with yellow highlighting indicating changes (under “Highlighted PDF”), and (c) a clean updated manuscript without highlights (“Main Manuscript”).<br/><br/>"
        "Best regards,<br/>"
        "<b>Shanta Rangaswamy et al.</b>"
    )
    story.append(Paragraph(open_text, body_style))
    story.append(Spacer(1, 12))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#E2E8F0'), spaceAfter=12))
    
    # Concern 1
    c1_title = "<b>Concern # 1:</b> Table 2 gives accuracy scores from other papers. However, these methods used different training and test images."
    c1_resp = (
        "<b>Author response:</b> We agree completely with the Editor. Comparing accuracy metrics across disparate dataset splits "
        "(as was done in the previous Table 2) introduces confounding variables that compromise scientific rigor. "
        "Evaluating models on different training and test images does not provide an apples-to-apples basis for comparison. "
        "To eliminate this discrepancy, the previous Table 2 has been removed from the manuscript. Instead, architectural paradigms are discussed qualitatively in Section II, and all empirical accuracy comparisons have been consolidated into a unified benchmark under identical dataset conditions in Section V."
    )
    c1_act = (
        "<b>Author action:</b> We updated the manuscript by removing the previous Table 2 (the cross-study literature comparison). "
        "In Section II-E (“Reported Results”), we revised the text to explain that fair comparisons require evaluating baselines under identical training and testing partitions. "
        "All quantitative accuracy comparisons are now presented strictly in the new unified benchmark table (Table V) in Section V."
    )
    story.append(Paragraph(c1_title, concern_style))
    story.append(Paragraph(c1_resp, body_style))
    story.append(Paragraph(c1_act, body_style))
    story.append(Spacer(1, 10))
    
    # Concern 2
    c2_title = "<b>Concern # 2:</b> In addition, Table 5 compares model size and speed but does not compare accuracy."
    c2_resp = (
        "<b>Author response:</b> We thank the Editor for this critical observation. In the original manuscript, Table 5 evaluated computational complexity "
        "(parameters, FLOPs, model size, and CPU latency) on laptop hardware, but omitted segmentation accuracy. "
        "Reporting computational efficiency in isolation fails to capture whether lightweight networks sacrifice vital topological connectivity and spatial precision. "
        "We have resolved this limitation by benchmarking the accuracy of the reference models alongside their computational cost."
    )
    c2_act = (
        "<b>Author action:</b> We updated the manuscript by expanding Table 5 into a comprehensive, multi-metric benchmark table (now Table V) that reports "
        "both segmentation accuracy (Road IoU, Relaxed F1, and Centerline Dice) and computational cost (Parameters, FLOPs, Model Size, and CPU Latency) for all evaluated architectures. "
        "Sections V-A and VI-A have been rewritten to analyze the multi-dimensional trade-offs between model efficiency and topological accuracy."
    )
    story.append(Paragraph(c2_title, concern_style))
    story.append(Paragraph(c2_resp, body_style))
    story.append(Paragraph(c2_act, body_style))
    story.append(Spacer(1, 10))
    
    # Concern 3
    c3_title = "<b>Concern # 3:</b> Please compare the proposed model with existing methods using the same training and test images and report both accuracy and cost."
    c3_resp = (
        "<b>Author response:</b> We have fully addressed this requirement by benchmarking the proposed lightweight network alongside established segmentation architectures "
        "(U-Net, D-LinkNet34, DeepLabV3+ MobileNetV3, and LR-ASPP MobileNetV3) on the exact same 6,226-tile DeepGlobe dataset partition "
        "(5,323 training, 280 selection, and 623 held-out test tiles) evaluated at full native 1024×1024 resolution. "
        "The empirical results confirm that:<br/>"
        "1. Compared to the heavy U-Net baseline (31.04M parameters, 1541.4 GFLOPs), our proposed network achieves 5.6 to 6.9 points higher IoU (59.8% – 61.1% vs. 54.2%) and 22.7 points higher centerline Dice (84.5% vs. 61.8%), while reducing parameter count by 94.8% and executing 7.8× to 11.6× faster on CPU.<br/>"
        "2. Compared to the DeepGlobe challenge winner D-LinkNet34 (31.10M parameters, 212.6 GFLOPs), our network achieves higher IoU (61.1% vs. 58.1%) and substantially better topological connectivity (84.5% vs. 66.2% clDice) while requiring 19.4× fewer parameters and 3.5× fewer FLOPs.<br/>"
        "3. Compared to fast lightweight baselines (LR-ASPP MobileNetV3, 3.22M parameters), our model improves IoU by 9.7 points (61.1% vs. 51.4%) and clDice by 25.5 points (84.5% vs. 59.0%), demonstrating that lightweight CNNs lack the global receptive field needed to traverse canopy occlusions without our transformer and strip convolution priors."
    )
    c3_act = (
        "<b>Author action:</b> We updated the manuscript by:<br/>"
        "a) Incorporating the consolidated benchmark table (Table V) in Section V-A, reporting Parameters (M), GFLOPs, Weight Size (MB), CPU Latency (s), Road IoU (%), Relaxed F1 (%), and Centerline Dice (clDice, %).<br/>"
        "b) Revising Section I (Introduction), Section V (Results), and Section VI (Discussion) to detail the empirical findings of this identical-split benchmark.<br/>"
        "c) Removing the limitation item in Section VI-C that previously stated “No baselines on the same split”, and replacing it with an in-depth analysis of the observed accuracy-efficiency Pareto frontier.<br/>"
        "d) Highlighting all added and modified text and table entries in yellow in the resubmitted “Highlighted PDF”."
    )
    story.append(Paragraph(c3_title, concern_style))
    story.append(Paragraph(c3_resp, body_style))
    story.append(Paragraph(c3_act, body_style))
    
    doc.build(story)
    print(f"Saved: {filename}")

if __name__ == "__main__":
    os.makedirs("docs", exist_ok=True)
    os.makedirs("paper", exist_ok=True)
    create_docx("docs/Response_to_Editor.docx")
    create_docx("paper/Response_to_Editor.docx")
    create_pdf("docs/Response_to_Editor.pdf")
    create_pdf("paper/Response_to_Editor.pdf")
