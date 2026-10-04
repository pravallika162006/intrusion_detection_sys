"""
ReportLab PDF Generator for IDS Phase 4.
Generates Dataset Analysis Reports and Live Monitoring Session Reports.
"""

import io
from datetime import datetime
from typing import Dict, Any, List, Optional
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from backend.utils.logger import setup_logger

logger = setup_logger("PDFGenerator")


def build_dataset_pdf_report(data: Dict[str, Any]) -> bytes:
    """
    Generates a PDF report for Dataset Analysis & Prediction.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=6,
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#64748b"),
        spaceAfter=15,
    )
    h2_style = ParagraphStyle(
        "SectionHeader",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#0284c7"),
        spaceBefore=12,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "BodyTextCustom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#334155"),
    )
    callout_style = ParagraphStyle(
        "CalloutText",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#475569"),
    )

    story = []

    # Title & Header
    story.append(Paragraph("Intrusion Detection System — Dataset Analysis Report", title_style))
    report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    story.append(Paragraph(f"Generated on: {report_date} | Task: {data.get('prediction_task', 'binary').upper()}", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cbd5e1"), spaceAfter=12))

    # Executive Summary
    summary = data.get("summary", {})
    story.append(Paragraph("1. Analysis Configuration & Executive Summary", h2_style))
    
    cfg_data = [
        [Paragraph("<b>Dataset Reference:</b>", body_style), Paragraph(str(data.get("filename", "N/A")), body_style)],
        [Paragraph("<b>Feature Mode:</b>", body_style), Paragraph(f"{data.get('feature_mode', '19')} Selected Features", body_style)],
        [Paragraph("<b>Trained Model:</b>", body_style), Paragraph(str(data.get("model_name", "xgboost_dt")), body_style)],
        [Paragraph("<b>Total Records Analyzed:</b>", body_style), Paragraph(f"{summary.get('total_records', 0):,}", body_style)],
        [Paragraph("<b>Normal Traffic Count:</b>", body_style), Paragraph(f"{summary.get('normal_count', 0):,}", body_style)],
        [Paragraph("<b>Potential Attack Count:</b>", body_style), Paragraph(f"{summary.get('attack_count', 0):,}", body_style)],
        [Paragraph("<b>Attack Percentage:</b>", body_style), Paragraph(f"{summary.get('attack_percentage', 0.0)}%", body_style)],
    ]
    t_cfg = Table(cfg_data, colWidths=[160, 380])
    t_cfg.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('PADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_cfg)
    story.append(Spacer(1, 10))

    # Attack Category Distribution
    categories = summary.get("attack_categories")
    if categories:
        story.append(Paragraph("2. Predicted Attack Categories Distribution", h2_style))
        cat_rows = [[Paragraph("<b>Attack Category</b>", body_style), Paragraph("<b>Count</b>", body_style), Paragraph("<b>Percentage</b>", body_style)]]
        tot = max(1, summary.get("total_records", 1))
        for cat, cnt in categories.items():
            pct = round((cnt / tot) * 100, 2)
            cat_rows.append([Paragraph(str(cat), body_style), Paragraph(f"{cnt:,}", body_style), Paragraph(f"{pct}%", body_style)])
        
        t_cat = Table(cat_rows, colWidths=[200, 170, 170])
        t_cat.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#e0f2fe')),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
            ('PADDING', (0,0), (-1,-1), 4),
        ]))
        story.append(t_cat)
        story.append(Spacer(1, 10))

    # Evaluation Metrics (Ground-truth labelled mode vs unlabelled mode)
    eval_metrics = data.get("evaluation")
    story.append(Paragraph("3. Model Performance & Evaluation Metrics", h2_style))
    
    if eval_metrics:
        eval_data = [
            [Paragraph("<b>Metric</b>", body_style), Paragraph("<b>Score</b>", body_style), Paragraph("<b>User Explanation</b>", body_style)],
            [Paragraph("Accuracy", body_style), Paragraph(f"{eval_metrics.get('accuracy', 0)*100:.2f}%", body_style), Paragraph("Percentage of total predictions that perfectly matched ground-truth labels.", body_style)],
            [Paragraph("Precision", body_style), Paragraph(f"{eval_metrics.get('precision', 0):.4f}", body_style), Paragraph("Ratio of correctly identified attacks out of all predicted attacks.", body_style)],
            [Paragraph("Recall", body_style), Paragraph(f"{eval_metrics.get('recall', 0):.4f}", body_style), Paragraph("Ratio of actual attacks that were successfully detected by the model.", body_style)],
            [Paragraph("F1-Score", body_style), Paragraph(f"{eval_metrics.get('f1_score', 0):.4f}", body_style), Paragraph("Balanced harmonic mean of Precision and Recall.", body_style)],
            [Paragraph("Macro F1", body_style), Paragraph(f"{eval_metrics.get('macro_f1', 0):.4f}", body_style), Paragraph("Unweighted average F1 across all individual attack classes.", body_style)],
            [Paragraph("Weighted F1", body_style), Paragraph(f"{eval_metrics.get('weighted_f1', 0):.4f}", body_style), Paragraph("Class-frequency weighted average F1 score.", body_style)],
        ]
        t_eval = Table(eval_data, colWidths=[100, 70, 370])
        t_eval.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0284c7')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
            ('PADDING', (0,0), (-1,-1), 5),
        ]))
        story.append(t_eval)

        # Confusion Matrix
        cm = eval_metrics.get("confusion_matrix")
        if cm:
            story.append(Spacer(1, 8))
            story.append(Paragraph("<b>Confusion Matrix:</b>", body_style))
            cm_str = str(cm)
            story.append(Paragraph(f"<font fontName='Courier'>{cm_str}</font>", body_style))
    else:
        story.append(Paragraph("<i>This dataset is unlabelled. Ground-truth Accuracy, Precision, Recall, and F1 evaluation metrics cannot be calculated. Displaying prediction distribution only.</i>", callout_style))

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


def build_live_pdf_report(data: Dict[str, Any]) -> bytes:
    """
    Generates a PDF report for a Live Monitoring Session.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=6,
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#64748b"),
        spaceAfter=15,
    )
    h2_style = ParagraphStyle(
        "SectionHeader",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#0284c7"),
        spaceBefore=12,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "BodyTextCustom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#334155"),
    )
    callout_style = ParagraphStyle(
        "CalloutText",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#475569"),
    )

    story = []

    # Title & Subtitle
    story.append(Paragraph("Intrusion Detection System — Live Monitoring Session Report", title_style))
    report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    mode_str = "Controlled TEST MODE (Synthetic Traffic)" if data.get("test_mode") else f"Live Capture ({data.get('interface_name', 'Default')})"
    story.append(Paragraph(f"Generated on: {report_date} | Mode: {mode_str}", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cbd5e1"), spaceAfter=12))

    # Session Summary Table
    story.append(Paragraph("1. Monitoring Session Details", h2_style))
    packet_count = data.get("packet_count") if data.get("packet_count") is not None else data.get("packets_captured", 0)
    flow_count = data.get("flow_count") if data.get("flow_count") is not None else data.get("total_flows", 0)
    normal_count = data.get("normal_count") if data.get("normal_count") is not None else data.get("normal_flows", 0)
    attack_count = data.get("attack_count") if data.get("attack_count") is not None else data.get("attack_alerts", 0)
    duration_str = data.get("duration_formatted") or data.get("duration", "N/A")

    attack_pct = data.get("attack_percentage")
    if attack_pct is None:
        attack_pct = round((attack_count / flow_count) * 100, 1) if flow_count > 0 else 0.0

    sess_data = [
        [Paragraph("<b>Network Interface:</b>", body_style), Paragraph(str(data.get("interface_name", "N/A")), body_style)],
        [Paragraph("<b>Session Duration:</b>", body_style), Paragraph(str(duration_str), body_style)],
        [Paragraph("<b>Packets Captured:</b>", body_style), Paragraph(f"{packet_count:,}", body_style)],
        [Paragraph("<b>Flows Processed:</b>", body_style), Paragraph(f"{flow_count:,}", body_style)],
        [Paragraph("<b>Normal Predictions:</b>", body_style), Paragraph(f"{normal_count:,}", body_style)],
        [Paragraph("<b>Potential Attack Predictions:</b>", body_style), Paragraph(f"{attack_count:,}", body_style)],
        [Paragraph("<b>Attack Percentage:</b>", body_style), Paragraph(f"{attack_pct}%", body_style)],
    ]
    t_sess = Table(sess_data, colWidths=[160, 380])
    t_sess.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('PADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_sess)
    story.append(Spacer(1, 8))

    # Category Breakdown
    cats = data.get("category_distribution")
    if cats:
        story.append(Paragraph("2. Detected Category Breakdown", h2_style))
        cat_rows = [[Paragraph("<b>Category</b>", body_style), Paragraph("<b>Count</b>", body_style)]]
        for cat, cnt in cats.items():
            cat_rows.append([Paragraph(str(cat), body_style), Paragraph(f"{cnt:,}", body_style)])
        t_cat = Table(cat_rows, colWidths=[270, 270])
        t_cat.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#e0f2fe')),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
            ('PADDING', (0,0), (-1,-1), 4),
        ]))
        story.append(t_cat)
        story.append(Spacer(1, 8))

    # AI Security Advice & Incidents
    ai_recs = data.get("ai_recommendations", [])
    if ai_recs:
        story.append(Paragraph("3. AI Security Agent Incidents & Recommendations", h2_style))
        for idx, rec in enumerate(ai_recs[:5]):
            story.append(Paragraph(f"<b>Incident #{idx+1}: {rec.get('threat_summary', 'Attack Detected')}</b>", body_style))
            story.append(Paragraph(f"<b>Severity:</b> {rec.get('severity', 'Medium')}", body_style))
            story.append(Paragraph(f"<b>Explanation:</b> {rec.get('explanation', 'N/A')}", body_style))
            
            actions = rec.get("recommended_actions", [])
            if actions:
                story.append(Paragraph("<b>Recommended Security Actions:</b>", body_style))
                for act in actions:
                    story.append(Paragraph(f"  • {act}", body_style))
            story.append(Spacer(1, 6))

    # Operational Limitation Disclaimer
    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>Important Operational Limitation Notice:</b><br/>"
                           "<i>Live network traffic does not contain ground-truth labels. Normal/Attack values are model "
                           "predictions and should not be interpreted as measured accuracy.</i>", callout_style))

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


def build_performance_pdf_report(data: Dict[str, Any]) -> bytes:
    """
    Generates an Academic PDF Performance Report showing Phase 1, Phase 2, and Phase 3 results.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#64748b"),
        spaceAfter=12,
    )
    h2_style = ParagraphStyle(
        "SectionHeader",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#0284c7"),
        spaceBefore=10,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#334155"),
    )
    callout_style = ParagraphStyle(
        "Callout",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#475569"),
    )

    story = []
    story.append(Paragraph("Intrusion Detection System — Academic Performance Report", title_style))
    story.append(Paragraph("Sydney M. Kasongo & Yanxia Sun (2020) Reproduction & Phase 3 Enhancement Suite | UNSW-NB15", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cbd5e1"), spaceAfter=10))

    # Phase 1 Table
    p1 = data.get("phase1", {})
    p1_bin = p1.get("binary_results", [])
    if p1_bin:
        story.append(Paragraph("1. Phase 1: 42-Feature Baseline Binary Classification", h2_style))
        table_rows = [["Model", "Tr. AC (%)", "Val. AC (%)", "Test AC (%)", "Precision (%)", "Recall (%)", "F1 (%)"]]
        for r in p1_bin:
            table_rows.append([
                str(r.get("ML method", r.get("Model", ""))),
                str(r.get("Tr. AC (%)", r.get("Tr_AC", "N/A"))),
                str(r.get("Val. AC (%)", r.get("Val_AC", "N/A"))),
                str(r.get("Test AC (%)", r.get("Test_AC", "N/A"))),
                str(r.get("Precision (%)", "N/A")),
                str(r.get("Recall (%)", "N/A")),
                str(r.get("F1-Score (%)", "N/A")),
            ])
        t1 = Table(table_rows, colWidths=[90, 75, 75, 75, 75, 75, 75])
        t1.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0284c7')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
            ('PADDING', (0,0), (-1,-1), 3),
        ]))
        story.append(t1)
        story.append(Spacer(1, 8))

    # Phase 2 Table
    p2 = data.get("phase2", {})
    p2_bin = p2.get("binary_results", [])
    if p2_bin:
        story.append(Paragraph("2. Phase 2: XGBoost 19-Feature Binary Classification", h2_style))
        table_rows2 = [["Model", "Tr. AC (%)", "Val. AC (%)", "Test AC (%)", "Precision (%)", "Recall (%)", "F1 (%)"]]
        for r in p2_bin:
            table_rows2.append([
                str(r.get("ML method", r.get("Model", ""))),
                str(r.get("Tr. AC (%)", "N/A")),
                str(r.get("Val. AC (%)", "N/A")),
                str(r.get("Test AC (%)", "N/A")),
                str(r.get("Precision (%)", "N/A")),
                str(r.get("Recall (%)", "N/A")),
                str(r.get("F1-Score (%)", "N/A")),
            ])
        t2 = Table(table_rows2, colWidths=[90, 75, 75, 75, 75, 75, 75])
        t2.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0f766e')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
            ('PADDING', (0,0), (-1,-1), 3),
        ]))
        story.append(t2)
        story.append(Spacer(1, 8))

    # Phase 3 Table
    p3 = data.get("phase3", {})
    final_test = p3.get("final_test_results", [])
    if final_test:
        story.append(Paragraph("3. Phase 3: Project Enhancement — Final Frozen Model Performance", h2_style))
        table_rows3 = [["Task", "Selected Model", "Status", "Test AC (%)", "Precision (%)", "Recall (%)", "F1 (%)"]]
        for r in final_test:
            table_rows3.append([
                str(r.get("Task", "")),
                str(r.get("Selected Model", "")),
                str(r.get("Status", "")),
                str(r.get("Test Accuracy (%)", "N/A")),
                str(r.get("Precision (%)", "N/A")),
                str(r.get("Recall (%)", "N/A")),
                str(r.get("F1-Score (%)", "N/A")),
            ])
        t3 = Table(table_rows3, colWidths=[60, 160, 100, 60, 55, 55, 50])
        t3.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#4338ca')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
            ('PADDING', (0,0), (-1,-1), 3),
        ]))
        story.append(t3)
        story.append(Spacer(1, 8))

    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>Note on Academic Reproducibility:</b><br/>"
                           "<i>Results may differ from the reference paper because of implementation details, "
                           "preprocessing normalization, hardware capabilities, stochastic training, and model tuning. "
                           "Phase 3 is an independent extension evaluating candidate model combinations strictly on validation data.</i>", callout_style))

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes

