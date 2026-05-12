"""
SkinLytics Professional PDF Generator - Einseitiger Analysebericht
Klares, medizinisches Design - perfekt für Patienten und Ärzte.
"""

import io
import datetime
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm, cm
from reportlab.lib.colors import HexColor, white, black
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SkinLyticsPDFGenerator:
    """
    Professioneller, einseitiger PDF-Generator für Hautanalyse-Berichte.
    """

    # Professionelles, medizinisches Farbschema
    COLORS = {
        'primary': HexColor('#0F5B7A'),      # Teal - Vertrauenswürdig, medizinisch
        'primary_light': HexColor('#E6F3F8'), # Sehr hell für Hintergründe
        'accent_suspicious': HexColor('#C92A2A'),  # Rot für Warnung
        'accent_normal': HexColor('#2B8A3E'),      # Grün für unauffällig
        'text_dark': HexColor('#1E2A3A'),
        'text_medium': HexColor('#475569'),
        'text_light': HexColor('#94A3B8'),
        'border': HexColor('#E2E8F0'),
        'background': HexColor('#F8FAFC'),
    }

    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_styles()

    def _get_logo(self):
        """Lädt und gibt das Logo zurück, falls vorhanden."""
        try:
            logo_path = os.path.join(os.path.dirname(__file__), '..', '..', 'frontend', 'image.png')
            if os.path.exists(logo_path):
                return Image(logo_path, width=40*mm, height=40*mm)
            else:
                return ""
        except Exception as e:
            logger.warning(f"Logo konnte nicht geladen werden: {e}")
            return ""

    def _setup_styles(self):
        """Erstellt die benötigten Text-Styles"""

        # Klinik-Header
        self.styles.add(ParagraphStyle(
            name='ClinicHeader',
            parent=self.styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=18,
            textColor=self.COLORS['primary'],
            alignment=TA_CENTER,
            spaceAfter=4,
        ))

        # Untertitel Klinik
        self.styles.add(ParagraphStyle(
            name='ClinicSub',
            parent=self.styles['Normal'],
            fontName='Helvetica',
            fontSize=9,
            textColor=self.COLORS['text_medium'],
            alignment=TA_CENTER,
            spaceAfter=12,
        ))

        # Berichtstitel
        self.styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=self.styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=14,
            textColor=self.COLORS['text_dark'],
            alignment=TA_CENTER,
            spaceAfter=16,
        ))

        # Bereichsüberschrift
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=11,
            textColor=self.COLORS['primary'],
            spaceAfter=8,
            spaceBefore=4,
        ))

        # Normaler Text
        if 'BodyText' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='BodyText',
                parent=self.styles['Normal'],
                fontName='Helvetica',
                fontSize=9,
                leading=12,
                textColor=self.COLORS['text_medium'],
                alignment=TA_LEFT,
                spaceAfter=4,
            ))
        else:
            # Update existing BodyText style
            self.styles['BodyText'].fontName = 'Helvetica'
            self.styles['BodyText'].fontSize = 9
            self.styles['BodyText'].leading = 12
            self.styles['BodyText'].textColor = self.COLORS['text_medium']
            self.styles['BodyText'].alignment = TA_LEFT
            self.styles['BodyText'].spaceAfter = 4

        # Kleiner Text für Metadaten
        self.styles.add(ParagraphStyle(
            name='MetaText',
            parent=self.styles['Normal'],
            fontName='Helvetica',
            fontSize=8,
            textColor=self.COLORS['text_light'],
            alignment=TA_LEFT,
        ))

        # Ergebnis-Badge (unauffällig)
        self.styles.add(ParagraphStyle(
            name='ResultNormal',
            parent=self.styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=11,
            textColor=self.COLORS['accent_normal'],
            alignment=TA_CENTER,
            backColor=HexColor('#E8F5E9'),
            borderPadding=6,
            borderWidth=1,
            borderColor=self.COLORS['accent_normal'],
            borderRadius=4,
        ))

        # Ergebnis-Badge (verdächtig)
        self.styles.add(ParagraphStyle(
            name='ResultSuspicious',
            parent=self.styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=11,
            textColor=self.COLORS['accent_suspicious'],
            alignment=TA_CENTER,
            backColor=HexColor('#FEF2F2'),
            borderPadding=6,
            borderWidth=1,
            borderColor=self.COLORS['accent_suspicious'],
            borderRadius=4,
        ))

        # Konfidenz-Anzeige
        self.styles.add(ParagraphStyle(
            name='ConfidenceText',
            parent=self.styles['Normal'],
            fontName='Helvetica',
            fontSize=9,
            textColor=self.COLORS['text_medium'],
        ))

        # Disclaimer (klein)
        self.styles.add(ParagraphStyle(
            name='Disclaimer',
            parent=self.styles['Normal'],
            fontName='Helvetica-Oblique',
            fontSize=7,
            leading=9,
            textColor=self.COLORS['text_light'],
            alignment=TA_JUSTIFY,
        ))

        # Footer
        self.styles.add(ParagraphStyle(
            name='Footer',
            parent=self.styles['Normal'],
            fontName='Helvetica',
            fontSize=7,
            textColor=self.COLORS['text_light'],
            alignment=TA_CENTER,
        ))

    def generate_report(self, analysis_data, note_text=""):
        """
        Generiert einen EINSEITIGEN, professionellen PDF-Bericht.

        Args:
            analysis_data (dict): Analyseergebnisse
            note_text (str): Patientennotiz

        Returns:
            bytes: PDF als Bytes
        """
        buffer = io.BytesIO()

        # A4 mit Standardrändern
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            topMargin=18*mm,
            bottomMargin=18*mm,
            leftMargin=18*mm,
            rightMargin=18*mm,
            title="SkinLytics Hautanalyse-Bericht",
            author="SkinLytics",
        )

        story = []

        # ===== HEADER MIT LOGO =====
        header_table = Table([
            [Paragraph("SkinLytics", self.styles['ClinicHeader']), self._get_logo()],
            [Paragraph("KI-gestützte Hautanalyse | Medizinischer Bericht", self.styles['ClinicSub']), ""]
        ], colWidths=[120*mm, 50*mm])
        header_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(header_table)

        # Trennlinie
        line = Table([['']], colWidths=[170*mm], rowHeights=[1*mm])
        line.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, -1), self.COLORS['primary'])]))
        story.append(line)
        story.append(Spacer(1, 6*mm))

        # ===== METADATEN =====
        report_id = f"SK-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        analysis_date = analysis_data.get('date', datetime.datetime.now().strftime('%d.%m.%Y %H:%M'))

        meta_data = [
            [Paragraph("<b>Berichtsnummer:</b> " + report_id, self.styles['BodyText']), Paragraph("<b>Erstellt:</b> " + analysis_date, self.styles['BodyText'])],
            [Paragraph("<b>Analysezeitpunkt:</b> " + analysis_date, self.styles['BodyText']), Paragraph("", self.styles['BodyText'])],
        ]
        if note_text and note_text.strip():
            note_short = note_text[:120] + '...' if len(note_text) > 120 else note_text
            meta_data.append([Paragraph("<b>Notiz:</b> " + note_short, self.styles['BodyText']), Paragraph("", self.styles['BodyText'])])

        meta_table = Table(meta_data, colWidths=[85*mm, 85*mm])
        meta_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), self.COLORS['background']),
            ('GRID', (0, 0), (-1, -1), 0.3, self.COLORS['border']),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(meta_table)
        story.append(Spacer(1, 8*mm))

        # ===== ERGEBNIS =====
        story.append(Paragraph("Befund der Hautanalyse", self.styles['SectionHeader']))

        result_text = analysis_data.get('result_text', 'Analyse abgeschlossen')
        is_suspicious = analysis_data.get('is_suspicious', False)
        confidence = analysis_data.get('confidence', 0)

        # Prozentwert normalisieren
        if confidence <= 1:
            confidence_percent = round(confidence * 100)
        else:
            confidence_percent = round(confidence)

        # Ergebnis-Badge
        if is_suspicious:
            story.append(Paragraph(f"⚠️  {result_text}", self.styles['ResultSuspicious']))
        else:
            story.append(Paragraph(f"✓  {result_text}", self.styles['ResultNormal']))

        story.append(Spacer(1, 5*mm))

        # Konfidenz-Balken
        bar_filled = int(confidence_percent / 100 * 60)  # 60mm max
        bar_empty = 60 - bar_filled

        bar_color = self.COLORS['accent_normal'] if confidence_percent >= 70 else \
                   HexColor('#E67700') if confidence_percent >= 50 else \
                   self.COLORS['accent_suspicious']

        bar = Table([['', '']], colWidths=[bar_filled*mm, bar_empty*mm], rowHeights=[5*mm])
        bar.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, 0), bar_color),
            ('BACKGROUND', (1, 0), (1, 0), HexColor('#F1F5F9')),
            ('TOPPADDING', (0,0), (-1,-1), 0), 
            ('BOTTOMPADDING', (0,0), (-1,-1), 0)
        ]))

        conf_table = Table([
            [Paragraph(f"<b>KI-Konfidenz:</b> {confidence_percent}%", self.styles['ConfidenceText']), ""],
            [bar, ""],
        ], colWidths=[80*mm, 90*mm])
        conf_table.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
        story.append(conf_table)
        story.append(Spacer(1, 8*mm))

        # ===== KLASSIFIKATIONSTABELLE (kompakt) =====
        predictions = analysis_data.get('predictions', [])
        if predictions:
            story.append(Paragraph("Detaillierte Klassifikation", self.styles['SectionHeader']))

            # Tabellenkopf mit nur 2 Spalten
            table_data = [
                [Paragraph("<b>Läsionstyp</b>", self.styles['BodyText']),
                 Paragraph("<b>Wahrscheinlichkeit</b>", self.styles['BodyText'])]
            ]

            for pred in predictions[:5]:  # Max 5 Einträge für Einseitigkeit
                name = pred.get('name', 'Unbekannt')
                conf = pred.get('confidence', 0)
                if conf <= 1:
                    conf_percent = round(conf * 100)
                else:
                    conf_percent = round(conf)

                table_data.append([
                    Paragraph(name, self.styles['BodyText']),
                    Paragraph(f"{conf_percent}%", self.styles['BodyText']),
                ])

            # Tabelle erstellen mit 2 Spalten
            table = Table(table_data, colWidths=[100*mm, 60*mm])
            table.setStyle(TableStyle([
                # Kopf - nur 2 Spalten mit hellblauem Hintergrund
                ('BACKGROUND', (0, 0), (1, 0), self.COLORS['primary_light']),
                ('TEXTCOLOR', (0, 0), (1, 0), self.COLORS['text_dark']),
                ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (1, 0), 9),
                ('ALIGN', (0, 0), (1, 0), 'CENTER'),
                ('VALIGN', (0, 0), (1, 0), 'MIDDLE'),
                ('TOPPADDING', (0, 0), (1, 0), 6),
                ('BOTTOMPADDING', (0, 0), (1, 0), 6),
                # Daten - nur 2 Spalten
                ('GRID', (0, 1), (1, -1), 0.3, self.COLORS['border']),
                ('VALIGN', (0, 1), (1, -1), 'MIDDLE'),
                ('TOPPADDING', (0, 1), (1, -1), 5),
                ('BOTTOMPADDING', (0, 1), (1, -1), 5),
                ('LEFTPADDING', (0, 0), (1, -1), 6),
                ('RIGHTPADDING', (0, 0), (1, -1), 6),
            ]))
            story.append(table)
            story.append(Spacer(1, 8*mm))


        # ===== DISCLAIMER (kompakt) =====
        story.append(Paragraph("Wichtiger Hinweis", self.styles['SectionHeader']))

        disclaimer_lines = [
            "Dieser Bericht basiert auf einer KI-gestützten Bildanalyse und stellt KEINE ärztliche Diagnose dar.",
            "Die Ergebnisse sind rein informativ. Bei Auffälligkeiten konsultieren Sie bitte einen Dermatologen.",
            "Gemäß § 1295 ff. ABGB haftet SkinLytics nicht für Schäden aus der Interpretation dieses Berichts."
        ]
        
        for line in disclaimer_lines:
            story.append(Paragraph(line, self.styles['Disclaimer']))
            story.append(Spacer(1, 2*mm))
        story.append(Spacer(1, 3*mm))

        # DSGVO-Hinweis
        story.append(Paragraph(
            "<b>DSGVO:</b> Sie haben jederzeit Recht auf Auskunft, Löschung und Berichtigung Ihrer Daten. "
            "Kontakt: SkinLytics, Spengergasse 20, 1050 Wien | Skinlytics@outlook.com",
            self.styles['Disclaimer']
        ))

        # ===== FOOTER =====
        story.append(Spacer(1, 8*mm))
        footer_line = Table([['']], colWidths=[170*mm], rowHeights=[0.5*mm])
        footer_line.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, -1), self.COLORS['border'])]))
        story.append(footer_line)
        story.append(Spacer(1, 4*mm))

        footer_text = f"Bericht erstellt: {datetime.datetime.now().strftime('%d.%m.%Y %H:%M')} | SkinLytics® | Seite 1/1"
        story.append(Paragraph(footer_text, self.styles['Footer']))

        # ===== PDF GENERIEREN =====
        doc.build(story)
        buffer.seek(0)

        return buffer.getvalue()

    def create_pdf_from_analysis_result(self, analysis_result, note_text=""):
        """
        Komfort-Methode: Erstellt PDF aus API-Ergebnis.

        Args:
            analysis_result (dict): Ergebnis der Analyse
            note_text (str): Patientennotiz

        Returns:
            bytes: PDF-Datei
        """
        try:
            prediction_data = analysis_result.get('prediction', {})
            if not prediction_data:
                prediction_data = analysis_result

            # Vorhersagen parsen
            predictions = []
            if 'top_predictions' in prediction_data:
                predictions = prediction_data.get('top_predictions', [])
            elif 'predictions' in prediction_data:
                predictions = prediction_data.get('predictions', [])
            elif 'probabilities' in prediction_data:
                probs = prediction_data.get('probabilities', {})
                predictions = [{'name': k, 'confidence': v} for k, v in probs.items()]

            predictions.sort(key=lambda x: x.get('confidence', 0), reverse=True)

            # Verdächtig prüfen
            is_suspicious = prediction_data.get('is_suspicious', False)
            if not is_suspicious:
                suspicious_classes = ['melanom', 'mel', 'bcc', 'akiec', 'basalzell', 'aktinische']
                class_name = prediction_data.get('class_name', '').lower()
                is_suspicious = any(cls in class_name for cls in suspicious_classes)

            analysis_data = {
                'date': datetime.datetime.now().strftime('%d.%m.%Y %H:%M'),
                'result_text': prediction_data.get('class_name', 'Analyse abgeschlossen'),
                'is_suspicious': is_suspicious,
                'confidence': prediction_data.get('confidence', 0),
                'predictions': predictions,
                'model_version': 'HAM10000 Ensemble (ResNet50)'
            }

            return self.generate_report(analysis_data, note_text=note_text)

        except Exception as e:
            logger.error(f"PDF-Fehler: {e}")
            raise


# Test
if __name__ == "__main__":
    test_data = {
        'prediction': {
            'class_name': 'Aktinische Keratose - Abklärungsbedürftig',
            'confidence': 0.73,
            'is_suspicious': True,
            'top_predictions': [
                {'name': 'Aktinische Keratose', 'confidence': 0.73, 'suspicious': True},
                {'name': 'Basalzellkarzinom', 'confidence': 0.12, 'suspicious': True},
                {'name': 'Melanozytärer Nävus', 'confidence': 0.08, 'suspicious': False},
                {'name': 'Dermatofibrom', 'confidence': 0.04, 'suspicious': False},
                {'name': 'Melanom', 'confidence': 0.03, 'suspicious': True},
            ]
        }
    }

    gen = SkinLyticsPDFGenerator()
    pdf = gen.create_pdf_from_analysis_result(test_data, note_text="Patient berichtet über Veränderung in den letzten 3 Monaten.")

    with open('skinlytics_report.pdf', 'wb') as f:
        f.write(pdf)

    print("✅ PDF erstellt: skinlytics_report.pdf (1 Seite, professionell)")