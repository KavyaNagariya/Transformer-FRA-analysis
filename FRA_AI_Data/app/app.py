"""Flask web UI for FRA upload, analysis, visualization, and PDF export."""

import os
import sys
from datetime import datetime

from flask import Flask, render_template, request, send_file, abort

# Project root (FRA_AI_Data) on sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.pipeline import process_fra
from src.utils.report_generator import generate_report

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(BASE_DIR, "data")
RAW_DATA_FOLDER = os.path.join(BASE_DIR, "data", "raw")
REPORT_FOLDER = os.path.join(BASE_DIR, "reports")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RAW_DATA_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)


@app.route("/")
def index():
    return render_template("landing.html")


@app.route("/history")
def history():
    """List uploaded CSV files in the data directory for the history UI."""
    records = []
    if os.path.exists(UPLOAD_FOLDER):
        files = [
            f
            for f in os.listdir(UPLOAD_FOLDER)
            if f.endswith(".csv") and os.path.isfile(os.path.join(UPLOAD_FOLDER, f))
        ]
        for f in files:
            file_path = os.path.join(UPLOAD_FOLDER, f)
            mtime = os.path.getmtime(file_path)
            records.append(
                {
                    "id": f.replace(".csv", ""),
                    "date": datetime.fromtimestamp(mtime).strftime("%Y-%m-%d"),
                    "status": "Processed",
                }
            )
    return render_template("history.html", records=records)


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/analysis")
def diagnosis_dashboard():
    return render_template("index.html", status=None)


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        file = request.files.get("file")
        if not file or file.filename == "":
            return "Error: No file selected.", 400

        safe_name = os.path.basename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, safe_name)
        file.save(file_path)

        baseline_path = os.path.join(RAW_DATA_FOLDER, "fra_healthy.csv")
        reference_path = baseline_path if os.path.isfile(baseline_path) else None

        result = process_fra(file_path, reference_path=reference_path)

        if not result.get("ok"):
            err = result.get("error", "Unknown error")
            return f"Analysis failed: {err}", 500

        diagnosis = result["diagnosis"]
        corr_val = float(result.get("correlation", 0))
        if corr_val > 0.98:
            ui_status = "Healthy"
        elif corr_val > 0.90:
            ui_status = "Warning"
        else:
            ui_status = "Danger"
        report_name = f"fra_report_{result.get('run_id', 'latest')}.pdf"
        report_path = os.path.join(REPORT_FOLDER, report_name)
        try:
            generate_report(result, out_path=report_path)
        except Exception as ex:
            report_name = None
            app.logger.warning("PDF generation failed: %s", ex)

        return render_template(
            "index.html",
            status=ui_status,
            diagnosis_severity=diagnosis.get("severity", "Medium"),
            transformer_id=safe_name,
            date_now=datetime.now().strftime("%b %d, %Y %I:%M %p"),
            corr=round(float(result.get("correlation", 0)), 4),
            shift=round(float(result.get("max_deviation_db", 0)), 2),
            frequencies=_chart_frequencies(result),
            healthy=_chart_healthy(result),
            faulty=_chart_faulty(result),
            confidence=int(round(float(diagnosis.get("confidence", 0)))),
            fault_type=diagnosis.get("fault", "Unknown"),
            recommendation=diagnosis.get("recommendation", ""),
            explanation=diagnosis.get("explanation", ""),
            severity=diagnosis.get("severity", "Medium"),
            anomaly_score=round(float(diagnosis.get("anomaly_score", 0)), 2),
            features=result.get("features", {}),
            insights=result.get("insights", []),
            plot_single=result.get("plots", {}).get("single_url"),
            plot_compare=result.get("plots", {}).get("comparison_url"),
            plot_diff=result.get("plots", {}).get("difference_url"),
            report_file=report_name,
            ml_fault=result.get("ml", {}).get("fault"),
            ml_confidence=result.get("ml", {}).get("confidence"),
        )
    except Exception as e:
        import traceback

        app.logger.exception("analyze")
        return f"Internal Server Error: {e}\n{traceback.format_exc()}", 500


def _chart_frequencies(result: dict) -> list:
    """Rebuild overlap grid for Chart.js from saved paths via re-parse (lightweight: use pipeline file)."""
    # Prefer interpolating from parser if we had stored series; for UI we re-load from disk
    from src.parser import load_fra_data

    test = load_fra_data(result["file_path"])
    ref = load_fra_data(result["reference_path"])
    if test is None or ref is None:
        return []
    import numpy as np

    f_r = np.asarray(ref["Frequency"], dtype=float).ravel()
    m_r = np.asarray(ref["Magnitude"], dtype=float).ravel()
    f_t = np.asarray(test["Frequency"], dtype=float).ravel()
    f_min = max(float(f_r.min()), float(f_t.min()))
    f_max = min(float(f_r.max()), float(f_t.max()))
    if f_max <= f_min:
        return [f_min]
    grid = np.linspace(f_min, f_max, min(600, max(50, len(f_t))))
    return grid.tolist()


def _chart_healthy(result: dict) -> list:
    from src.parser import load_fra_data
    import numpy as np

    test = load_fra_data(result["file_path"])
    ref = load_fra_data(result["reference_path"])
    if test is None or ref is None:
        return []
    f_r = np.asarray(ref["Frequency"], dtype=float).ravel()
    m_r = np.asarray(ref["Magnitude"], dtype=float).ravel()
    f_t = np.asarray(test["Frequency"], dtype=float).ravel()
    f_min = max(float(f_r.min()), float(f_t.min()))
    f_max = min(float(f_r.max()), float(f_t.max()))
    if f_max <= f_min:
        grid = np.array([f_min])
    else:
        grid = np.linspace(f_min, f_max, min(600, max(50, len(f_t))))
    mag_h = np.interp(grid, f_r, m_r)
    return mag_h.tolist()


def _chart_faulty(result: dict) -> list:
    from src.parser import load_fra_data
    import numpy as np

    test = load_fra_data(result["file_path"])
    ref = load_fra_data(result["reference_path"])
    if test is None or ref is None:
        return []
    f_r = np.asarray(ref["Frequency"], dtype=float).ravel()
    f_t = np.asarray(test["Frequency"], dtype=float).ravel()
    m_t = np.asarray(test["Magnitude"], dtype=float).ravel()
    f_min = max(float(f_r.min()), float(f_t.min()))
    f_max = min(float(f_r.max()), float(f_t.max()))
    if f_max <= f_min:
        grid = np.array([f_min])
    else:
        grid = np.linspace(f_min, f_max, min(600, max(50, len(f_t))))
    mag_u = np.interp(grid, f_t, m_t)
    return mag_u.tolist()


@app.route("/download-report")
def download_report():
    """Download a generated PDF from the reports folder (basename only)."""
    name = request.args.get("f", "")
    if not name or name != os.path.basename(name):
        abort(400)
    path = os.path.join(REPORT_FOLDER, name)
    if not os.path.isfile(path):
        abort(404)
    return send_file(path, as_attachment=True, download_name=name)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
