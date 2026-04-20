from flask import Blueprint, flash, redirect, request, send_file, session, url_for

from models.log import Log
from services.report_export import export_batch_report


report_export_bp = Blueprint("report_export", __name__)


@report_export_bp.get("/reports/export/<batch_id>")
def export_report(batch_id: str):
    if not session.get("user_id"):
        flash("Please sign in to continue.", "error")
        return redirect(url_for("login_page"))

    export_format = request.args.get("format", "csv").strip().lower()
    if export_format not in {"csv", "xlsx"}:
        flash("Unsupported export format requested.", "error")
        return redirect(url_for("results", batch_id=batch_id))

    logs = (
        Log.query.filter_by(batch_id=batch_id)
        .order_by(Log.id.asc())
        .all()
    )
    if not logs:
        flash("No report data found for that batch.", "error")
        return redirect(url_for("results"))

    file_buffer, filename, mimetype = export_batch_report(logs, batch_id, export_format)
    return send_file(
        file_buffer,
        as_attachment=True,
        download_name=filename,
        mimetype=mimetype,
    )
