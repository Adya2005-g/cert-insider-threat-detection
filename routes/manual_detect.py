from flask import Blueprint, flash, redirect, render_template, request, session, url_for

from services.manual_rules import evaluate_manual_behavior


manual_detect_bp = Blueprint("manual_detect", __name__)


@manual_detect_bp.post("/manual-detect")
def manual_detect():
    if not session.get("user_id"):
        flash("Please sign in to continue.", "error")
        return redirect(url_for("login_page"))

    try:
        analysis = evaluate_manual_behavior(request.form)
        return render_template(
            "behavior_profiling.html",
            analysis=analysis,
            form_data=analysis["inputs"],
        )
    except ValueError as exc:
        flash(str(exc), "error")
        return render_template(
            "behavior_profiling.html",
            analysis=None,
            form_data=request.form,
        ), 400
