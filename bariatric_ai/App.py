import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names*")

import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import pandas as pd

from recommendations import generate_recommendations
from field_schema import FIELD_SCHEMA

MODEL_PATH = "bariatric_lgbm_bundle.joblib"


def get_meta(col: str):
    """
    Returns user-friendly metadata for a feature column.
    Fallback to raw column name if not defined in FIELD_SCHEMA.
    """
    meta = FIELD_SCHEMA.get(col, {})
    label = meta.get("label", col)
    unit = meta.get("unit", "")
    hint = meta.get("hint", "")
    ftype = meta.get("type", None)  # "num" | "bool" | "choice"
    choices = meta.get("choices", None)
    return label, unit, hint, ftype, choices


def infer_field_type(col: str):
    """
    Decide widget type. Prefer schema if available.
    """
    _, _, _, ftype, _ = get_meta(col)
    if ftype:
        return ftype

    # fallback heuristic
    if col == "sex":
        return "choice"
    binary_like = {
        "smoking", "dm2", "htn", "osa", "gerd", "nafld", "depression",
        "vomiting", "reflux", "diarrhea", "dumping",
    }
    if col in binary_like or (col.startswith("early_") and col.split("early_")[1] in binary_like):
        return "bool"
    return "num"


class BariatricApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Bariatrik Sleeve CDSS (AI) – Desktop")
        self.geometry("1050x760")

        # Load model bundle
        self.bundle = joblib.load(MODEL_PATH)
        self.feature_cols = self.bundle["feature_cols"]

        # Vars for UI
        self.vars = {}  # col -> tk variable

        # Mode (patient/clinician/both)
        self.mode_var = tk.StringVar(value="both")

        self._build_ui()

    def _build_ui(self):
        # Top controls
        top = ttk.Frame(self)
        top.pack(fill="x", padx=10, pady=8)

        ttk.Label(top, text="Natija rejimi:").pack(side="left")
        ttk.Radiobutton(top, text="Bemor", variable=self.mode_var, value="patient").pack(side="left", padx=6)
        ttk.Radiobutton(top, text="Shifokor", variable=self.mode_var, value="clinician").pack(side="left", padx=6)
        ttk.Radiobutton(top, text="Ikkalasi", variable=self.mode_var, value="both").pack(side="left", padx=6)

        ttk.Button(top, text="Hisoblash", command=self.on_predict).pack(side="right")
        ttk.Button(top, text="Tozalash", command=self.on_clear).pack(side="right", padx=8)

        # Scrollable form area
        container = ttk.Frame(self)
        container.pack(fill="both", expand=True, padx=10, pady=5)

        canvas = tk.Canvas(container)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        self.form_frame = ttk.Frame(canvas)

        self.form_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.form_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Build form fields dynamically
        self._build_form_fields()

        # Output area
        out = ttk.LabelFrame(self, text="Natija")
        out.pack(fill="both", expand=False, padx=10, pady=8)

        self.output = tk.Text(out, height=14, wrap="word")
        self.output.pack(fill="both", expand=True, padx=8, pady=6)

        # Disclaimer
        self.output.insert("end", "Diqqat: Bu dastur klinik qarorni almashtirmaydi.\n\n")

    def _build_form_fields(self):
        # Two columns layout
        left = ttk.Frame(self.form_frame)
        right = ttk.Frame(self.form_frame)
        left.grid(row=0, column=0, sticky="nsew", padx=8, pady=6)
        right.grid(row=0, column=1, sticky="nsew", padx=8, pady=6)

        self.form_frame.columnconfigure(0, weight=1)
        self.form_frame.columnconfigure(1, weight=1)

        half = (len(self.feature_cols) + 1) // 2
        cols_left = self.feature_cols[:half]
        cols_right = self.feature_cols[half:]

        def add_fields(frame, cols):
            for r, col in enumerate(cols):
                label, unit, hint, _, choices = get_meta(col)
                ftype = infer_field_type(col)

                # Label text (include unit if present)
                label_text = label + (f" ({unit})" if unit else "")
                ttk.Label(frame, text=label_text).grid(row=r, column=0, sticky="w", pady=2)

                if ftype == "choice":
                    # choices from schema, else default for sex
                    vals = choices if choices else (["F", "M"] if col == "sex" else [])
                    var = tk.StringVar(value=vals[0] if vals else "")
                    cb = ttk.Combobox(frame, textvariable=var, values=vals, width=14, state="readonly")
                    cb.grid(row=r, column=1, sticky="w", pady=2)
                    self.vars[col] = var

                elif ftype == "bool":
                    var = tk.IntVar(value=0)
                    chk = ttk.Checkbutton(frame, variable=var)
                    chk.grid(row=r, column=1, sticky="w", pady=2)
                    self.vars[col] = var

                else:
                    var = tk.StringVar(value="")
                    entry = ttk.Entry(frame, textvariable=var, width=16)
                    entry.grid(row=r, column=1, sticky="w", pady=2)
                    self.vars[col] = var

                # Hint (small text under the field)
                if hint:
                    ttk.Label(frame, text=hint, foreground="#666").grid(row=r, column=2, sticky="w", padx=8)

        add_fields(left, cols_left)
        add_fields(right, cols_right)

    def on_clear(self):
        for col, var in self.vars.items():
            ftype = infer_field_type(col)
            if ftype == "choice":
                # reset to first choice if exists
                _, _, _, _, choices = get_meta(col)
                vals = choices if choices else (["F", "M"] if col == "sex" else [])
                var.set(vals[0] if vals else "")
            elif ftype == "bool":
                var.set(0)
            else:
                var.set("")
        self.output.delete("1.0", "end")
        self.output.insert("end", "Diqqat: Bu dastur klinik qarorni almashtirmaydi.\n\n")

    def _parse_value(self, col: str, var):
        ftype = infer_field_type(col)
        if ftype == "choice":
            v = var.get().strip()
            return v if v != "" else None
        if ftype == "bool":
            return int(var.get())

        # numeric
        s = var.get().strip()
        if s == "":
            return None
        try:
            return float(s)
        except ValueError:
            label, _, _, _, _ = get_meta(col)
            raise ValueError(f"'{label}' uchun son kiriting (yoki bo‘sh qoldiring).")

    def on_predict(self):
        try:
            # Build features dict
            features = {col: self._parse_value(col, var) for col, var in self.vars.items()}

            # Build X for model (single row)
            X = pd.DataFrame([features], dtype=object)[self.feature_cols]

            # Predict risks
            p_readm = float(self.bundle["classifiers"]["readmission_30d"].predict_proba(X)[:, 1][0])
            p_comp  = float(self.bundle["classifiers"]["complication_90d"].predict_proba(X)[:, 1][0])
            p_def   = float(self.bundle["classifiers"]["any_def_12m"].predict_proba(X)[:, 1][0])
            twl     = float(self.bundle["regressor"].predict(X)[0])

            risks = {
                "p_readmission_30d": p_readm,
                "p_complication_90d": p_comp,
                "p_any_def_12m": p_def,
                "pred_twl_12m": twl,
            }

            rec = generate_recommendations(risks=risks, features=features, mode=self.mode_var.get())

            # Render output
            self.output.delete("1.0", "end")
            self.output.insert("end", "=== Prognozlar ===\n")
            self.output.insert("end", f"Readmission (30 kun): {p_readm*100:.2f}%\n")
            self.output.insert("end", f"Asorat (90 kun):     {p_comp*100:.4f}%\n")
            self.output.insert("end", f"Deficiency (12 oy):  {p_def*100:.2f}%\n")
            self.output.insert("end", f"TWL (12 oy):         {twl*100:.2f}%\n\n")

            self.output.insert("end", "=== Tavsiyalar ===\n")
            for i, item in enumerate(rec["items"], 1):
                self.output.insert("end", f"{i}. [P{item['priority']}] {item['title']}\n")
                if self.mode_var.get() in ("patient", "both"):
                    self.output.insert("end", f"   - (Bemor) {item.get('patient_text','')}\n")
                if self.mode_var.get() in ("clinician", "both"):
                    self.output.insert("end", f"   - (Shifokor) {item.get('clinician_text','')}\n")
                    tr = item.get("triggers", [])
                    if tr:
                        self.output.insert("end", f"   - Trigger: {', '.join(tr)}\n")
                self.output.insert("end", "\n")

            self.output.insert("end", rec["disclaimer_uz"] + "\n")

        except Exception as e:
            messagebox.showerror("Xato", str(e))


if __name__ == "__main__":
    app = BariatricApp()
    app.mainloop()