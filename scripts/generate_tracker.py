import os
import json
import glob
from datetime import datetime

EXP_DIR = "/home/leonard/Documents/ENAC/cl-embedded/experiments"
OUTPUT_FILE = "/home/leonard/Documents/ENAC/cl-embedded/docs/experiments_tracker.md"

SUPERVISED_MODELS = {"ewc", "hdc", "tinyol", "tinyol_ae"}


def read_metrics(results_dir: str, model: str) -> tuple[str, str, str]:
    """Return (status_icon, metric_str, af_str) from any known metrics format."""
    if not os.path.isdir(results_dir):
        return "🔴", "N/A", ""

    # Type A: metrics.json  →  data["cl_metrics"][model]["aa"]
    path_a = os.path.join(results_dir, "metrics.json")
    if os.path.exists(path_a):
        try:
            with open(path_a) as f:
                data = json.load(f)
            cl = data.get("cl_metrics", {})
            model_key = model if model in cl else next(
                (k for k in cl if k not in {"memory", "joint", "naive"}), None
            )
            if model_key and "aa" in cl.get(model_key, {}):
                aa = cl[model_key]["aa"]
                af = cl[model_key].get("af")
                af_str = f" / AF: {af:.3f}" if af is not None else ""
                return "🟢", f"Avg Acc: {aa:.3f}{af_str}", af_str
        except Exception:
            pass
        return "🟢", "N/A", ""

    # Type B: metrics_cl.json  →  data["acc_final"]
    path_b = os.path.join(results_dir, "metrics_cl.json")
    if os.path.exists(path_b):
        try:
            with open(path_b) as f:
                data = json.load(f)
            aa = data.get("acc_final")
            af = data.get("avg_forgetting")
            metric = f"Avg Acc: {aa:.3f}" if aa is not None else "N/A"
            af_str = f" / AF: {af:.3f}" if af is not None else ""
            if aa is not None:
                metric += af_str
            return "🟢", metric, af_str
        except Exception:
            pass
        return "🟢", "N/A", ""

    # Type C: metrics_single_task.json  →  data["accuracy"]
    path_c = os.path.join(results_dir, "metrics_single_task.json")
    if os.path.exists(path_c):
        try:
            with open(path_c) as f:
                data = json.load(f)
            acc = data.get("accuracy")
            f1 = data.get("f1")
            parts = []
            if acc is not None:
                parts.append(f"Acc: {acc:.3f}")
            if f1 is not None:
                parts.append(f"F1: {f1:.3f}")
            return "🟢", " / ".join(parts) or "N/A", ""
        except Exception:
            pass
        return "🟢", "N/A", ""

    # Type D: cl_anomaly_metrics_*.json  →  data["avg_auroc"]
    anomaly_files = glob.glob(os.path.join(results_dir, "cl_anomaly_metrics_*.json"))
    if anomaly_files:
        try:
            with open(anomaly_files[0]) as f:
                data = json.load(f)
            auroc = data.get("avg_auroc")
            metric = f"Avg AUROC: {auroc:.3f}" if auroc is not None else "N/A"
            return "🟢", metric, ""
        except Exception:
            pass
        return "🟢", "N/A", ""

    return "🔴", "N/A", ""


def parse_experiments() -> list[dict]:
    folders = [
        f for f in os.listdir(EXP_DIR)
        if f.startswith("exp_") and os.path.isdir(os.path.join(EXP_DIR, f))
    ]

    def sort_key(name: str) -> tuple:
        parts = name.split("_")
        try:
            return (int(parts[1]), name)
        except (IndexError, ValueError):
            return (9999, name)

    exps = []
    for folder in sorted(folders, key=sort_key):
        parts = folder.split("_")
        if len(parts) < 3:
            continue

        model = parts[2]
        scenario = "_".join(parts[3:])

        if "pump" in folder:
            dataset = "Pump"
        elif "monitoring" in folder:
            dataset = "Equipment"
        elif "cwru" in folder:
            dataset = "CWRU"
        elif "pronostia" in folder:
            dataset = "Pronostia"
        elif "dataset2" in folder:
            dataset = "Equipment"
        else:
            dataset = "Unknown"

        results_dir = os.path.join(EXP_DIR, folder, "results")
        status_icon, metric_str, _ = read_metrics(results_dir, model)
        status = "🟢 Terminé" if status_icon == "🟢" else "🔴 Échec / ⏳ En cours"
        learning_type = "Supervisé" if model in SUPERVISED_MODELS else "Non-supervisé"

        exps.append({
            "folder": folder,
            "dataset": dataset,
            "scenario": scenario,
            "model": model,
            "type": learning_type,
            "status": status,
            "status_icon": status_icon,
            "metric": metric_str,
        })

    return exps


def generate_markdown(exps: list[dict]) -> str:
    md = "# 🧪 Suivi des Expériences (Experiments Tracker)\n\n"
    md += (
        "Ce document fournit une vue d'ensemble de toutes tes expériences. "
        "Il a été conçu pour être **simple à comprendre** au premier coup d'œil "
        "et **facile à mettre à jour**.\n\n"
    )
    md += f"> Généré automatiquement le {datetime.now().strftime('%Y-%m-%d %H:%M')} "
    md += "par `scripts/generate_tracker.py`.\n\n"
    md += "--- \n\n## 🌳 Vue Arborescente (Graphe Mermaid)\n\n"
    md += "```mermaid\ngraph LR\n"
    md += "    classDef dataset fill:#e1bee7,stroke:#8e24aa,stroke-width:2px,color:#000;\n"
    md += "    classDef scenario fill:#bbdefb,stroke:#1e88e5,stroke-width:2px,color:#000;\n"
    md += "    classDef run_success fill:#c8e6c9,stroke:#43a047,stroke-width:1px,color:#000;\n"
    md += "    classDef run_fail fill:#ffcdd2,stroke:#e53935,stroke-width:1px,color:#000;\n\n"
    md += "    Root[🔍 Expériences CL]\n\n"

    # Organize by dataset → scenario
    datasets: dict[str, dict[str, list]] = {}
    for exp in exps:
        datasets.setdefault(exp["dataset"], {}).setdefault(exp["scenario"], []).append(exp)

    for ds_id, ds_name in enumerate(datasets):
        ds_node = f"DS_{ds_id}"
        md += f"    Root --> {ds_node}[{ds_name}]:::dataset\n"
        for sc_id, sc_name in enumerate(datasets[ds_name]):
            sc_node = f"SC_{ds_id}_{sc_id}"
            md += f"    {ds_node} --> {sc_node}[{sc_name}]:::scenario\n"
            for m_id, exp in enumerate(datasets[ds_name][sc_name]):
                m_node = f"M_{ds_id}_{sc_id}_{m_id}"
                label = f"{exp['model']} {exp['status_icon']}"
                class_type = "run_success" if exp["status_icon"] == "🟢" else "run_fail"
                md += f"    {sc_node} --> {m_node}[{label}]:::{class_type}\n"

    md += "```\n\n"
    md += "> **Légende** : 🟢 Succès / Terminé | 🔴 Échec / Absent / En cours\n\n"

    # Tabular view
    md += "--- \n\n## 📌 Vue Tabulaire\n\n"
    md += "| Dataset | Scénario | Apprentissage | Modèle | Statut | Métriques Clés | Dossier |\n"
    md += "|---------|----------|---------------|--------|--------|----------------|---------|\n"
    for exp in exps:
        md += (
            f"| **{exp['dataset']}** | {exp['scenario']} | {exp['type']} "
            f"| {exp['model']} | {exp['status']} | {exp['metric']} | `{exp['folder']}` |\n"
        )

    # Summary section
    total = len(exps)
    success = sum(1 for e in exps if e["status_icon"] == "🟢")
    fail = total - success

    md += "\n--- \n\n## 📊 Résumé\n\n"
    md += f"**Total : {total} expériences — 🟢 {success} terminées / 🔴 {fail} en cours ou échouées**\n\n"

    # Breakdown by dataset
    md += "### Par dataset\n\n"
    md += "| Dataset | Total | 🟢 Succès | 🔴 Échec |\n"
    md += "|---------|-------|-----------|----------|\n"
    for ds_name in datasets:
        ds_exps = [e for e in exps if e["dataset"] == ds_name]
        ds_ok = sum(1 for e in ds_exps if e["status_icon"] == "🟢")
        md += f"| {ds_name} | {len(ds_exps)} | {ds_ok} | {len(ds_exps) - ds_ok} |\n"

    # Breakdown by model
    md += "\n### Par modèle\n\n"
    md += "| Modèle | Type | Total | 🟢 Succès | 🔴 Échec |\n"
    md += "|--------|------|-------|-----------|----------|\n"
    all_models = sorted({e["model"] for e in exps})
    for model in all_models:
        m_exps = [e for e in exps if e["model"] == model]
        m_ok = sum(1 for e in m_exps if e["status_icon"] == "🟢")
        m_type = "Supervisé" if model in SUPERVISED_MODELS else "Non-supervisé"
        md += f"| {model} | {m_type} | {len(m_exps)} | {m_ok} | {len(m_exps) - m_ok} |\n"

    return md


if __name__ == "__main__":
    exps = parse_experiments()
    md = generate_markdown(exps)
    with open(OUTPUT_FILE, "w") as f:
        f.write(md)
    success = sum(1 for e in exps if e["status_icon"] == "🟢")
    print(f"Generated successfully: {len(exps)} experiments ({success} ✅, {len(exps)-success} ❌)")
    print(f"Output: {OUTPUT_FILE}")
