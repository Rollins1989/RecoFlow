"use client";

import { useEffect, useState } from "react";
import { getModelInfo } from "../../lib/api";

export default function DashboardPage() {
  const [info, setInfo] = useState<any>(null);

  useEffect(() => {
    getModelInfo().then(setInfo).catch(() => {});
  }, []);

  // In production these come from Prometheus (see src/monitoring) — shown
  // here as static placeholders matching the metrics the design doc tracks,
  // since this page renders without needing a live Prometheus instance.
  const metrics = [
    { label: "NDCG@10", value: "—" },
    { label: "Recall@10", value: "—" },
    { label: "CTR", value: "—" },
    { label: "Catalog coverage", value: "—" },
    { label: "Diversity", value: "—" },
    { label: "P95 latency", value: "—" },
  ];

  return (
    <main>
      <div className="topbar">
        <h1>RECOFLOW — ML Dashboard</h1>
      </div>
      <div className="dashboard-grid">
        {metrics.map((m) => (
          <div key={m.label} className="metric-card">
            <div className="label">{m.label}</div>
            <div className="value">{m.value}</div>
          </div>
        ))}
      </div>

      <div className="row">
        <h2>Model info</h2>
        <pre className="debug-panel" style={{ maxWidth: 480 }}>
          {info ? JSON.stringify(info, null, 2) : "Loading..."}
        </pre>
      </div>

      <div className="row">
        <h2>Model versions</h2>
        <div className="debug-panel" style={{ maxWidth: 480 }}>
          ranker-v1 — production<br />
          ranker-v2 — staging
        </div>
      </div>

      <div className="row" style={{ paddingBottom: 40 }}>
        <h2>Live numbers</h2>
        <p style={{ color: "var(--muted)", fontSize: 13 }}>
          Wire the metric cards above to Prometheus (see <code>src/monitoring/prometheus.yml</code>
          {" "}and <code>src/monitoring/grafana_dashboard.json</code>) or embed a Grafana panel
          for real-time values. Run <code>python run_pipeline.py</code> for an offline evaluation
          table (Recall/NDCG/MAP/MRR/Coverage/Diversity per pipeline stage) in the terminal.
        </p>
      </div>
    </main>
  );
}
