"use client";

import { useState } from "react";
import type { Recommendation } from "../lib/api";

export default function DebugPanel({ rec }: { rec: Recommendation }) {
  const [open, setOpen] = useState(false);
  return (
    <div>
      <div className="debug-toggle" onClick={() => setOpen((o) => !o)}>
        {open ? "Hide debug ▲" : "Debug ▼"}
      </div>
      {open && (
        <div className="debug-panel">
          <div><b>Candidate source:</b> {rec.candidate_sources.join(", ") || "n/a"}</div>
          <div><b>Final score:</b> {rec.score.toFixed(3)}</div>
          <div><b>Reason:</b> {rec.reason}</div>
        </div>
      )}
    </div>
  );
}
