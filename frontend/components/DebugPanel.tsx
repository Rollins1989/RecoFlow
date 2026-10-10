"use client";

import { useState } from "react";
import type { Recommendation } from "../lib/api";

export default function DebugPanel({ rec }: { rec: Recommendation }) {
  const [open, setOpen] = useState(false);
  return (
    <div>
      <button type="button" className="debug-toggle" aria-expanded={open} onClick={() => setOpen((o) => !o)}>
        {open ? "Hide debug" : "Show debug"}
      </button>
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
