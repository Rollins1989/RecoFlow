"use client";

import { useEffect, useState } from "react";
import type { Recommendation } from "../lib/api";
import { postEvent } from "../lib/api";
import DebugPanel from "./DebugPanel";

export default function ProductRow({
  title,
  recommendations,
  userId,
  showDebug,
}: {
  title: string;
  recommendations: Recommendation[];
  userId: number;
  showDebug: boolean;
}) {
  const handleClick = (itemId: number) => {
    postEvent({ user_id: userId, item_id: itemId, event: "click" }).catch(() => {});
  };

  return (
    <div className="row">
      <h2>{title}</h2>
      <div className="card-grid">
        {recommendations.map((rec) => (
          <div key={rec.item_id} className="product-card" onClick={() => handleClick(rec.item_id)}>
            <div className="thumb">#{rec.item_id}</div>
            <div className="title">Product {rec.item_id}</div>
            <div className="score">{rec.reason}</div>
            {showDebug && <DebugPanel rec={rec} />}
          </div>
        ))}
      </div>
    </div>
  );
}
