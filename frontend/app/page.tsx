"use client";

import { useEffect, useState } from "react";
import { getRecommendations, search, type Recommendation } from "../lib/api";
import ProductRow from "../components/ProductRow";

const DEMO_USER_ID = 1;

export default function HomePage() {
  const [recs, setRecs] = useState<Recommendation[]>([]);
  const [searchResults, setSearchResults] = useState<Recommendation[] | null>(null);
  const [query, setQuery] = useState("");
  const [showDebug, setShowDebug] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getRecommendations(DEMO_USER_ID, 12)
      .then((res) => setRecs(res.recommendations))
      .catch((e) => setError(String(e)));
  }, []);

  const runSearch = async () => {
    if (!query.trim()) {
      setSearchResults(null);
      return;
    }
    try {
      const res = await search(query, DEMO_USER_ID, 12);
      setSearchResults(
        res.results.map((r: any) => ({
          item_id: r.item_id,
          score: r.score,
          reason: `Matched search: "${query}"`,
          candidate_sources: ["search"],
        }))
      );
    } catch (e) {
      setError(String(e));
    }
  };

  return (
    <main>
      <div className="topbar">
        <h1>RECOFLOW</h1>
        <input
          className="search-box"
          placeholder="Search products..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && runSearch()}
        />
        <div className="debug-toggle" onClick={() => setShowDebug((s) => !s)}>
          {showDebug ? "Debug mode: ON" : "Debug mode: OFF"}
        </div>
      </div>

      {error && <div className="row" style={{ color: "#ff6b6b" }}>{error}</div>}

      {searchResults ? (
        <ProductRow title={`Results for "${query}"`} recommendations={searchResults} userId={DEMO_USER_ID} showDebug={showDebug} />
      ) : (
        <>
          <ProductRow title="Recommended for You" recommendations={recs} userId={DEMO_USER_ID} showDebug={showDebug} />
          <ProductRow
            title="Because you viewed similar items"
            recommendations={recs.filter((r) => r.candidate_sources.includes("content"))}
            userId={DEMO_USER_ID}
            showDebug={showDebug}
          />
          <ProductRow
            title="Trending in your favorite categories"
            recommendations={recs.filter((r) => r.candidate_sources.includes("trending"))}
            userId={DEMO_USER_ID}
            showDebug={showDebug}
          />
        </>
      )}
    </main>
  );
}
