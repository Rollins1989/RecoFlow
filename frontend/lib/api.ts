const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export type Recommendation = {
  item_id: number;
  score: number;
  reason: string;
  candidate_sources: string[];
};

export type RecommendationsResponse = {
  user_id: number;
  recommendations: Recommendation[];
  model_version: string;
  cache_hit: boolean;
  generated_at: string;
};

export async function getRecommendations(userId: number, limit = 10): Promise<RecommendationsResponse> {
  const res = await fetch(`${API_URL}/recommendations/${userId}?limit=${limit}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed to fetch recommendations: ${res.status}`);
  return res.json();
}

export async function getSimilar(itemId: number, limit = 10) {
  const res = await fetch(`${API_URL}/similar/${itemId}?limit=${limit}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed to fetch similar items: ${res.status}`);
  return res.json();
}

export async function postEvent(payload: {
  user_id: number; item_id: number; event: string; session_id?: string; device?: string;
}) {
  const res = await fetch(`${API_URL}/events`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(`Failed to post event: ${res.status}`);
  return res.json();
}

export async function search(query: string, userId?: number, limit = 20) {
  const res = await fetch(`${API_URL}/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, user_id: userId, limit }),
  });
  if (!res.ok) throw new Error(`Search failed: ${res.status}`);
  return res.json();
}

export async function getUserProfile(userId: number) {
  const res = await fetch(`${API_URL}/user/${userId}/profile`, { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed to fetch profile: ${res.status}`);
  return res.json();
}

export async function getModelInfo() {
  const res = await fetch(`${API_URL}/model/info`, { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed to fetch model info: ${res.status}`);
  return res.json();
}
