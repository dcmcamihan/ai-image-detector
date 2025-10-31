import { API_BASE_URL } from "./config";

export type PredictResponse = {
  id?: number;
  prediction: "ai" | "nature" | string;
  probabilities: { ai: number; nature: number };
  confidence?: number;
  filename: string;
  image_url?: string; // server relative path like /images/<file>
  created_at?: string;
};

export async function predictImage(file: File, opts: { tta?: boolean } = {}): Promise<PredictResponse> {
  const form = new FormData();
  form.append("file", file, file.name);

  const params = new URLSearchParams();
  if (opts.tta) params.set("tta", "1");

  const url = `${API_BASE_URL}/predict${params.toString() ? `?${params.toString()}` : ""}`;

  const res = await fetch(url, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const msg = await res.text().catch(() => res.statusText);
    throw new Error(`Predict failed (${res.status}): ${msg}`);
  }

  return res.json();
}

export async function postFeedback(id: number, payload: { ground_truth_label?: "ai" | "nature"; generator_label?: string }) {
  const res = await fetch(`${API_BASE_URL}/feedback/${id}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(`Feedback failed (${res.status})`);
  return res.json();
}

export type StatsResponse = {
  total: number;
  ai_rate: number; // 0..1
  avg_confidence: number; // 0..1
  recent_activity: number;
  accuracy_over_time: { date: string; accuracy: number }[]; // accuracy in %
  distribution: Record<string, number>; // { ai: n, nature: n }
};

export async function getStats(): Promise<StatsResponse> {
  const res = await fetch(`${API_BASE_URL}/stats`);
  if (!res.ok) throw new Error(`Stats failed (${res.status})`);
  return res.json();
}

export async function getAdvancedStats(): Promise<{
  daily_confidence: { date: string; avg_conf: number }[];
  top_uncertain: { id: number; filename: string; image_url: string; confidence: number }[];
  per_source: Record<string, number>;
}> {
  const res = await fetch(`${API_BASE_URL}/stats/advanced`);
  if (!res.ok) throw new Error(`Advanced stats failed (${res.status})`);
  return res.json();
}

export type PredictionItem = {
  id: number;
  created_at: string;
  filename: string;
  image_url: string; // server relative
  label: "AI" | "Real";
  confidence: number; // in %
  source: string;
};

export type PredictionsResponse = {
  total: number;
  page: number;
  page_size: number;
  items: PredictionItem[];
};

export async function getPredictions(params: {
  page?: number;
  page_size?: number;
  label?: "ai" | "real" | "all";
  q?: string;
  sort?: "newest" | "oldest";
} = {}): Promise<PredictionsResponse> {
  const { page = 1, page_size = 24, label = "all", q, sort = "newest" } = params;
  const search = new URLSearchParams();
  search.set("page", String(page));
  search.set("page_size", String(page_size));
  if (label !== "all") search.set("label", label);
  if (q) search.set("q", q);
  search.set("sort", sort);

  const res = await fetch(`${API_BASE_URL}/predictions?${search.toString()}`);
  if (!res.ok) throw new Error(`Predictions failed (${res.status})`);
  return res.json();
}

export async function getPrediction(id: number): Promise<{
  id: number;
  created_at: string;
  filename: string;
  image_url: string;
  mime_type: string;
  width: number;
  height: number;
  prediction: string;
  probabilities: { ai: number; nature: number };
  confidence: number;
  source: string;
}> {
  const res = await fetch(`${API_BASE_URL}/predictions/${id}`);
  if (!res.ok) throw new Error(`Get prediction failed (${res.status})`);
  return res.json();
}
