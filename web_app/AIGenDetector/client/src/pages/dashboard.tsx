import { useEffect, useState } from "react";
import { StatCard } from "@/components/stat-card";
import { AccuracyChart } from "@/components/accuracy-chart";
import { DistributionChart } from "@/components/distribution-chart";
import { Image, BarChart3, TrendingUp, Activity } from "lucide-react";
import { getStats, StatsResponse, getPredictions, PredictionItem, getPrediction } from "@/lib/api";
import { API_BASE_URL } from "@/lib/config";

export default function Dashboard() {
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [recent, setRecent] = useState<(PredictionItem & { generator_json?: Record<string, number> | null })[]>([]);

  useEffect(() => {
    setLoading(true);
    getStats()
      .then((s) => setStats(s))
      .finally(() => setLoading(false));
    // Load recent items and enrich AI items with generator_json
    getPredictions({ page: 1, page_size: 9, sort: "newest" })
      .then(async (res) => {
        const items = res.items;
        const enriched = await Promise.all(
          items.map(async (it) => {
            if (it.label === "AI") {
              try {
                const d = await getPrediction(it.id);
                return { ...it, generator_json: (d as any).generator_json ?? null };
              } catch {
                return { ...it, generator_json: null };
              }
            }
            return it as any;
          })
        );
        setRecent(enriched);
      })
      .catch(() => setRecent([]));
  }, []);

  const total = stats?.total ?? 0;
  const aiRate = stats ? Math.round(stats.ai_rate * 100) : 0;
  const avgConf = stats ? Math.round(stats.avg_confidence * 100) : 0;
  const recentCount = stats?.recent_activity ?? 0;
  const accuracySeries = stats?.accuracy_over_time ?? [];
  const distData = (() => {
    if (!stats?.distribution) return [] as { name: string; value: number }[];
    const ai = stats.distribution["ai"] ?? 0;
    const real = stats.distribution["nature"] ?? 0;
    return [
      { name: "AI-Generated", value: ai },
      { name: "Real", value: real },
    ];
  })();

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-foreground mb-2">Welcome back!</h1>
        <p className="text-muted-foreground">
          {new Date().toLocaleDateString("en-US", {
            weekday: "long",
            year: "numeric",
            month: "long",
            day: "numeric",
          })}
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard title="Total Images Analyzed" value={total.toLocaleString()} icon={Image} />
        <StatCard title="AI Detection Rate" value={`${aiRate}%`} icon={BarChart3} />
        <StatCard title="Avg. Confidence Score" value={`${avgConf}%`} icon={TrendingUp} />
        <StatCard title="Recent Activity" value={String(recentCount)} icon={Activity} />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <AccuracyChart data={accuracySeries} />
        <DistributionChart data={distData} />
      </div>

      {/* Recent predictions */}
      <div className="space-y-4">
        <h3 className="text-lg font-semibold text-foreground">Recent</h3>
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4">
          {recent.map((it) => {
            const imgUrl = it.image_url.startsWith("http") ? it.image_url : `${API_BASE_URL}${it.image_url}`;
            const isAI = it.label === "AI";
            const gen = (it as any).generator_json as Record<string, number> | undefined;
            let chips: { name: string; v: number; isTop: boolean }[] = [];
            if (isAI && gen && Object.keys(gen).length > 0) {
              const entries = Object.entries(gen).sort((a, b) => (b[1] as number) - (a[1] as number));
              const maxVal = entries.length ? (entries[0][1] as number) : 0;
              chips = entries.map(([name, v]) => ({ name, v, isTop: Math.abs((v as number) - maxVal) < 1e-6 }));
            }
            return (
              <div key={it.id} className="p-3 border rounded-lg space-y-2 hover:shadow-lg hover:-translate-y-0.5 transition-all duration-200 bg-background">
                <img src={imgUrl} alt={it.filename} className="w-full h-28 object-cover rounded-md" />
                <div className="flex items-center justify-between text-xs">
                  <span className={isAI ? "text-destructive" : "text-foreground"}>{isAI ? "AI-Generated" : "Real"}</span>
                  <span className="text-muted-foreground">{Math.round(it.confidence)}%</span>
                </div>
                {isAI && chips.length > 0 && (
                  <div className="flex flex-wrap gap-1.5 text-[10px]">
                    {chips.map((c) => (
                      <span
                        key={c.name}
                        className={
                          "px-2 py-0.5 rounded-full border " +
                          (c.isTop ? "bg-destructive/10 text-destructive border-destructive/30" : "bg-muted text-foreground border-transparent")
                        }
                        title={`${Math.round(c.v * 100)}%`}
                      >
                        {c.name} · {Math.round(c.v * 100)}%
                      </span>
                    ))}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
