import { useEffect, useState } from "react";
import { ImageGrid } from "@/components/image-grid";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { DetectionResult } from "@/components/detection-result";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Search } from "lucide-react";
import { getPredictions, PredictionItem, getPrediction } from "@/lib/api";
import { API_BASE_URL } from "@/lib/config";

export default function Dataset() {
  const [filter, setFilter] = useState("all"); // all | ai | real
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(1);
  const [pageSize] = useState(24);
  const [total, setTotal] = useState(0);
  const [items, setItems] = useState<PredictionItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [detail, setDetail] = useState<any | null>(null);
  const [detailOpen, setDetailOpen] = useState(false);
  const threshold = 0.6;

  useEffect(() => {
    let isMounted = true;
    setLoading(true);
    getPredictions({ page, page_size: pageSize, label: filter as any, q: search })
      .then((res) => {
        if (!isMounted) return;
        setTotal(res.total);
        setItems(res.items);
      })
      .finally(() => setLoading(false));
    return () => {
      isMounted = false;
    };
  }, [page, pageSize, filter, search]);

  const handleOpenDetails = async (img: { id: string }) => {
    const id = Number(img.id);
    setSelectedId(id);
    setDetailOpen(true);
    try {
      const d = await getPrediction(id);
      setDetail(d);
    } catch (e) {
      console.error(e);
    }
  };

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-foreground mb-2">Dataset Viewer</h1>
        <p className="text-muted-foreground">
          Explore and filter images from the training dataset
        </p>
      </div>

      <div className="flex flex-col sm:flex-row gap-4">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
          <Input
            placeholder="Search images..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="pl-10"
            data-testid="input-search"
          />
        </div>
        <Select value={filter} onValueChange={setFilter}>
          <SelectTrigger className="w-full sm:w-48" data-testid="select-filter">
            <SelectValue placeholder="Filter by class" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Images</SelectItem>
            <SelectItem value="ai">AI-Generated</SelectItem>
            <SelectItem value="real">Real</SelectItem>
          </SelectContent>
        </Select>
      </div>

      <ImageGrid
        images={items.map((it) => ({
          id: String(it.id),
          url: it.image_url.startsWith("http") ? it.image_url : `${API_BASE_URL}${it.image_url}`,
          label: it.label,
          confidence: Math.round(it.confidence),
          source: it.source,
          size: "224x224",
        }))}
        onItemClick={handleOpenDetails}
      />

      <div className="flex items-center justify-between mt-6">
        <div className="text-sm text-muted-foreground">
          Showing {items.length} of {total} images
        </div>
        <div className="flex gap-2">
          <button
            className="px-3 py-1 border rounded-md disabled:opacity-50"
            onClick={() => setPage((p) => Math.max(1, p - 1))}
            disabled={page <= 1 || loading}
          >
            Prev
          </button>
          <span className="text-sm text-foreground">Page {page}</span>
          <button
            className="px-3 py-1 border rounded-md disabled:opacity-50"
            onClick={() => setPage((p) => (p * pageSize < total ? p + 1 : p))}
            disabled={page * pageSize >= total || loading}
          >
            Next
          </button>
        </div>
      </div>

      <div className="text-center text-sm text-muted-foreground py-8">Total stored images: {total}</div>

      <Dialog open={detailOpen} onOpenChange={(o) => { setDetailOpen(o); if (!o) { setSelectedId(null); setDetail(null); } }}>
        <DialogContent className="max-w-3xl max-h-[85vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Prediction Details</DialogTitle>
          </DialogHeader>
          {!detail && (
            <div className="space-y-4">
              <div className="h-64 w-full bg-muted animate-pulse rounded" />
              <div className="h-4 w-1/3 bg-muted animate-pulse rounded" />
              <div className="h-4 w-1/2 bg-muted animate-pulse rounded" />
              <div className="h-4 w-2/3 bg-muted animate-pulse rounded" />
            </div>
          )}
          {detail && (
            <DetectionResult
              id={detail.id}
              imageUrl={detail.image_url.startsWith("http") ? detail.image_url : `${API_BASE_URL}${detail.image_url}`}
              label={String(detail.prediction).toLowerCase() === "ai" ? "AI-Generated" : "Real"}
              confidence={Math.round(Math.max(detail.probabilities.ai, detail.probabilities.nature) * (detail.probabilities.ai <= 1 ? 100 : 1))}
              probabilities={{
                ai: Math.round((detail.probabilities.ai <= 1 ? detail.probabilities.ai * 100 : detail.probabilities.ai)),
                real: Math.round((detail.probabilities.nature <= 1 ? detail.probabilities.nature * 100 : detail.probabilities.nature)),
              }}
              meta={{
                model_version: detail.model_version,
                inference_time_ms: detail.inference_time_ms,
                mime_type: detail.mime_type,
                width: detail.width,
                height: detail.height,
                file_size: detail.file_size,
                blur_score: detail.blur_score,
                generator_json: detail.generator_json ?? null,
              }}
              threshold={threshold}
              onAnalyzeAnother={() => setDetailOpen(false)}
            />
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}
