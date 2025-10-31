import { useState } from "react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ConfidenceBar } from "./confidence-bar";
import { Bot, CheckCircle2, RotateCcw } from "lucide-react";
import { postFeedback } from "@/lib/api";

interface DetectionResultProps {
  id?: number;
  imageUrl: string;
  label: "AI-Generated" | "Real";
  confidence: number;
  probabilities?: {
    ai: number;
    real: number;
  };
  meta?: {
    model_version?: { model_name?: string; checksum?: string; head?: string };
    inference_time_ms?: number;
    mime_type?: string;
    width?: number;
    height?: number;
    file_size?: number;
    blur_score?: number;
    generator_json?: Record<string, number> | null;
  };
  threshold?: number; // 0..1
  onAnalyzeAnother: () => void;
}

export function DetectionResult({
  id,
  imageUrl,
  label,
  confidence,
  probabilities = { ai: 0, real: 0 },
  meta,
  threshold = 0.6,
  onAnalyzeAnother,
}: DetectionResultProps) {
  const isAI = label === "AI-Generated";
  const [showDetails, setShowDetails] = useState(false);
  const maxProb = Math.max(probabilities.ai, probabilities.real) / 100;
  const uncertain = maxProb < threshold;

  const palette = [
    "hsl(var(--chart-1))",
    "hsl(var(--chart-2))",
    "hsl(var(--chart-3))",
    "hsl(var(--chart-4))",
    "hsl(var(--chart-5))",
    "hsl(var(--muted-foreground))",
    "hsl(var(--primary))",
  ];

  return (
    <Card className="p-8 space-y-6">
      <div className="relative rounded-lg overflow-hidden max-h-64 flex items-center justify-center bg-muted">
        <img
          src={imageUrl}
          alt="Analyzed"
          className="max-h-64 object-contain"
          data-testid="img-analyzed"
        />
      </div>

      <div className="space-y-4">
        <div className="flex items-center gap-3">
          {isAI ? (
            <Bot className="w-6 h-6 text-destructive" />
          ) : (
            <CheckCircle2 className="w-6 h-6 text-chart-4" />
          )}
          <Badge
            variant={isAI ? "destructive" : "default"}
            className="text-lg px-4 py-2 rounded-full"
            data-testid="badge-result"
          >
            {uncertain ? `${label} (Uncertain)` : label}
          </Badge>
        </div>

        <ConfidenceBar confidence={confidence} label={label} />

        {probabilities.ai > 0 && (
          <div className="pt-4 space-y-2">
            <p className="text-sm font-medium text-muted-foreground">
              Probability Distribution
            </p>
            <div className="space-y-2">
              <div className="flex items-center justify-between text-sm">
                <span className="text-foreground">AI-Generated</span>
                <span className="font-medium text-muted-foreground">
                  {probabilities.ai}%
                </span>
              </div>
              <div className="flex items-center justify-between text-sm">
                <span className="text-foreground">Real</span>
                <span className="font-medium text-muted-foreground">
                  {probabilities.real}%
                </span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Generator breakdown (7 models, sorted desc, colored) - only for AI */}
      {isAI && meta?.generator_json && Object.keys(meta.generator_json).length > 0 && (
        <div className="pt-4 space-y-2">
          <p className="text-sm font-medium text-muted-foreground">Generator Likelihoods</p>
          <div className="space-y-2">
            {Object.entries(meta.generator_json)
              .sort((a, b) => (b[1] as number) - (a[1] as number))
              .slice(0, 7)
              .map(([name, val], idx) => {
                const pct = Math.max(0, Math.min(100, Math.round((val as number) * 100)));
                const color = palette[idx % palette.length];
                return (
                  <div key={name} className="space-y-1">
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-foreground">{name}</span>
                      <span className="font-medium text-muted-foreground">{pct}%</span>
                    </div>
                    <div className="h-2 w-full bg-muted rounded">
                      <div className="h-2 rounded" style={{ width: `${pct}%`, backgroundColor: color }} />
                    </div>
                  </div>
                );
              })}
          </div>
          {/* Dynamic attribution chips: show all generators with %; highlight all max-tied top generators */}
          {showDetails && isAI && (
            <div className="pt-3">
              <p className="text-sm font-medium text-muted-foreground mb-2">Generators (all)</p>
              {(() => {
                const entries = Object.entries(meta.generator_json as Record<string, number>)
                  .sort((a, b) => (b[1] as number) - (a[1] as number));
                const maxVal = entries.length ? (entries[0][1] as number) : 0;
                return (
                  <div className="flex flex-wrap gap-2 text-xs">
                    {entries.map(([name, v]) => {
                      const isTop = Math.abs((v as number) - maxVal) < 1e-6;
                      const pct = Math.round((v as number) * 100);
                      return (
                        <span
                          key={name}
                          className={
                            "px-2 py-1 rounded-full border " +
                            (isTop ? "bg-destructive/10 text-destructive border-destructive/30" : "bg-muted text-foreground border-transparent")
                          }
                          title={`${pct}%`}
                        >
                          {name} · {pct}%
                        </span>
                      );
                    })}
                  </div>
                );
              })()}
            </div>
          )}
        </div>
      )}

      {/* Metadata & model version (hidden by default; placed after generator section) */}
      {meta && showDetails && (
        <div className="grid grid-cols-2 gap-4 text-sm">
          {meta.model_version?.checksum && (
            <div>
              <p className="text-muted-foreground">Model</p>
              <p className="font-medium text-foreground">
                {meta.model_version.model_name ?? "model"} · {meta.model_version.checksum}
              </p>
            </div>
          )}
          {typeof meta.inference_time_ms === "number" && (
            <div>
              <p className="text-muted-foreground">Inference Time</p>
              <p className="font-medium text-foreground">{meta.inference_time_ms.toFixed(1)} ms</p>
            </div>
          )}
          {(meta.mime_type || meta.width || meta.height) && (
            <div>
              <p className="text-muted-foreground">Image</p>
              <p className="font-medium text-foreground">
                {meta.mime_type ?? "image"} {meta.width && meta.height ? `· ${meta.width}x${meta.height}` : ""}
              </p>
            </div>
          )}
          {(typeof meta.file_size === "number" || typeof meta.blur_score === "number") && (
            <div>
              <p className="text-muted-foreground">Quality</p>
              <p className="font-medium text-foreground">
                {typeof meta.file_size === "number" ? `${(meta.file_size / 1024).toFixed(1)} KB` : ""}
                {typeof meta.blur_score === "number" ? ` · Blur ${(meta.blur_score * 100).toFixed(1)}%` : ""}
              </p>
            </div>
          )}
        </div>
      )}

      {/* Toggle details */}
      <div className="flex justify-end">
        <Button size="sm" variant="ghost" onClick={() => setShowDetails((s) => !s)}>
          {showDetails ? "Hide details" : "Show details"}
        </Button>
      </div>

      <Button
        onClick={onAnalyzeAnother}
        variant="outline"
        className="w-full"
        data-testid="button-analyze-another"
      >
        <RotateCcw className="w-4 h-4 mr-2" />
        Analyze Another Image
      </Button>
    </Card>
  );
}
