import { useState } from "react";
import { UploadBox } from "@/components/upload-box";
import { DetectionResult } from "@/components/detection-result";
import { predictImage } from "@/lib/api";

export default function Upload() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [imageUrl, setImageUrl] = useState<string>("");
  const [isAnalyzed, setIsAnalyzed] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<{
    id?: number;
    label: "AI-Generated" | "Real";
    confidence: number;
    probs: { ai: number; real: number };
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
  } | null>(null);

  const handleFileSelect = async (file: File) => {
    setSelectedFile(file);
    const url = URL.createObjectURL(file);
    setImageUrl(url);
    setLoading(true);
    setIsAnalyzed(false);
    setResult(null);

    try {
      const res = await predictImage(file);
      // Backend returns prediction: "ai" | "nature" and probabilities { ai, nature } in [0,1] or [0,100]
      const pred = String(res.prediction).toLowerCase();
      const aiProbRaw = res.probabilities.ai;
      const natureProbRaw = res.probabilities.nature;
      const aiProb = aiProbRaw <= 1 ? aiProbRaw * 100 : aiProbRaw;
      const realProb = natureProbRaw <= 1 ? natureProbRaw * 100 : natureProbRaw;
      const label: "AI-Generated" | "Real" = pred === "ai" ? "AI-Generated" : "Real";
      const confidence = Math.round(Math.max(aiProb, realProb));

      setResult({
        id: typeof res.id === "number" ? res.id : undefined,
        label,
        confidence,
        probs: { ai: Math.round(aiProb), real: Math.round(realProb) },
        meta: {
          model_version: (res as any).model_version,
          inference_time_ms: (res as any).inference_time_ms,
          mime_type: (res as any).mime_type,
          width: (res as any).width,
          height: (res as any).height,
          file_size: (res as any).file_size,
          blur_score: (res as any).blur_score,
          generator_json: (res as any).generator_json ?? null,
        },
      });
      setIsAnalyzed(true);
    } catch (err) {
      console.error("Prediction failed", err);
      alert("Prediction failed. Please ensure the backend is running on http://localhost:8000 and try again.");
      setSelectedFile(null);
      setImageUrl("");
      setIsAnalyzed(false);
    } finally {
      setLoading(false);
    }
  };

  const handleAnalyzeAnother = () => {
    setSelectedFile(null);
    setImageUrl("");
    setIsAnalyzed(false);
  };

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-foreground mb-2">Upload & Detect</h1>
        <p className="text-muted-foreground">
          Upload an image to analyze whether it's AI-generated or real
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <div>
          {!selectedFile ? (
            <UploadBox onFileSelect={handleFileSelect} />
          ) : (
            <div className="min-h-96 flex items-center justify-center border-2 border-dashed border-border rounded-xl p-12">
              <div className="text-center space-y-4">
                <img
                  src={imageUrl}
                  alt="Uploaded"
                  className="max-h-80 mx-auto rounded-lg"
                />
                {loading && (
                  <p className="text-sm text-muted-foreground animate-pulse">
                    Analyzing image...
                  </p>
                )}
              </div>
            </div>
          )}
        </div>

        <div>
          {isAnalyzed && result && (
            <DetectionResult
              id={result.id}
              imageUrl={imageUrl}
              label={result.label}
              confidence={result.confidence}
              probabilities={{ ai: result.probs.ai, real: result.probs.real }}
              meta={result.meta}
              onAnalyzeAnother={handleAnalyzeAnother}
            />
          )}
        </div>
      </div>
    </div>
  );
}
