import { useEffect, useState } from "react";

interface ConfidenceBarProps {
  confidence: number;
  label: "AI-Generated" | "Real";
}

export function ConfidenceBar({ confidence, label }: ConfidenceBarProps) {
  const [animatedWidth, setAnimatedWidth] = useState(0);

  useEffect(() => {
    const timer = setTimeout(() => {
      setAnimatedWidth(confidence);
    }, 100);
    return () => clearTimeout(timer);
  }, [confidence]);

  const isAI = label === "AI-Generated";

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium text-muted-foreground">Confidence</span>
        <span className="text-2xl font-display font-bold text-foreground">
          {confidence}%
        </span>
      </div>
      <div className="h-4 bg-muted rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-1000 ease-out ${
            isAI
              ? "bg-gradient-to-r from-chart-5 to-destructive"
              : "bg-gradient-to-r from-chart-4 to-chart-3"
          }`}
          style={{ width: `${animatedWidth}%` }}
        />
      </div>
    </div>
  );
}
