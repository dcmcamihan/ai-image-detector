import { DetectionResult } from "../detection-result";

export default function DetectionResultExample() {
  return (
    <div className="p-6 max-w-2xl">
      <DetectionResult
        imageUrl="https://images.unsplash.com/photo-1677442136019-21780ecad995?w=400"
        label="AI-Generated"
        confidence={92}
        probabilities={{ ai: 92, real: 8 }}
        onAnalyzeAnother={() => console.log("Analyze another clicked")}
      />
    </div>
  );
}
