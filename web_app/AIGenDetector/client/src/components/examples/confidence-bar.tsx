import { ConfidenceBar } from "../confidence-bar";

export default function ConfidenceBarExample() {
  return (
    <div className="p-6 space-y-8 max-w-2xl">
      <ConfidenceBar confidence={92} label="AI-Generated" />
      <ConfidenceBar confidence={87} label="Real" />
    </div>
  );
}
