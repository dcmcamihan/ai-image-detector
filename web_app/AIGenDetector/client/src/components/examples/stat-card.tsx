import { StatCard } from "../stat-card";
import { BarChart3, Image, TrendingUp } from "lucide-react";

export default function StatCardExample() {
  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-6 p-6 max-w-5xl">
      <StatCard
        title="Total Images Analyzed"
        value="1,247"
        icon={Image}
        trend={{ value: 12, isPositive: true }}
      />
      <StatCard
        title="AI Detection Rate"
        value="68%"
        icon={BarChart3}
        trend={{ value: 3, isPositive: false }}
      />
      <StatCard
        title="Avg. Confidence Score"
        value="94.2%"
        icon={TrendingUp}
      />
    </div>
  );
}
