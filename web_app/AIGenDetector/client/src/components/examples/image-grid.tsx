import { ImageGrid } from "../image-grid";

const mockImages = [
  {
    id: "1",
    url: "https://images.unsplash.com/photo-1677442136019-21780ecad995?w=300",
    label: "AI" as const,
    confidence: 92,
    source: "GenImage",
    size: "512x512",
  },
  {
    id: "2",
    url: "https://images.unsplash.com/photo-1506748686214-e9df14d4d9d0?w=300",
    label: "Real" as const,
    confidence: 88,
    source: "Kaggle",
    size: "1024x1024",
  },
  {
    id: "3",
    url: "https://images.unsplash.com/photo-1469474968028-56623f02e42e?w=300",
    label: "Real" as const,
    confidence: 95,
    source: "Kaggle",
    size: "800x600",
  },
  {
    id: "4",
    url: "https://images.unsplash.com/photo-1682687220063-4742bd7fd538?w=300",
    label: "AI" as const,
    confidence: 89,
    source: "GenImage",
    size: "512x512",
  },
];

export default function ImageGridExample() {
  return (
    <div className="p-6 max-w-5xl">
      <ImageGrid images={mockImages} />
    </div>
  );
}
