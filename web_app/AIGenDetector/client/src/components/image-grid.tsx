import { useState } from "react";
import { Badge } from "@/components/ui/badge";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

interface ImageData {
  id: string;
  url: string;
  label: "AI" | "Real";
  confidence: number;
  source: string;
  size: string;
}

interface ImageGridProps {
  images: ImageData[];
  onItemClick?: (image: ImageData) => void;
}

export function ImageGrid({ images, onItemClick }: ImageGridProps) {
  const [selectedImage, setSelectedImage] = useState<ImageData | null>(null);

  return (
    <>
      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
        {images.map((image) => (
          <div
            key={image.id}
            className="relative aspect-square rounded-lg overflow-hidden cursor-pointer hover-elevate active-elevate-2 transition-transform"
            onClick={() => (onItemClick ? onItemClick(image) : setSelectedImage(image))}
            data-testid={`img-thumbnail-${image.id}`}
          >
            <img
              src={image.url}
              alt={`${image.label} image`}
              className="w-full h-full object-cover"
            />
            <Badge
              variant={image.label === "AI" ? "destructive" : "default"}
              className="absolute top-2 right-2 text-xs"
            >
              {image.label}
            </Badge>
          </div>
        ))}
      </div>

      <Dialog open={!!selectedImage} onOpenChange={() => setSelectedImage(null)}>
        <DialogContent className="max-w-2xl" data-testid="modal-image-details">
          <DialogHeader>
            <DialogTitle>Image Details</DialogTitle>
          </DialogHeader>
          {selectedImage && (
            <div className="space-y-4">
              <img
                src={selectedImage.url}
                alt="Full size"
                className="w-full rounded-lg"
              />
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <p className="text-muted-foreground">Label</p>
                  <Badge
                    variant={selectedImage.label === "AI" ? "destructive" : "default"}
                  >
                    {selectedImage.label}
                  </Badge>
                </div>
                <div>
                  <p className="text-muted-foreground">Confidence</p>
                  <p className="font-medium text-foreground">{selectedImage.confidence}%</p>
                </div>
                <div>
                  <p className="text-muted-foreground">Source</p>
                  <p className="font-medium text-foreground">{selectedImage.source}</p>
                </div>
                <div>
                  <p className="text-muted-foreground">Size</p>
                  <p className="font-medium text-foreground">{selectedImage.size}</p>
                </div>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </>
  );
}
