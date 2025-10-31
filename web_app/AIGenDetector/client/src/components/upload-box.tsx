import { Upload, Image as ImageIcon } from "lucide-react";
import { useState, useRef } from "react";

interface UploadBoxProps {
  onFileSelect: (file: File) => void;
}

export function UploadBox({ onFileSelect }: UploadBoxProps) {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) {
      onFileSelect(file);
    }
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      onFileSelect(file);
    }
  };

  return (
    <div
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      onClick={() => fileInputRef.current?.click()}
      className={`
        min-h-96 flex flex-col items-center justify-center
        border-2 border-dashed rounded-xl p-12 cursor-pointer
        transition-all duration-200
        bg-card/70 supports-[backdrop-filter]:backdrop-blur-sm hover-elevate active-elevate-2
        ${isDragging ? "border-primary bg-primary/10" : "border-border"}
      `}
      data-testid="upload-dropzone"
    >
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        onChange={handleFileInput}
        className="hidden"
        data-testid="input-file"
      />
      <div className="flex items-center justify-center w-16 h-16 rounded-full bg-primary/10 mb-6 shadow-xs">
        {isDragging ? (
          <ImageIcon className="w-8 h-8 text-primary" />
        ) : (
          <Upload className="w-8 h-8 text-primary" />
        )}
      </div>
      <h3 className="text-xl font-semibold mb-2 text-foreground tracking-tight">
        Drag & drop your image here
      </h3>
      <p className="text-sm text-muted-foreground mb-4">or click to browse</p>
      <p className="text-xs text-muted-foreground">
        Supports: JPG, PNG, WebP (max 10MB)
      </p>
    </div>
  );
}
