import { UploadBox } from "../upload-box";

export default function UploadBoxExample() {
  return (
    <div className="p-6 max-w-2xl">
      <UploadBox
        onFileSelect={(file) => console.log("File selected:", file.name)}
      />
    </div>
  );
}
