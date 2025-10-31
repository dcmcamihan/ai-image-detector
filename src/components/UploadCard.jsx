// web_app/src/components/UploadCard.jsx
import { useState } from "react";

export default function UploadCard() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  async function handleUpload(e) {
    e.preventDefault();
    if (!file) return;
    setLoading(true);
    const fd = new FormData();
    fd.append("file", file, file.name);

    try {
      const res = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: fd,
      });
      const data = await res.json();
      setResult(data);
    } catch (err) {
      console.error(err);
      setResult({ error: "Network or server error" });
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="max-w-xl mx-auto p-6 bg-white rounded-xl shadow-md">
      <h2 className="text-xl font-semibold mb-4">AI Image Detector — Upload</h2>

      <form onSubmit={handleUpload}>
        <input
          type="file"
          accept="image/*"
          onChange={(e) => setFile(e.target.files[0])}
          className="mb-4"
        />
        <button type="submit" disabled={!file || loading} className="px-4 py-2 bg-blue-600 text-white rounded">
          {loading ? "Analyzing..." : "Analyze"}
        </button>
      </form>

      {result && (
        <div className="mt-6">
          {result.error ? (
            <div className="text-red-600">{result.error}</div>
          ) : (
            <>
              <p><strong>Prediction:</strong> {result.prediction}</p>
              <p><strong>AI probability:</strong> {(result.probabilities.ai * 100).toFixed(2)}%</p>
              <p><strong>Nature probability:</strong> {(result.probabilities.nature * 100).toFixed(2)}%</p>
            </>
          )}
        </div>
      )}
    </div>
  );
}