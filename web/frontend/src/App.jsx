import { useEffect, useState } from "react";
import "./App.css";
import { CapabilityGrid } from "./components/CapabilityGrid.jsx";
import { Header } from "./components/Header.jsx";
import { PredictionPanel } from "./components/PredictionPanel.jsx";
import { UploadPanel } from "./components/UploadPanel.jsx";
import { getHealth, predictImage } from "./lib/api.js";

const SAMPLES = ["apple_00037.jpg", "apple_00448.jpg", "apple_00533.jpg"];

export default function App() {
  const [status, setStatus] = useState(null);
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;

    getHealth()
      .then((data) => {
        if (active) setStatus(data);
      })
      .catch(() => {
        if (active) setStatus({ status: "offline", model_loaded: false });
      });

    return () => {
      active = false;
      if (preview) URL.revokeObjectURL(preview);
    };
  }, []);

  function setSelectedFile(selectedFile) {
    if (preview) URL.revokeObjectURL(preview);
    setFile(selectedFile);
    setPreview(selectedFile ? URL.createObjectURL(selectedFile) : null);
    setResult(null);
    setError("");
  }

  function handleFileChange(event) {
    setSelectedFile(event.target.files?.[0] || null);
  }

  async function handleSampleSelect(name) {
    try {
      const response = await fetch(`/samples/${name}`);
      if (!response.ok) throw new Error(`Sample image not found: ${name}`);
      const blob = await response.blob();
      setSelectedFile(new File([blob], name, { type: blob.type || "image/jpeg" }));
    } catch (err) {
      setError(err.message);
    }
  }

  async function handleSubmit() {
    if (!file) return;

    setLoading(true);
    setError("");

    try {
      setResult(await predictImage(file));
    } catch (err) {
      setResult(null);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="app-shell">
      <Header status={status} />
      <CapabilityGrid />
      <section className="workspace" aria-label="Model inference workspace">
        <UploadPanel
          error={error}
          file={file}
          loading={loading}
          onFileChange={handleFileChange}
          onSampleSelect={handleSampleSelect}
          onSubmit={handleSubmit}
          preview={preview}
          samples={SAMPLES}
        />
        <PredictionPanel result={result} />
      </section>
    </main>
  );
}
