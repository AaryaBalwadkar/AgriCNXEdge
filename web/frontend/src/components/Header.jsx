export function Header({ status }) {
  const modelReady = Boolean(status?.model_loaded);
  const label = modelReady ? "Model loaded" : status?.status === "offline" ? "API offline" : "Model pending";

  return (
    <header className="topbar">
      <div>
        <p className="eyebrow">AgriCNXEdge Web</p>
        <h1>Apple crop-health inference dashboard</h1>
        <p className="intro">Run the exported ADC ONNX model through a FastAPI service and inspect crop-health signals from a clean web interface.</p>
      </div>
      <span className={modelReady ? "status-pill status-ok" : "status-pill status-warn"}>{label}</span>
    </header>
  );
}
